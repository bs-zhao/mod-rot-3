"""E6 step 1 - train K decoders on the FROZEN subject latents (S5 posterior means) using the OBSERVED choices only.
Same architecture as the NP decoder ([x, z] -> 128,256,128 -> 2 logits, GELU).  The encoder / Z are not touched.
Optional rotation-invariant smoothness penalty  lam * E ||grad_z logit||^2  (invariant to orthogonal changes of z).
python train.py MODEL REPEAT K [LAM=0] [EPOCHS=30] [TAG=base]"""
import sys, time, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "three_gaps"))
from common import io
BETA = {"CR3": (0.00005, 4), "C1": (0.0005, 4), "C2": (0.0005, 4), "C3": (0.0005, 6)}
model, rep, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
lam = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 30
tag = sys.argv[6] if len(sys.argv) > 6 else "base"
dev = io.pick_device("auto")
beta, zd = BETA[model]
data, truth = io.load_all(model, beta, zd, 0, rep, need_checkpoint=False)
cond = data["condition"]
out = ROOT / "outputs" / "x6_plan06" / "e06" / f"{model}_{cond}_{tag}"; out.mkdir(parents=True, exist_ok=True)
z = torch.tensor(data["z"], dtype=torch.float32, device=dev)
si = torch.tensor(data["subject_index"], dtype=torch.long, device=dev)
x = torch.tensor(data["x"], dtype=torch.float32, device=dev)
y = torch.tensor(np.asarray(data["y"]).reshape(-1), dtype=torch.long, device=dev)
tr = torch.zeros(len(z), dtype=torch.bool, device=dev); tr[torch.tensor(data["train_ids"], device=dev)] = True
va = torch.zeros(len(z), dtype=torch.bool, device=dev); va[torch.tensor(data["val_ids"], device=dev)] = True
rows_tr, rows_va = torch.nonzero(tr[si]).squeeze(1), torch.nonzero(va[si]).squeeze(1)
p, d = x.shape[1], z.shape[1]


def make():
    layers, dims = [], [p + d, 128, 256, 128, 2]
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2: layers.append(nn.GELU())
    m = nn.Module(); m.decoder = nn.Sequential(*layers); return m.to(dev)


def batch_ce(net, rows, grad_pen=False):
    inp = torch.cat([x[rows], z[si[rows]]], 1)
    if grad_pen: inp = inp.requires_grad_(True)
    lg = net.decoder(inp); ce = nn.functional.cross_entropy(lg, y[rows])
    if not grad_pen: return ce, None
    eta = lg[:, 1] - lg[:, 0]
    g = torch.autograd.grad(eta.sum(), inp, create_graph=True)[0][:, p:]       # d eta / d z
    return ce, (g ** 2).sum(1).mean()


log = []
for k in range(K):
    torch.manual_seed(1000 + k); net = make()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, bad, t0 = 1e9, 0, time.time()
    for ep in range(epochs):
        perm = rows_tr[torch.randperm(len(rows_tr), device=dev)]
        for i in range(0, len(perm), 8192):
            rows = perm[i:i + 8192]
            ce, gp = batch_ce(net, rows, grad_pen=lam > 0)
            loss = ce + (lam * gp if lam > 0 else 0.0)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        with torch.no_grad():
            vce = float(np.mean([float(batch_ce(net, rows_va[i:i + 65536])[0]) for i in range(0, len(rows_va), 65536)]))
        if vce < best - 1e-5: best, bad = vce, 0; torch.save(net.decoder.state_dict(), out / f"dec_{k}.pt")
        else:
            bad += 1
            if bad >= 4: break
    log.append(dict(k=k, best_val_ce=best, epochs=ep + 1, secs=time.time() - t0))
    print(f"{model} rep{rep} {tag} decoder {k}: best val CE {best:.4f} ({ep+1} epochs, {time.time()-t0:.0f}s)", flush=True)
json.dump(dict(model=model, rep=rep, K=K, lam=lam, epochs=epochs, tag=tag, log=log), open(out / "train_log.json", "w"), indent=1)
