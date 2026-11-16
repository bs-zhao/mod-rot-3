"""Step 3b - decoders for PROPER cross-fitting: frozen latents, observed choices, DISJOINT halves of the TRAIN subjects.

The e06 ensemble (x6_plan06) used the same TRAIN subjects for every decoder, so its decoders share the sampling noise of the
choices.  Here the TRAIN subjects are split into `halves` disjoint groups (seed fixed) and `seeds` decoders are trained per group
(same architecture / optimiser / early stopping on VAL as e06/train.py: [x, z] -> 128, 256, 128 -> 2 logits, GELU).  Decoders of
different groups have independent errors, both from the sample of choices and from the optimisation.
Then the derivative fields (g, H_UU, H_UX; logit and prob channels) are cached on the EXP13 grid (same 300 subjects x stimuli).

python cld-crossfit-v3/step3_gap3/train_halves.py MODEL REPEAT [HALVES=2] [SEEDS=5] [EPOCHS=30]
"""
import sys, time, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "common"))
import xfit as XF
from xfit import X, io
from common.decoder import decoder_h
from common.derivs import base_derivatives

model, rep = sys.argv[1], int(sys.argv[2])
halves = int(sys.argv[3]) if len(sys.argv) > 3 else 2
seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 5
epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 30
dev = io.pick_device("auto")
data, truth = io.load_all(model, *XF.BETA[model], 0, rep, need_checkpoint=False)
out = XF.OUT / "dec" / f"{model}_{data['condition']}_h{halves}"; out.mkdir(parents=True, exist_ok=True)
z = torch.tensor(data["z"], dtype=torch.float32, device=dev)
si = torch.tensor(data["subject_index"], dtype=torch.long, device=dev)
x = torch.tensor(data["x"], dtype=torch.float32, device=dev)
y = torch.tensor(np.asarray(data["y"]).reshape(-1), dtype=torch.long, device=dev)
p, d = x.shape[1], z.shape[1]
folds = XF.split_folds(data["train_ids"], halves, seed=20261115)
va = torch.zeros(len(z), dtype=torch.bool, device=dev); va[torch.tensor(data["val_ids"], device=dev)] = True
rows_va = torch.nonzero(va[si]).squeeze(1)


def make():
    layers, dims = [], [p + d, 128, 256, 128, 2]
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2: layers.append(nn.GELU())
    m = nn.Module(); m.decoder = nn.Sequential(*layers); return m.to(dev)


def ce(net, rows):
    return nn.functional.cross_entropy(net.decoder(torch.cat([x[rows], z[si[rows]]], 1)), y[rows])


log = []
for h, ids in enumerate(folds):
    tr = torch.zeros(len(z), dtype=torch.bool, device=dev); tr[torch.tensor(ids, device=dev)] = True
    rows_tr = torch.nonzero(tr[si]).squeeze(1)
    for k in range(seeds):
        fp = out / f"half{h}_seed{k}.pt"
        if fp.exists(): continue
        torch.manual_seed(5000 + 100 * h + k); net = make()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best, bad, t0 = 1e9, 0, time.time()
        for ep in range(epochs):
            perm = rows_tr[torch.randperm(len(rows_tr), device=dev)]
            for i in range(0, len(perm), 8192):
                loss = ce(net, perm[i:i + 8192]); opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            with torch.no_grad():
                vce = float(np.mean([float(ce(net, rows_va[i:i + 65536])) for i in range(0, len(rows_va), 65536)]))
            if vce < best - 1e-5: best, bad = vce, 0; torch.save(net.decoder.state_dict(), fp)
            else:
                bad += 1
                if bad >= 4: break
        log.append(dict(half=h, seed=k, n_subjects=len(ids), best_val_ce=best, epochs=ep + 1, secs=time.time() - t0))
        print(f"{model} rep{rep} half{h} seed{k}: best val CE {best:.4f} ({ep+1} epochs, {time.time()-t0:.0f}s)", flush=True)
if log: json.dump(log, open(out / "train_log.json", "w"), indent=1)

# ---------------------------------------------------------------- derivative fields on the EXP13 grid
c, cond, _ = X.load(model, rep, "prob")
grid = torch.tensor(data["V"][c["ids"]], dtype=torch.float64, device=dev)
for ch in ("prob", "logit"):
    fp = out / f"fields_{ch}.pt"
    if fp.exists(): continue
    F = {nm: [] for nm in ("g", "Htt", "Htx")}
    for h in range(halves):
        for k in range(seeds):
            net = make(); net.decoder.load_state_dict(torch.load(out / f"half{h}_seed{k}.pt", map_location="cpu")); net = net.to(dev).eval().requires_grad_(False)
            b = base_derivatives(decoder_h(net, data["mu_z"], data["Minv"], ch), grid, c["X"].double().to(dev), which=("g", "Htt", "Htx"), chunk=2048)
            for nm in F: F[nm].append(b[nm].cpu())
    torch.save({nm: torch.stack(v).reshape(halves, seeds, *v[0].shape) for nm, v in F.items()}, fp)
    print("saved", fp, flush=True)
