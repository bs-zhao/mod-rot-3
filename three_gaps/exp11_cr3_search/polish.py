"""EXP11: stage A hard axis -> loss search on the complement -> free polish of ALL rows with the same loss (small steps).
python polish.py MODEL CHANNEL REPEAT LOSS[,LOSS]  [STARTS]"""
import sys, time, numpy as np, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import harness as H, floor as FL
from common.rotlib import effective_A
torch.set_num_threads(4)
model, ch, rep, losses = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].split(",")
starts = int(sys.argv[5]) if len(sys.argv) > 5 else 12
cond = FL.CONDS[model] + ("_rep02" if rep == 2 else "")
c = H.load(model, cond, "decoder", ch); F, _ = H.stageA(c, ch)
R = np.load(H.io.ROOT / "outputs" / "three_gaps" / "exp00" / f"{model}_{cond}_pca0" / "R_oracle_raw.npy")
t_dev = torch.tensor(c["V"][c["ids"]], dtype=torch.float32)
Mo = torch.tensor(R, dtype=torch.float32)
print(f"== {model} {ch} rep{rep}: oracle %.3f/%.3f/%.3f" % H.evaluate(c, R), flush=True)
for loss in losses:
    f, lam_min, sig2, E = FL.make_loss(c, loss)
    fin = H.optimize(c, f, "SO", F=F, starts=starts, steps=250, seed=7)
    M0 = torch.tensor(fin[0][1], dtype=torch.float32)
    W = torch.zeros(len(M0), len(M0), requires_grad=True); opt = torch.optim.Adam([W], lr=0.01)
    for _ in range(300):
        M = torch.matrix_exp(W - W.T) @ M0
        L = f(None, effective_A(M, t_dev), M, c); opt.zero_grad(); L.backward(); opt.step()
    with torch.no_grad():
        M = (torch.matrix_exp(W - W.T) @ M0)
        Lp = float(f(None, effective_A(M, t_dev), M, c)); Lo = float(f(None, effective_A(Mo, t_dev), Mo, c)); Lh = float(fin[0][0])
    print(f"{loss:10s} L: hardA {Lh:.4f}  polished {Lp:.4f}  oracle {Lo:.4f} | TEST hardA %.3f/%.3f/%.3f -> polished %.3f/%.3f/%.3f" %
          (H.evaluate(c, M0.numpy()) + H.evaluate(c, M.numpy())), flush=True)
