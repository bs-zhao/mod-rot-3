"""EXP13 natural-scale combination: log neff(UX) + mu * self-curvature share of total second-order energy (X standardised).
python natural.py MODEL CHANNEL REPEAT MU[,MU...] [STARTS]"""
import sys, time, numpy as np, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "exp11_cr3_search"))
import harness as H, losses2 as L2, screen as SC
torch.set_num_threads(4)
model, ch, rep = sys.argv[1], sys.argv[2], int(sys.argv[3]); mus = [float(x) for x in sys.argv[4].split(",")]
starts = int(sys.argv[5]) if len(sys.argv) > 5 else 24
c, cond = SC.load(model, rep, ch); d = c["g"].shape[-1]
SX, T4 = L2.moments(c, standardize_x=True)
R = np.load(SC.io.ROOT / "outputs" / "three_gaps" / "exp00" / f"{model}_{cond}_pca0" / "R_oracle_raw.npy")
print(f"== {model} {ch} rep{rep}: oracle %.3f/%.3f/%.3f" % H.evaluate(c, R), flush=True)
for mu in mus:
    fn = L2.make_natural(mu)
    f = lambda b, A, M, cc: fn(*L2.energies(A, SX, T4))
    t = time.time(); fin = H.optimize(c, f, "SO", F=None, starts=starts, steps=250, seed=7)
    print(f"{model} {ch} rep{rep} mu={mu:<5g} TEST mcc/leak/max %.3f %.3f %.3f ({time.time()-t:.0f}s)" % H.evaluate(c, fin[0][1]), flush=True)
