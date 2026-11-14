"""EXP13: terms weighted by their achievable-simplicity z-score (T-free).
For each generic term k: Haar null (mean m_k, sd s_k over random rotations) and the term's own optimum L_k*.
z_k = (m_k - L_k*) / s_k  = how many null-sds of simplicity the data admit under term k.  Combined loss = sum_k z_k * (L_k - m_k)/s_k.
python weighted.py MODEL CHANNEL REPEAT [STARTS]"""
import sys, time, numpy as np, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "exp11_cr3_search"))
import harness as H, losses2 as L2, screen as SC
from common.rotlib import effective_A, haar
torch.set_num_threads(4)
model, ch, rep = sys.argv[1], sys.argv[2], int(sys.argv[3]); starts = int(sys.argv[4]) if len(sys.argv) > 4 else 24
c, cond = SC.load(model, rep, ch); d = c["g"].shape[-1]
SX, T4 = L2.moments(c, standardize_x=True)
names = ["UX_neff", "UUd_share"]
term = {k: (lambda A, k=k: L2.LOSSES[k](*L2.energies(A, SX, T4))) for k in names}
t_dev = torch.tensor(c["V"][c["ids"]], dtype=torch.float32)
rng = np.random.default_rng(1); Ms = [torch.tensor(haar(d, rng), dtype=torch.float32) for _ in range(300)]
m, s, z = {}, {}, {}
with torch.no_grad():
    for k in names:
        v = np.array([float(term[k](effective_A(M, t_dev))) for M in Ms]); m[k], s[k] = v.mean(), v.std() + 1e-9
for k in names:
    fin = H.optimize(c, lambda b, A, M, cc, k=k: term[k](A), "SO", F=None, starts=starts, steps=250, seed=7)
    z[k] = max((m[k] - fin[0][0]) / s[k], 0.0)
print(f"== {model} {ch} rep{rep}: z-scores " + " ".join(f"{k}={z[k]:.2f}" for k in names), flush=True)
f = lambda b, A, M, cc: sum(z[k] * (term[k](A) - m[k]) / s[k] for k in names)
R = np.load(SC.io.ROOT / "outputs" / "three_gaps" / "exp00" / f"{model}_{cond}_pca0" / "R_oracle_raw.npy")
fin = H.optimize(c, f, "SO", F=None, starts=starts, steps=250, seed=7)
print(f"{model} {ch} rep{rep} z-weighted: TEST mcc/leak/max %.3f %.3f %.3f | oracle %.3f/%.3f/%.3f" % (H.evaluate(c, fin[0][1]) + H.evaluate(c, R)), flush=True)
