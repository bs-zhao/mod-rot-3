"""Step 0b - sanity check of the frozen baseline: what does the EXP13 loss return when the second-order decoder fields are replaced by
PURE NOISE (no function structure at all)?  If that were close to 0.82, the baseline would be reading the latent covariance, not the
function.  Also reports the PCA axes of the latent.  Run from the repository root:  python cld-crossfit-v3/step0_estimand/noise_only.py"""
import sys, numpy as np, torch
sys.path.insert(0, 'x6_plan06/common'); import x6lib as X
from common import io
torch.set_num_threads(4)
for rep in (1, 2):
    c, cond, R = X.load("CR3", rep, "prob")
    V, te, tr = c["V"], c["te"], c["tr"]
    print(f"rep{rep} oracle %.3f/%.3f/%.3f  identity %.3f/%.3f/%.3f" % (X.evaluate(c, R) + X.evaluate(c, np.eye(4))))
    ev, evec = np.linalg.eigh(np.cov(V[tr].T)); print("  PCA axes  %.3f/%.3f/%.3f" % X.evaluate(c, evec.T), " eig", ev.round(2))
    for seed in range(3):
        rng = np.random.default_rng(seed); c2 = dict(c)
        c2["Htt"] = torch.tensor(rng.standard_normal(c["Htt"].shape), dtype=torch.float32)
        c2["Htt"] = (c2["Htt"] + c2["Htt"].transpose(2, 3)) / 2
        c2["Htx"] = torch.tensor(rng.standard_normal(c["Htx"].shape), dtype=torch.float32)
        for names in (("UX_neff", "UUd_share"), ("UUd_share",), ("UX_neff",)):
            b = X.Baseline(c2, names=names); m = X.evaluate(c2, b.search()[0][1])
            print(f"  pure noise fields seed{seed} {'+'.join(names):20s} mcc/leak/max %.3f/%.3f/%.3f" % m, flush=True)
