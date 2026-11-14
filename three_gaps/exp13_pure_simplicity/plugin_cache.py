"""EXP13 diagnostic only (analytic Y never counts as success): second-order derivative field of the analytic CR3 logit evaluated at the linear
estimate of T from V (T_std ~ A V + b), in the same V coordinates / grid as the decoder cache. Saved as cache2_CR3_<cond>_plugin.pt"""
import sys, torch, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from common import io
from common.analytic import ETA
from common.derivs import base_derivatives
import cache2 as C2
rep = int(sys.argv[1]) if len(sys.argv) > 1 else 1
cond = "beta5em05_lambda0_ctx0p25to0p8_L4" + ("_rep02" if rep == 2 else "")
c = torch.load(C2.OUT / f"cache2_CR3_{cond}_logit.pt", weights_only=False)
data, truth = io.load_all("CR3", 0.00005, 4, 0, rep, need_checkpoint=False)
V, T, tr = data["V"], truth.std(), data["train_ids"]
B = np.linalg.lstsq(np.c_[V[tr], np.ones(len(tr))], T[tr], rcond=None)[0]; A, b = B[:-1].T, B[-1]
dev = io.pick_device("auto")
At, bt, mu, sd = [torch.tensor(z, dtype=torch.float64, device=dev) for z in (A, b, truth.mu, truth.sd)]
h = lambda v, x: ETA["CR3"](mu + sd * (At @ v + bt), x)
grid = torch.tensor(V[c["ids"]], dtype=torch.float64, device=dev)
out = base_derivatives(h, grid, c["X"].to(dev), which=("g", "Htt", "Htx"), chunk=2048)
c2 = dict(c); c2["g"], c2["Htt"], c2["Htx"], c2["h"] = out["g"].cpu(), out["Htt"].cpu(), out["Htx"].cpu(), out["h"].cpu()
torch.save(c2, C2.OUT / f"cache2_CR3_{cond}_plugin.pt"); print("saved plugin cache")
