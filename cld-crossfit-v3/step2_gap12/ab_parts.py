"""Step 2c - which part of the prob-channel Hessian carries the Gap-1 identification?   H_p = s1 * H_eta  +  s2 * g_eta g_eta^T
  A = s1 * H_eta        needs the true second derivative of the logit
  B = s2 * g g^T        needs ONLY the logit and its first derivative (s1 = p(1-p), s2 = s1 (1-2p))
If B alone passed, the method would not need second derivatives at all.  Analytic field, mixed true coordinates (Gap 1).
python cld-crossfit-v3/step2_gap12/ab_parts.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "common"))
import xfit as XF
from xfit import X, io
from common.analytic import make_h
from common.derivs import base_derivatives
torch.set_num_threads(4)
dev = io.pick_device("auto"); model = "CR3"
data, truth = io.load_all(model, *XF.BETA[model], 0, 1, need_checkpoint=False)
c, cond, _ = X.load(model, 1, "prob"); ids, Xb, te = c["ids"], c["X"].double(), data["test_ids"]
Tst = truth.std(); Q0 = XF.haar(4, np.random.default_rng(20261115)); W = Tst @ Q0.T; M_true = Q0.T
t_grid = torch.tensor(W[ids], dtype=torch.float64); Qt = torch.tensor(Q0, dtype=torch.float64, device=dev)
h0 = make_h(model, "logit", truth.mu, truth.sd)
b = base_derivatives(lambda w, x: h0(Qt.T @ w, x), t_grid.to(dev), Xb.to(dev), which=("g", "Htt"), chunk=2048, out_dtype=torch.float64)
eta, g, H = b["h"].cpu(), b["g"].cpu(), b["Htt"].cpu()
p = torch.sigmoid(eta); s1 = p * (1 - p); s2 = s1 * (1 - 2 * p)
gp = s1[..., None] * g
PART = {"A": s1[..., None, None] * H, "B": s2[..., None, None] * g[..., :, None] * g[..., None, :]}
PART["A+B"] = PART["A"] + PART["B"]
COMBOS = [("UUd_share",), ("UUI_softL0_03",), ("UUI_softL0_10",), ("UUI_softL0_30",), ("GI_softL0_10",),
          ("UUI_softL0_10", "GI_softL0_10"), ("UUI_softL0_10", "UUd_share")]
rows = []
for nm, Hp in PART.items():
    out = XF.run_losses(XF.moments([dict(g=gp, Htt=Hp)], cross=False), t_grid, COMBOS, W, Tst, te, M_ref=M_true)
    for row, M in out:
        row.update(part=nm, ang_true=XF.axis_error(M @ Q0)[0]); rows.append(row)
    print(f"{nm:4s} " + " | ".join(f"{r['loss']}: {r['ang_true']:.1f}deg" for r, _ in out), flush=True)
pd.DataFrame(rows).to_csv(HERE / "results" / "ab_parts.csv", index=False)
