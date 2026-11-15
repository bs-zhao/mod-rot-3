"""Phase 4 synthetic check: f(u) = u1^2 - u2^2 (curvature on the axes) vs the same function in 45-degree-rotated coordinates (f = 2 u1 u2).
EXP13's UUd_share rewards the 45-degree version (diagonal curvature share drops from 1 to 0): a known counter-example, quantified here with the catalogue terms."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import cldlib as L
H0 = torch.tensor([[2.0, 0.0], [0.0, -2.0]]); R = torch.tensor([[np.cos(np.pi/4), -np.sin(np.pi/4)], [np.sin(np.pi/4), np.cos(np.pi/4)]], dtype=torch.float32)
H1 = R @ H0 @ R.T
for name, H in (("axes (u1^2-u2^2)", H0), ("rotated 45deg (2 u1 u2)", H1)):
    e = H ** 2
    print(f"{name:26s} UUd_share {float(L.C.UU['diagshare'](e)):.3f}  offshare {float(L.C.UU['offshare'](e)):.3f}  off_neff {float(L.C.UU['off_neff'](e)) if e[0,1]>0 else float('nan'):.3f}")
print("=> the diagonal-share term prefers the rotated (mixed) coordinates here; the off-diagonal terms prefer the axes. In CR3 / C1-C3 adding an off-diagonal term as a third part made every model worse (phase1_ctx results, kind=ux+uu+off).")
