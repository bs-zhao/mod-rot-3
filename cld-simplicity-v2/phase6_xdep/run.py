"""Phase 6 runner (CR3): X-dependence of g_U estimated by flexible non-autograd estimators; fidelity vs the decoder's autograd H_UX, then the frozen EXP13 loss with SX from the estimate
(UU part = frozen autograd T4).  python run.py MODEL REP [STARTS STEPS]
Hyper-parameters inside each method are chosen by cross-validated prediction of g.  T / Oracle only post hoc."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))
import cldlib as L
import methods as M

torch.set_num_threads(4)
model, rep = sys.argv[1], int(sys.argv[2])
STARTS, STEPS = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (24, 250)
RES = HERE / "results"
RES.mkdir(exist_ok=True)
dev = L.Dev(model, rep)
st, c = dev.st, dev.c
G = c["g"].double()
Xb = st.X.double()
true = c["Htx"].double()
S, K, d = G.shape
p = Xb.shape[1]
Vsc = float(np.sqrt(np.trace(L.X.population_cov(st.V[c["ids"]])) / d))
Vgrid = torch.tensor(st.V[c["ids"]], dtype=torch.float64)
METHODS = {
    "krr_iso": lambda: M.m_krr(G, Xb, False), "krr_ard": lambda: M.m_krr(G, Xb, True),
    "sharedbasis_r4": lambda: M.m_shared_basis(G, Xb, 4), "sharedbasis_r8": lambda: M.m_shared_basis(G, Xb, 8), "sharedbasis_r16": lambda: M.m_shared_basis(G, Xb, 16),
    "cp_r4": lambda: M.m_cp(G, Xb, 4), "cp_r8": lambda: M.m_cp(G, Xb, 8),
    "localLinear": lambda: M.m_local_linear(G, Xb, None, True), "chord_knn": lambda: M.m_local_linear(G, Xb, None, False),
    "gam_spline": lambda: M.m_gam(G, Xb),
    "sobolev_mlp": lambda: M.m_sobolev_mlp(G, c["h"].double(), Vgrid, Xb, Vsc, device=__import__("os").environ.get("MLP_DEVICE", str(L.X.DEV)), epochs=25),
}
import os
if os.environ.get("ONLY"):
    METHODS = {k: v for k, v in METHODS.items() if k in os.environ["ONLY"].split(",")}
rows = []
SXtrue = torch.einsum("skaj,skbj->jab", true, true) / (S * K)


def run(name, Hx, info, secs):
    SX = torch.einsum("skaj,skbj->jab", Hx, Hx) / (S * K)
    rel_h = float(((Hx - true) ** 2).sum() / (true ** 2).sum())
    rel_sx = float(torch.linalg.norm(SX - SXtrue) / torch.linalg.norm(SXtrue))
    ml = L.X.MomentLoss(SX.float().numpy(), dev.ml.T4.numpy(), L.X.population_cov(st.V[c["ids"]]), d, scales={"UX_neff": 1.0, "UUd_share": 1.0})
    dev2 = L.Dev.__new__(L.Dev)
    dev2.__dict__.update(dev.__dict__)
    dev2.ml = ml
    loss, Q = L.search_terms(dev2, L.FROZEN, None, 0.0, STARTS, STEPS)
    mcc, leak, lmax = L.X.evaluate(st, Q)
    rows.append(dict(model=model, rep=rep, cand=name, status="ok", mcc=mcc, leak=leak, leak_max=lmax, oracle_angle_max=L.X.signed_perm_angle(Q, st.oracle),
                     rel_err_Htx=rel_h, rel_err_SX=rel_sx, info=str(info)[:200], secs=secs))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items() if k != "info"}, flush=True)
    pd.DataFrame(rows).to_csv(RES / f"{os.environ.get('TAGOUT','dev')}_rep{rep}_{model}.csv", index=False)


if not os.environ.get("ONLY"):
    run("autograd_frozen", true, {}, 0.0)
for name, fn in METHODS.items():
    t0 = time.time()
    try:
        Hx, info = fn()
        run(name, Hx, info, time.time() - t0)
    except Exception as e:                                     # failures are recorded, not hidden
        rows.append(dict(model=model, rep=rep, cand=name, status=f"error: {type(e).__name__}: {e}"[:200]))
        print("ERROR", name, e, flush=True)
        pd.DataFrame(rows).to_csv(RES / f"{os.environ.get('TAGOUT', 'dev')}_rep{rep}_{model}.csv", index=False)
