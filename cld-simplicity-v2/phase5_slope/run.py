"""Phase 5: U-X interaction from FIRST derivatives.  One autograd pass gives g(v,x) = dh/dv at every (subject, stimulus); the X-dependence of g is then estimated by
regression over the real stimulus bank (no x-autograd, no off-support x queries):  g_i(x) ~ a_i + sum_j beta_ij x_j [+ quadratic terms];  H_UX_hat = d g / d x (analytic from the fit).
The UU part keeps the frozen autograd Hessian moments.  python run.py MODEL REP [STARTS STEPS]"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import cldlib as L
torch.set_num_threads(2)
model, rep = sys.argv[1], int(sys.argv[2])
STARTS, STEPS = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (24, 250)
RES = HERE / "results"; RES.mkdir(exist_ok=True)
dev = L.Dev(model, rep); st, c = dev.st, dev.c
g = c["g"].double()                                  # (S,K,d)
Xb = st.X.double(); S, K, d = g.shape; p = Xb.shape[1]
Xc = Xb - Xb.mean(0)                                 # raw X units (frozen UX units), centred
def design(deg):
    cols = [torch.ones(K, 1, dtype=torch.float64), Xc]
    quad = []
    if deg == 2:
        for j in range(p):
            for l in range(j, p):
                quad.append((Xc[:, j] * Xc[:, l]).unsqueeze(1))
        cols += quad
    return torch.cat(cols, 1)
def htx_hat(deg, lam):
    Dm = design(deg); nb = Dm.shape[1]
    # ridge on non-intercept terms, scaled by the design energy (unit-free)
    P = torch.eye(nb, dtype=torch.float64) * lam * (Dm ** 2).mean(0).mean(); P[0, 0] = 0
    B = torch.linalg.solve(Dm.T @ Dm + K * P, Dm.T @ g.reshape(-1, K, d).permute(1, 0, 2).reshape(K, S * d)).reshape(nb, S, d)   # (nb,S,d)
    beta = B[1:1 + p]                                                            # (p,S,d)
    Hx = beta.permute(1, 2, 0).unsqueeze(1).expand(S, K, d, p).clone()           # linear part: constant slope
    if deg == 2:
        idx = 1 + p
        for j in range(p):
            for l in range(j, p):
                gam = B[idx]                                                    # (S,d): coefficient of x_j x_l
                if j == l:
                    Hx[..., j] += 2 * gam.unsqueeze(1) * Xc[:, j].view(1, K, 1)
                else:
                    Hx[..., j] += gam.unsqueeze(1) * Xc[:, l].view(1, K, 1)
                    Hx[..., l] += gam.unsqueeze(1) * Xc[:, j].view(1, K, 1)
                idx += 1
    return Hx
true = c["Htx"].double()
rows = []
for name, deg, lam in [("frozen_autograd", None, 0), ("lin_slope", 1, 1e-6), ("quad_slope", 2, 1e-4)]:
    t0 = time.time()
    if deg is None:
        ml = dev.ml
        fit = float("nan")
    else:
        Hx = htx_hat(deg, lam)
        SX = torch.einsum("skaj,skbj->jab", Hx, Hx) / (S * K)
        fit = float(((Hx - true) ** 2).sum() / (true ** 2).sum())          # diagnostic only: relative error of the reconstructed H_UX vs autograd
        ml = L.X.MomentLoss(SX.float().numpy(), dev.ml.T4.numpy(), L.X.population_cov(st.V[c["ids"]]), d, scales={"UX_neff": 1.0, "UUd_share": 1.0})
    dev2 = L.Dev.__new__(L.Dev); dev2.__dict__.update(dev.__dict__); dev2.ml = ml
    loss, Q = L.search_terms(dev2, L.FROZEN, None, 0.0, STARTS, STEPS)
    mcc, leak, lmax = L.X.evaluate(st, Q)
    rows.append(dict(model=model, rep=rep, cand=name, status="ok", mcc=mcc, leak=leak, leak_max=lmax, oracle_angle_max=L.X.signed_perm_angle(Q, st.oracle), rel_err_Htx=fit, secs=time.time() - t0))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
pd.DataFrame(rows).to_csv(RES / f"dev_rep{rep}_{model}.csv", index=False)
