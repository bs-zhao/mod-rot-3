"""Phase 3: derivatives from a SMOOTH SURROGATE fitted only to the frozen decoder's output values (probabilities) on the historical (subject x bank-stimulus) grid.

Surrogate: total-degree-k polynomial in the joint variables (V / s_V, standardised X) with ridge (the polynomial space of degree <= k is invariant under linear changes of V,
so the surrogate does not prefer any axes).  Hyper-parameters (degree, ridge) are chosen by held-out PREDICTION error on other TRAIN subjects, never by T.
Derivatives of the surrogate (autograd) feed the frozen EXP13 moment loss unchanged.
python run.py MODEL REP [STARTS=8] [STEPS=150]
"""
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.func import hessian, vmap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import cldlib as L

torch.set_num_threads(4)
model, rep = sys.argv[1], int(sys.argv[2])
STARTS, STEPS = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 3 else (8, 150)
RES = HERE / "results"
RES.mkdir(exist_ok=True)
dev = L.Dev(model, rep)
st, c = dev.st, dev.c
d, p = st.d, st.p
DEVICE = L.X.DEV
Xb = st.X.to(torch.float64)
sx_mu, sx_sd = Xb.mean(0), Xb.std(0) + 1e-12
Vsc = float(np.sqrt(np.trace(L.X.population_cov(st.V[c["ids"]])) / d))


def feats(Vq, Xq, E):
    z = torch.cat([Vq / Vsc, (Xq - sx_mu.to(Xq.device)) / sx_sd.to(Xq.device)], -1)
    return torch.prod(z.unsqueeze(-2) ** E, -1)


def exps(k):
    n = d + p
    out = [e for e in itertools.product(range(k + 1), repeat=n) if sum(e) <= k] if n * k < 40 else None
    if out is None:                                   # combinatorial generation for larger n
        from sklearn.preprocessing import PolynomialFeatures
        pf = PolynomialFeatures(k, include_bias=True).fit(np.zeros((1, n)))
        out = [tuple(r) for r in pf.powers_]
    return torch.tensor(out, dtype=torch.float64, device=DEVICE)


ids = c["ids"]
Kb = len(Xb)
Vgrid = torch.tensor(st.V[ids], dtype=torch.float64).repeat_interleave(Kb, 0)
Xgrid = Xb.repeat(len(ids), 1)
y = c["h"].double().reshape(-1)                                        # decoder probabilities at the grid (frozen cache)
# held-out subjects: other TRAIN subjects, decoder queried directly
rng = np.random.default_rng(5)
others = np.setdiff1d(st.tr, ids)
hid = np.sort(rng.choice(others, 150, replace=False))
with torch.no_grad():
    hf = st.h("prob")
    yv = torch.stack([torch.vmap(lambda xx, v=torch.tensor(st.V[s], dtype=torch.float64, device=DEVICE): hf(v, xx))(Xb.to(DEVICE)) for s in hid]).reshape(-1).cpu()
Vv = torch.tensor(st.V[hid], dtype=torch.float64).repeat_interleave(Kb, 0)
Xv = Xb.repeat(len(hid), 1)
rows = []
for k, lam in [(3, 1e-6), (4, 1e-6)]:
    t0 = time.time()
    E = exps(k)
    F = torch.cat([feats(Vgrid[i:i + 20000].to(DEVICE), Xgrid[i:i + 20000].to(DEVICE), E) for i in range(0, len(Vgrid), 20000)])
    G = F.T @ F
    w = torch.linalg.solve(G + lam * len(F) * torch.eye(G.shape[0], dtype=G.dtype, device=DEVICE), F.T @ y.to(DEVICE))
    pred = (F @ w).cpu()
    Fv = feats(Vv.to(DEVICE), Xv.to(DEVICE), E)
    pv = (Fv @ w).cpu()
    r2 = 1 - float(((pv - yv) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()); r2tr = 1 - float(((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    # surrogate second derivatives on the grid -> frozen moments
    fsur = lambda z: (torch.prod(z.unsqueeze(0) ** E, -1) @ w)
    Hfn = vmap(hessian(fsur))
    SX = torch.zeros(p, d, d, dtype=torch.float64)
    T4 = torch.zeros(d * d, d * d, dtype=torch.float64)
    n = 0
    for i in range(0, len(Vgrid), 5000):
        z = torch.cat([Vgrid[i:i + 5000] / Vsc, (Xgrid[i:i + 5000] - sx_mu) / sx_sd], -1).to(DEVICE)
        Hz = Hfn(z).cpu()                                   # (n, d+p, d+p) in scaled variables
        Hvv = Hz[:, :d, :d] / Vsc ** 2
        Hvx = Hz[:, :d, d:] / (Vsc * sx_sd)
        SX += torch.einsum("naj,nbj->jab", Hvx, Hvx)
        Hf = Hvv.reshape(len(z), -1)
        T4 += Hf.T @ Hf
        n += len(z)
    SX, T4 = (SX / n).float(), (T4 / n).reshape(d, d, d, d).float()
    ml = L.X.MomentLoss(SX.numpy(), T4.numpy(), L.X.population_cov(st.V[ids]), d, scales={"UX_neff": 1.0, "UUd_share": 1.0})
    dev2 = L.Dev.__new__(L.Dev)
    dev2.__dict__.update(dev.__dict__)
    dev2.ml = ml
    loss, Q = L.search_terms(dev2, L.FROZEN, None, 0.0, STARTS, STEPS)
    mcc, leak, lmax = L.X.evaluate(st, Q)
    rows.append(dict(model=model, rep=rep, cand=f"sieve|deg{k}|ridge{lam}", status="ok", mcc=mcc, leak=leak, leak_max=lmax, oracle_angle_max=L.X.signed_perm_angle(Q, st.oracle),
                     heldout_R2=r2, train_R2=r2tr, n_terms=len(E), secs=time.time() - t0))
    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RES / f"dev_rep{rep}_{model}.csv", index=False)
