"""EXP22 smoothed Hessian: fixed Gaussian kernel in the ORIGINAL physical V coordinates (never re-defined per rotation).

f_sigma(V,X) = E_delta f(V + delta, X),  delta = sigma * (eps @ L.T),  L L^T = C_ref (covariance kernel) or tr(C_ref)/d * I (spherical kernel).
Inner nodes: antithetic scrambled Sobol (n_in/2 base nodes + their negatives), the SAME offsets for every centre (shared, reproducible).
For each centre we accumulate, over the inner nodes,
  * the signed mean Hessian  H_bar = mean_j H(v + delta_j, x)          -> "smooth-then-square" moments  E_c[ H_bar o H_bar ]
  * the mean of the squared Hessians                                    -> "square-then-average" moments E_c[ mean_j H_j o H_j ] (enlarged measure, noise energy kept)
Because the kernel is fixed in physical coordinates, H_UU = A H_VV,sigma A^T and H_UX = A H_VX,sigma, so the EXP13 moment contraction applies unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp20_latent_integration"))
import x7core as X  # noqa: E402
import nodes as N  # noqa: E402
from common.derivs import base_derivatives  # noqa: E402


def inner_offsets(n_in, d, kernel, C_ref, seed=7):
    """unit-sigma offsets (n_in, d): antithetic RQMC; kernel 'cov' (C_ref) or 'sph' (tr(C)/d I)."""
    if n_in % 2 or (n_in // 2) & (n_in // 2 - 1):
        raise ValueError("n_in/2 must be a power of two (e.g. 8, 16, 32)")
    eps, _ = N.rqmc_eps(n_in // 2, d, seed)
    eps = np.concatenate([eps, -eps])
    if kernel == "cov":
        return eps @ np.linalg.cholesky(C_ref).T
    if kernel == "sph":
        return eps * np.sqrt(np.trace(C_ref) / d)
    raise ValueError(kernel)


def smoothed_subject_moments(st, ch, ids, sigma, kernel, C_ref, n_in, bs=16, chunk=2048, seed=7):
    """per-subject moments (SX, T4) for [smooth-then-square] and [square-then-average]; plus p-change diagnostics. float64 accumulation."""
    d, p = st.d, st.p
    delta = sigma * inner_offsets(n_in, d, kernel, C_ref, seed) if sigma > 0 else np.zeros((1, d))
    nj = len(delta)
    h = st.h(ch)
    X_ = st.X.to(X.DEV)
    out = dict(SX_s=[], T4_s=[], SX_e=[], T4_e=[], dp=[], p0=[])
    for i in range(0, len(ids), bs):
        V0 = torch.tensor(st.V[ids[i:i + bs]], dtype=torch.float64, device=X.DEV)
        B, K = len(V0), len(X_)
        Hs_tt = torch.zeros(B, K, d, d, dtype=torch.float64, device=X.DEV)
        Hs_tx = torch.zeros(B, K, d, p, dtype=torch.float64, device=X.DEV)
        SXe = torch.zeros(B, p, d, d, dtype=torch.float64)
        T4e = torch.zeros(B, d, d, d, d, dtype=torch.float64)
        hsum = torch.zeros(B, K, dtype=torch.float64, device=X.DEV)
        for j in range(nj):
            grid = V0 + torch.tensor(delta[j], dtype=torch.float64, device=X.DEV)
            b = base_derivatives(h, grid, X_, which=("Htt", "Htx"), chunk=chunk)
            tt, tx = b["Htt"].double(), b["Htx"].double()
            Hs_tt += tt
            Hs_tx += tx
            hsum += b["h"].double()
            a, c = X.point_moments(tt.cpu(), tx.cpu())
            SXe += a
            T4e += c
        Hs_tt /= nj
        Hs_tx /= nj
        hsum /= nj
        SXs, T4s = X.point_moments(Hs_tt.cpu(), Hs_tx.cpu())
        out["SX_s"].append(SXs.numpy())
        out["T4_s"].append(T4s.numpy())
        out["SX_e"].append((SXe / nj).numpy())
        out["T4_e"].append((T4e / nj).numpy())
        out["p0"].append(hsum.cpu().numpy())                       # mean smoothed probability per (subject, X)
    return {k: np.concatenate(v) for k, v in out.items()}
