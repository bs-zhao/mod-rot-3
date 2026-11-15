"""EXP23 multi-scale aggregation of the frozen EXP13 loss over smoothing scales sigma_k (fixed Gaussian kernels in physical V space).

A  signed field averaging : E_A = E[(sum_k w_k H_k)^2]        -> Hessian of the mixture-smoothed function; needs the mixed field, computed once
B  energy averaging       : E_B = sum_k w_k E[H_k^2]          -> weighted average of the cached per-scale moments
C  loss averaging         : L_C = sum_k w_k [L_UX,k/s_UX,k + L_UU,k/s_UU,k]   (each scale has its own Haar scales)
Single-scale degeneracy: with one scale A = B = C = the single-scale loss (unit-tested).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp20_latent_integration"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp22_smoothed_hessian"))
import x7core as X  # noqa: E402
import smooth_fields as SF  # noqa: E402
from common.derivs import base_derivatives  # noqa: E402


class LossAveraged:
    """C: weighted sum of per-scale normalised losses (shared search machinery of MomentLoss)."""

    def __init__(self, losses, weights):
        self.L, self.w = losses, np.asarray(weights, float) / np.sum(weights)
        self.d = losses[0].d

    def total(self, M):
        return sum(float(w) * l.total(M) for w, l in zip(self.w, self.L))

    search = X.MomentLoss.search


def energy_averaged(moms, weights):
    """B: weighted average of per-scale (SX, T4)."""
    w = np.asarray(weights, float) / np.sum(weights)
    return sum(wi * m[0] for wi, m in zip(w, moms)), sum(wi * m[1] for wi, m in zip(w, moms))


def mixture_subject_moments(st, ch, ids, sigmas, weights, kernel, C_ref, n_in, bs=16, chunk=2048, seed=7):
    """A: per-subject moments of the MIXTURE-smoothed Hessian H_mix = sum_k w_k mean_j H(v + sigma_k delta_j)."""
    w = np.asarray(weights, float) / np.sum(weights)
    d, p = st.d, st.p
    h, X_ = st.h(ch), st.X.to(X.DEV)
    offs = [(wk, sk * SF.inner_offsets(n_in, d, kernel, C_ref, seed)) for wk, sk in zip(w, sigmas)]
    SXs, T4s = [], []
    for i in range(0, len(ids), bs):
        V0 = torch.tensor(st.V[ids[i:i + bs]], dtype=torch.float64, device=X.DEV)
        B, K = len(V0), len(X_)
        Htt = torch.zeros(B, K, d, d, dtype=torch.float64, device=X.DEV)
        Htx = torch.zeros(B, K, d, p, dtype=torch.float64, device=X.DEV)
        for wk, delta in offs:
            for j in range(len(delta)):
                b = base_derivatives(h, V0 + torch.tensor(delta[j], dtype=torch.float64, device=X.DEV), X_, which=("Htt", "Htx"), chunk=chunk)
                Htt += wk * b["Htt"].double() / len(delta)
                Htx += wk * b["Htx"].double() / len(delta)
        a, c = X.point_moments(Htt.cpu(), Htx.cpu())
        SXs.append(a.numpy())
        T4s.append(c.numpy())
    return np.concatenate(SXs), np.concatenate(T4s)
