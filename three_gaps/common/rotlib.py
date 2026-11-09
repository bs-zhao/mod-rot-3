"""Views of derivative blocks, coordinate transforms, rotations and axis metrics.

A *view* = (base block, number r of leading U-indices, post-op, aggregation).
Under u = M t (M any invertible d x d) followed by per-axis z-scoring
u~ = S^-1 (u - mean) with S = diag(sd over subjects of M t), every U-index of a
derivative tensor transforms with  A = S @ inv(M).T .  For orthogonal M = Q
this is S @ Q.  Hence every loss below is exactly invariant to any diagonal
rescaling D applied after Q (G0 audit in tests).
"""
from __future__ import annotations

import math

import numpy as np
import torch

# name: (base, r, post, mode)
VIEWS = {
    "G/pool": ("g", 1, None, "pool"),
    "G/trial": ("g", 1, None, "trial"),
    "Gc/trial": ("gc", 1, None, "trial"),
    "HUX/pool": ("Htx", 1, None, "pool"),
    "HUX/trial": ("Htx", 1, None, "trial"),
    "HUU/pool": ("Htt", 2, None, "pool"),
    "HUUd/pool": ("Htt", 2, "diag2", "pool"),
    "HUUo/pool": ("Htt", 2, "off2", "pool"),
    "HUU/trial": ("Htt", 2, None, "trial"),
    "HUUd/trial": ("Htt", 2, "diag2", "trial"),
    "TUUX/pool": ("Tttx", 2, None, "pool"),
    "TUUXd/pool": ("Tttx", 2, "diag2", "pool"),
    "TUXX/pool": ("Ttxx", 1, None, "pool"),
    "TUUUd/pool": ("Tttt", 3, "diag3", "pool"),
}


def add_centered(base):
    if "g" in base:
        base["gc"] = base["g"] - base["g"].mean(1, keepdim=True)
    return base


def effective_A(M, t_subj):
    """A = S inv(M)^T with S = sd_subjects(M t)."""
    u = t_subj @ M.T
    s = u.std(0, unbiased=False)
    return s[:, None] * torch.linalg.inv(M).T


def transform(B, r, A):
    for k in range(r):
        ax = 2 + k
        B = torch.movedim(torch.tensordot(B, A, dims=([ax], [1])), -1, ax)
    return B


def apply_view(base, view, A):
    name, r, post, mode = VIEWS[view]
    B = transform(base[name], r, A)
    if post == "diag2":
        B = torch.movedim(torch.diagonal(B, dim1=2, dim2=3), -1, 2)
    elif post == "off2":
        d = B.shape[2]
        eye = torch.eye(d, device=B.device, dtype=B.dtype)
        B = B * (1 - eye).reshape(1, 1, d, d, *([1] * (B.ndim - 4)))
    elif post == "diag3":
        B = torch.einsum("skiii...->ski...", B)
    S, K, d = B.shape[:3]
    if mode == "pool":
        return B.reshape(S * K, d, -1)
    return torch.movedim(B, 1, 2).reshape(S, d, -1)


# ---------------------------------------------------------------- rotations
def haar(d, rng):
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def givens(d, i, j, theta):
    G = np.eye(d)
    c, s = math.cos(theta), math.sin(theta)
    G[i, i] = c; G[j, j] = c; G[i, j] = -s; G[j, i] = s
    return G


def axis_error(Q):
    """Worst / mean angle (deg) between rows of Q and the matched signed basis axes."""
    from scipy.optimize import linear_sum_assignment
    C = np.abs(np.asarray(Q, float))
    C = C / np.linalg.norm(np.asarray(Q, float), axis=1, keepdims=True)
    r, c = linear_sum_assignment(-C)
    ang = np.degrees(np.arccos(np.clip(C[r, c], 0, 1)))
    return float(ang.max()), float(ang.mean()), c


def corr_matrix(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    s = np.sqrt((a * a).sum(0)[:, None] * (b * b).sum(0)[None, :])
    return (a.T @ b) / np.maximum(s, 1e-300)


def match_report(U, T):
    """Hungarian matched |corr| statistics of U (n,d) vs T (n,d)."""
    from scipy.optimize import linear_sum_assignment
    C = np.abs(corr_matrix(U, T))
    r, c = linear_sum_assignment(-C)
    matched = C[r, c]
    off = C.copy(); off[r, c] = np.nan
    return dict(mcc=float(matched.mean()), worst=float(matched.min()),
                off_mean=float(np.nanmean(off)), off_max=float(np.nanmax(off)),
                perm=c.tolist(), C=C)
