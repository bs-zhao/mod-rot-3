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
    # projected onto an orthonormal stimulus basis Psi (cells = basis functions);
    # for a fixed bank Psi = trial indicators, so these equal the */trial views.
    "GP": ("gP", 1, None, "trial"),
    "HUXP": ("HtxP", 1, None, "trial"),
    "HUUP": ("HttP", 2, None, "trial"),
    "HUUdP": ("HttP", 2, "diag2", "trial"),
    "HUUoP": ("HttP", 2, "off2", "trial"),
    # design-category cells (CR3 bank: gain_only / loss_only / gamma_neutral / other);
    # for continuous-X models the categories are the Legendre basis functions.
    "GCat": ("gP", 1, None, "cat"),
    "HUUCat": ("HttP", 2, None, "cat"),
    "HUUdCat": ("HttP", 2, "diag2", "cat"),
}


def add_centered(base):
    if "g" in base:
        base["gc"] = base["g"] - base["g"].mean(1, keepdim=True)
    return base


def legendre_basis(X, deg):
    """Orthonormal (under U(-1,1)^p) tensor Legendre basis, total degree <= deg. X (K,p)->(K,nb)."""
    import itertools as it
    K, p = X.shape
    P = [torch.ones_like(X), X]
    for n in range(1, deg):
        P.append(((2 * n + 1) * X * P[n] - n * P[n - 1]) / (n + 1))
    P = [P[n] * math.sqrt(2 * n + 1) for n in range(deg + 1)]
    cols, names = [], []
    for total in range(deg + 1):
        for c in it.combinations_with_replacement(range(p), total):
            e = [0] * p
            for k in c:
                e[k] += 1
            col = torch.ones(K, dtype=X.dtype, device=X.device)
            for j, ej in enumerate(e):
                col = col * P[ej][:, j]
            cols.append(col); names.append(tuple(e))
    return torch.stack(cols, 1), names


def bank_categories(X):
    ga, la, pa, gb, lb, pb = X.T
    cat = torch.full((X.shape[0],), 3, dtype=torch.long, device=X.device)
    cat[(pa == 0.5) & (pb == 0.5)] = 2
    cat[(ga == 0) & (gb == 0)] = 1
    cat[(la == 0) & (lb == 0)] = 0
    return cat


def add_projected(base, X, bank, deg=3):
    """Project derivative fields over the K stimuli onto basis Psi (least squares).
    bank=True: Psi = indicators (projection = identity)."""
    if bank:
        for src, dst in (("g", "gP"), ("Htx", "HtxP"), ("Htt", "HttP")):
            if src in base:
                base[dst] = base[src]
        base["cat_index"] = bank_categories(X)
        return base
    Psi, _ = legendre_basis(X.to(base["g"].dtype), deg)
    Pm = torch.linalg.pinv(Psi)  # (nb, K)
    for src, dst in (("g", "gP"), ("Htx", "HtxP"), ("Htt", "HttP")):
        if src in base:
            base[dst] = torch.einsum("bk,sk...->sb...", Pm, base[src])
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
    if mode == "cat" and "cat_index" in base:
        # equal-weight energy per category: pad each category's (S*K_c, d, rest) block to S*K_max
        ci = base["cat_index"]
        cats = []
        kmax = max(int((ci == c).sum()) for c in range(int(ci.max()) + 1))
        for c in range(int(ci.max()) + 1):
            Bc = B[:, ci == c]
            kc = Bc.shape[1]
            Bc = torch.movedim(Bc, 2, 1).reshape(S, d, kc, -1) * math.sqrt(kmax / kc)
            if kc < kmax:
                Bc = torch.cat([Bc, Bc.new_zeros(S, d, kmax - kc, Bc.shape[-1])], 2)
            cats.append(Bc)
        Bc = torch.stack(cats, 3)  # (S, d, kmax, C, rest)
        return torch.movedim(Bc, 2, 1).reshape(S * kmax, d, -1)
    if mode == "cat":
        mode = "trial"
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
