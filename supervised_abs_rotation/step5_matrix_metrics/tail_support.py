"""Scale-free effective support, emphasizing small residual tails.

Single-largest-entry dominance and Shannon entropy need not agree with
effective support; they are deliberately absent from this primary score set.
"""
from __future__ import annotations

import numpy as np


def scores(energy):
    e = np.maximum(np.asarray(energy, dtype=float).ravel(), 0.)
    if not np.isfinite(e).all():
        raise ValueError("Non-finite energy")
    if e.sum() <= 1e-300:
        return {key: 0. for key in (
            "effective_support_p0p5", "effective_support_p1",
            "soft_count_eps001", "soft_count_eps003", "soft_count_eps010",
            "fractional_count_eps001", "fractional_count_eps003", "fractional_count_eps010",
            "tail2_l2")}
    p = np.sort(e/e.sum())[::-1]
    out = {}
    for label, power in (("p0p5", .5), ("p1", 1.)):
        out["effective_support_"+label] = float(np.sum(p**(power/2))**(1/(1-power/2)))
    cumulative = np.cumsum(p)
    for label, eps in (("001", .01), ("003", .03), ("010", .10)):
        out["soft_count_eps"+label] = float(np.sum(p/(p+eps**2)))
        target = 1-eps**2
        index = min(int(np.searchsorted(cumulative, target)), len(p)-1)
        before = cumulative[index-1] if index else 0.
        out["fractional_count_eps"+label] = float(index+(target-before)/max(p[index], 1e-300))
    out["tail2_l2"] = float(np.sqrt(p[2:].sum()))
    return out


def moment_covariance(field, pooling):
    """field[subject, context, ...] -> covariance of vectorized effects.

    Equal-context and equal-point modes remove overall effect magnitude
    before pooling. The normalization is invariant to U rotations.
    """
    field = np.asarray(field, dtype=float)
    flat = field.reshape(*field.shape[:2], -1)
    energy = np.sum(flat*flat, axis=-1)
    if pooling == "global":
        normalized = flat
    elif pooling == "context":
        scale = np.sqrt(np.mean(energy, axis=0))
        normalized = flat/np.maximum(scale[None, :, None], 1e-100)
    elif pooling == "point":
        normalized = flat/np.maximum(np.sqrt(energy)[..., None], 1e-100)
    else:
        raise ValueError(pooling)
    matrix = normalized.reshape(-1, normalized.shape[-1])
    return matrix.T @ matrix/len(matrix)


def uu_energy(covariance, q):
    d = len(q)
    return np.maximum(np.einsum("ai,bj,ci,dj,abcd->ij", q, q, q, q,
        covariance.reshape(d, d, d, d), optimize=True), 0.)
