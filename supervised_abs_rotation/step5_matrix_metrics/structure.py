"""Exploratory lower-is-simpler scores, with fixed signs for every dataset.

All quantities refer to a numerical surrogate, not the unknown generating
function. Sign consistency is NOT a proof of conditional monotonicity.
"""
from __future__ import annotations

import numpy as np


def concentration(energy, prefix):
    e = np.maximum(np.asarray(energy, dtype=float).ravel(), 0.)
    total = float(e.sum())
    if total <= 1e-24:
        return {f"{prefix}_{key}": 0. for key in ("tail1_l2", "tail2_l2", "tail4_l2", "entropy")}
    p = np.sort(e / total)[::-1]
    out = {f"{prefix}_tail{k}_l2": float(np.sqrt(max(0., p[k:].sum()))) for k in (1, 2, 4)}
    out[f"{prefix}_entropy"] = (float(-np.sum(p*np.log(np.maximum(p, 1e-300))) /
                                            np.log(len(p))) if len(p) > 1 else 0.)
    return out


def structural_scores(g, h, d, weights=None):
    """g[n,D], h[n,D,D] use common standardized coordinate units.

    The root-sum-square off-diagonal UU effect is separate from the
    concentration of those effects. Sparse one-edge interactions can have
    exactly zero tail1; that alone cannot distinguish them from no edges.
    """
    g = np.asarray(g, dtype=float); h = np.asarray(h, dtype=float)
    if (g.ndim != 2 or h.shape != (len(g), g.shape[1], g.shape[1]) or
            not 1 < d < g.shape[1] or not np.isfinite(g).all() or not np.isfinite(h).all()):
        raise ValueError("Invalid gradient/Hessian field")
    weights = np.ones(len(g)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (len(g),) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid observation weights")
    weights = weights / weights.sum()
    hu = h[:, :d, :d]; hx = h[:, :d, d:]; gu = g[:, :d]
    eu = np.einsum("n,nij->ij", weights, hu**2)
    ex = np.einsum("n,nij->ij", weights, hx**2)
    diagonal = np.diag(eu)
    ii, jj = np.triu_indices(d, 1)
    # Multiplicity makes this vector's sum equal the full UU Frobenius norm.
    uu_all = np.r_[diagonal, 2*eu[ii, jj]]
    out = concentration(uu_all, "UU_all")
    out.update(concentration(eu[ii, jj], "UU_edges"))
    out.update(concentration(ex, "UX_entries"))
    out.update(concentration(diagonal, "U_curvature"))
    out["UU_interaction_fraction"] = float(2*eu[ii, jj].sum() / max(eu.sum(), 1e-24))
    active = ex.sum(axis=0) > max(float(ex.sum())*1e-12, 1e-24)
    for k in (1, 2):
        columns = [concentration(ex[:, j], "col")[f"col_tail{k}_l2"]
                   for j in np.flatnonzero(active)]
        out[f"UX_column_tail{k}_l2"] = float(np.mean(columns)) if columns else 0.
    signed = np.einsum("n,ni->i", weights, gu)
    absolute = np.einsum("n,ni->i", weights, np.abs(gu))
    # This allows a global sign flip on any axis. Cross-X sign reversals can
    # be perfectly legitimate; do not label this conditional monotonicity.
    out["U_global_sign_inconsistency"] = float(
        np.maximum(absolute - np.abs(signed), 0.).sum() / max(absolute.sum(), 1e-24))
    out["control_UU_fro2"] = float(eu.sum())
    out["control_UX_fro2"] = float(ex.sum())
    out["control_U_gradient_fro2"] = float(np.einsum("n,ni->", weights, gu**2))
    return out


def rotate_fields(g, h, q, d):
    transform = np.eye(g.shape[-1]); transform[:d, :d] = q
    return (g @ transform,
            np.einsum("ij,njk,kl->nil", transform.T, h, transform, optimize=True))


def evaluate_curve(losses, cases, tolerance=1e-7):
    """Strict endpoint preference. Constant and sign-reversed curves fail."""
    base = float(losses["baseline"])
    span = float(np.ptp(list(losses.values())))
    tol = tolerance * max(1., max(abs(float(v)) for v in losses.values()))
    planes = {}
    for c in cases:
        if c.get("pair_1based"):
            key = tuple(c["pair_1based"])
            planes.setdefault(key, {})[int(c["angle_deg"])] = float(losses[c["id"]])
    passed = 0; equality_error = 0.
    for values in planes.values():
        if set(values) != {15, 30, 45, 60, 75, 90}:
            raise ValueError("Incomplete plane")
        equality_error = max(equality_error, abs(values[90] - base))
        passed += int(abs(values[90] - base) <= tol and
                      min(values[a] for a in (15, 30, 45, 60, 75)) > max(base, values[90]) + tol)
    rank = 1 + sum(float(v) < base - tol for v in losses.values())
    return dict(baseline_rank=rank, success_planes=passed, n_planes=len(planes),
                span=span, endpoint_error=equality_error,
                all_planes_pass=bool(passed == len(planes) and rank == 1 and span > tol))
