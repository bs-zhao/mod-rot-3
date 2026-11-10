"""Constrained monotone inverse-link profiling, no known sigmoid/logit supplied.

Find a piecewise linear strictly increasing t(y) that becomes well explained
by a rotation-invariant full tensor-sieve. Normalize t(ymax)-t(ymin)=1, and
restrict each segment slope away from zero to avoid the constant-link collapse.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import qr


def hinge_basis(y: np.ndarray, knots: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float).reshape(-1)
    knots = np.asarray(knots, float)
    ds = np.diff(knots)
    if len(knots) < 3 or not np.all(ds > 0):
        raise ValueError('strictly increasing knots required')
    B = np.maximum(0, np.minimum(y[:, None] - knots[:-1][None, :], ds[None, :]))
    # Smooth linear extrapolation instead of clipping unseen Y to endpoints.
    B[:, 0] += np.minimum(y - knots[0], 0)
    B[:, -1] += np.maximum(y - knots[-1], 0)
    return B


def learn_monotone_transform(y, F, n_segments=10, curvature=.01):
    y = np.asarray(y, float).ravel()
    F = np.asarray(F, float)
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(F))):
        raise ValueError('non-finite data')
    if len(y) != len(F) or np.ptp(y) < 1e-7:
        raise ValueError('inverse link requires informative continuous Y')
    # Equally-spaced quantiles minimize empty response segments at saturation.
    knots = np.quantile(y, np.linspace(0, 1, n_segments + 1))
    if len(np.unique(knots)) != len(knots):
        raise ValueError('response has too many ties for a monotone inverse link')
    B = hinge_basis(y, knots)
    # EXACT Euclidean column-space projection, not Ridge: the polynomial
    # Z×X sieve spans the same space under SO(d) rotations, whereas Ridge
    # coefficient penalties are NOT basis invariant for quadratic terms.
    # Rank-revealing pivoted QR handles CR3's duplicate/discrete X columns.
    A = np.column_stack([np.ones(len(F)), F])
    Qorth, Rtri, _ = qr(A, mode='economic', pivoting=True, check_finite=False)
    diag = abs(np.diag(Rtri))
    rank = int(np.count_nonzero(diag > max(diag.max(), 1) * 1e-9))
    residual = B - Qorth[:, :rank] @ (Qorth[:, :rank].T @ B)
    Q = residual.T @ residual / len(y)
    d = np.diff(knots)
    # slopes s_i >= 5% of average slope, with integral slope = 1.
    avg = 1 / np.sum(d)
    min_slope = .05 * avg
    D = np.diff(np.eye(len(d)), axis=0)
    # Link regularizer independent of response scale by transforming to segment increments.
    E = D @ np.diag(d)
    def obj(s):
        v = Q @ s
        rough = E @ s
        return float(s @ v + curvature * (rough @ rough))
    def jac(s): return 2 * (Q @ s + curvature * (E.T @ (E @ s)))
    res = minimize(obj, np.full(len(d), avg), jac=jac, method='SLSQP',
                   bounds=[(min_slope, None)] * len(d),
                   constraints=[{'type':'eq', 'fun':lambda s: float(d@s-1),
                                 'jac':lambda s: d}],
                   options={'ftol':1e-10, 'maxiter':1000})
    if not res.success or min(res.x) < min_slope - 1e-6 or abs(d@res.x - 1) > 1e-6:
        raise RuntimeError(f'monotone transform profile failed: {res.message}')
    return {'knots':knots.tolist(), 'slopes':res.x.tolist(), 'success':True,
            'min_relative_slope':float(min(res.x)/avg),
            'profile_residual':float(np.mean((residual @ res.x)**2)),
            'n_segments':len(d), 'shape':'continuous_strict_monotone_piecewise_linear'}


def apply(y, transform):
    return hinge_basis(y, np.asarray(transform['knots'])) @ np.asarray(transform['slopes'])
