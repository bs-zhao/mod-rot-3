"""Stable, theta-free interface shared by independent rotation plugins.

This module intentionally does not import the baseline model or subject truth.
Only the runner accesses held-out theta, *after* persisting the rotation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize


@dataclass(frozen=True)
class RotationInput:
    """Fitting-only, subject-aligned data. No theta or TEST subjects."""

    q: np.ndarray                 # FIT participants x k, already PCA-whitened
    x: np.ndarray                 # FIT participants x padded trials x x_dim
    y: np.ndarray                 # FIT participants x padded trials (0/1)
    mask: np.ndarray              # True for observed trials

    def __post_init__(self):
        q, x, y, m = (np.asarray(v) for v in (self.q, self.x, self.y, self.mask))
        if q.ndim != 2 or q.shape[1] < 2 or x.ndim != 3 or y.ndim != 2:
            raise ValueError('Expected q (N,K), x (N,T,D), y (N,T) with K>=2')
        if x.shape[:2] != y.shape or y.shape != m.shape or q.shape[0] != x.shape[0]:
            raise ValueError('Z/X/Y/mask must be aligned by subject and trial')
        if m.dtype != np.bool_ or not np.all(m.any(axis=1)):
            raise ValueError('mask must be bool with at least one observed trial per subject')
        if not (np.isfinite(q).all() and np.isfinite(x[m]).all() and np.isfinite(y[m]).all()):
            raise ValueError('Observed ZXY must be finite')
        if not np.all(np.isin(y[m], (0, 1))):
            raise ValueError('Binary choice Y must contain only 0/1 for observed trials')
        # Accidental in-place changes by one plugin must not alter later methods.
        for value in (q, x, y, m):
            value.setflags(write=False)

    @property
    def k(self) -> int:
        return int(self.q.shape[1])

    @property
    def n_subjects(self) -> int:
        return int(self.q.shape[0])

    def rotated(self, r: np.ndarray) -> np.ndarray:
        return self.q @ np.asarray(r, dtype=float).T


@dataclass
class RotationResult:
    R: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)
    extra_arrays: dict[str, np.ndarray] = field(default_factory=dict)


def skew_matrix(angles: np.ndarray, k: int) -> np.ndarray:
    """Free antisymmetric parameterization of SO(k)."""
    angles = np.asarray(angles, dtype=float).reshape(-1)
    if len(angles) != k * (k - 1) // 2:
        raise ValueError('Expected k*(k-1)//2 rotation angles')
    skew = np.zeros((k, k), dtype=float)
    i, j = np.triu_indices(k, 1)
    skew[i, j] = angles
    skew[j, i] = -angles
    return skew


def orthogonal_from_angles(angles: np.ndarray, k: int) -> np.ndarray:
    return expm(skew_matrix(angles, k))


def validate_result(result: RotationResult, k: int, *, atol: float = 1e-7) -> RotationResult:
    if not isinstance(result, RotationResult):
        raise TypeError('Plugin must return RotationResult(R, info={}, extra_arrays={})')
    r = np.asarray(result.R, dtype=np.float64)
    if r.shape != (k, k) or not np.isfinite(r).all():
        raise ValueError(f'Plugin returned invalid {k}x{k} rotation matrix')
    if not np.allclose(r @ r.T, np.eye(k), atol=atol, rtol=0) or not np.isclose(
        np.linalg.det(r), 1.0, atol=atol
    ):
        raise ValueError('Rotation must be in SO(k): R R.T=I and det(R)=+1')
    for key, v in result.extra_arrays.items():
        if not isinstance(key, str) or not key.isidentifier():
            raise ValueError('extra_arrays keys must be valid identifiers')
        if np.asarray(v).dtype.kind == 'O':
            raise ValueError('Object arrays are not allowed in saved method artifacts')
    result.R = r
    return result


def optimize_objective(
    data: RotationInput,
    objective: Callable[[RotationInput, np.ndarray, Mapping[str, Any]], float],
    config: Mapping[str, Any],
) -> RotationResult:
    """Reusable multistart optimizer; plugins can ignore it and supply fit().

    The user-facing scalar objective is evaluated on U=R q (column convention),
    i.e., rows use q @ R.T. No theta, known simulator, or NP decoder is used.
    """
    k = data.k
    starts = int(config.get('starts', 4))
    maxiter = int(config.get('maxiter', 200))
    seed = int(config.get('seed', 31))
    method = str(config.get('optimizer', 'Powell'))
    if starts < 1 or maxiter < 1 or method not in ('Powell', 'BFGS', 'Nelder-Mead'):
        raise ValueError('Invalid optimizer/starts/maxiter')
    rng = np.random.default_rng(seed)
    nd = k * (k - 1) // 2
    inits = [np.zeros(nd)] + [rng.normal(0, 0.35, nd) for _ in range(starts-1)]
    records = []

    def loss(a):
        value = float(objective(data, orthogonal_from_angles(a, k), config))
        return value if np.isfinite(value) else 1e100

    for initial in inits:
        opt = minimize(loss, initial, method=method,
                       options={'maxiter': maxiter})
        val = float(opt.fun)
        if not np.isfinite(val) or val >= 1e99:
            continue
        records.append((val, orthogonal_from_angles(opt.x, k), opt))
    if not records:
        raise RuntimeError('All optimizer starts returned nonfinite losses')
    best = min(records, key=lambda item: item[0])
    result = RotationResult(
        R=best[1],
        info={'fit_loss': best[0], 'starts': starts, 'successful_starts': len(records),
              'optimizer_converged': bool(best[2].success),
              'optimizer_message': str(best[2].message),
              'best_nfev': int(best[2].nfev), 'optimizer': method},
    )
    return validate_result(result, k)