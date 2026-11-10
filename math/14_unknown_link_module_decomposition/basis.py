"""Prespecified low-order stimulus dictionary and tensor Z/X regressors.

Only W is read. No oracle formulas or response values are imported here.
For CR3, we evaluate at *observed valid lottery rows*: no X perturbation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Dictionary:
    zdim: int
    xdim: int
    zmean: np.ndarray
    zstd: np.ndarray
    xmean: np.ndarray
    xstd: np.ndarray
    xterms: list[tuple[int, ...]]
    zterms: list[tuple[int, ...]]
    xgroup: list[str]

    @classmethod
    def fit(cls, W: np.ndarray, zdim: int, zdegree: int = 1, xdegree: int = 2):
        W = np.asarray(W, float)
        if W.ndim != 2 or zdim < 1 or zdim >= W.shape[1]:
            raise ValueError('W must contain Z and X columns')
        if zdegree not in (1, 2) or xdegree not in (1, 2):
            raise ValueError('Allowed Z and X degrees are 1 or 2')
        p = W.shape[1] - zdim
        z = W[:, :zdim]
        x = W[:, zdim:]
        zm = z.mean(axis=0); zs = np.maximum(z.std(axis=0), 1e-8)
        xm = x.mean(axis=0); xs = np.maximum(x.std(axis=0), 1e-8)
        zterms = [(i,) for i in range(zdim)]
        if zdegree == 2:
            zterms.extend((i, j) for i in range(zdim) for j in range(i, zdim))
        xterms = [(j,) for j in range(p)]
        if xdegree == 2:
            xterms.extend((i, j) for i in range(p) for j in range(i, p))
        groups = ['x' + '_'.join(map(str, sorted(set(t)))) for t in xterms]
        return cls(zdim, p, zm, zs, xm, xs, xterms, zterms, groups)

    def _features(self, values: np.ndarray, terms: list[tuple[int, ...]]) -> np.ndarray:
        return np.column_stack([np.prod(values[:, term], axis=1) for term in terms])

    def features(self, W: np.ndarray, R: np.ndarray | None = None):
        W = np.asarray(W, float)
        z = (W[:, :self.zdim] - self.zmean) / self.zstd
        if R is not None:
            z = z @ R.T
        x = (W[:, self.zdim:] - self.xmean) / self.xstd
        zf = self._features(z, self.zterms)
        xf = self._features(x, self.xterms)
        tensor = np.einsum('ni,nj->nij', zf, xf).reshape(len(W), -1)
        return np.column_stack([zf, xf, tensor]), (zf.shape[1], xf.shape[1])

    def groups_from_interactions(self, coefficients: np.ndarray) -> np.ndarray:
        """Axis x stimulus-module energy; shared Z interactions allocated equally."""
        p, q = len(self.zterms), len(self.xterms)
        assert coefficients.size == p * q
        c = coefficients.reshape(p, q)
        keys = list(dict.fromkeys(self.xgroup))
        strength2 = np.zeros((self.zdim, len(keys)))
        groups = {k: i for i, k in enumerate(keys)}
        for a, zterm in enumerate(self.zterms):
            unique = set(zterm)
            for b, group in enumerate(self.xgroup):
                for i in unique:
                    strength2[i, groups[group]] += float(c[a, b] ** 2) / len(unique)
        return np.sqrt(strength2)


def givens(d: int, i: int, j: int, angle: float) -> np.ndarray:
    if not 0 <= i < d or not 0 <= j < d or i == j:
        raise ValueError('invalid Givens axes')
    t = np.deg2rad(float(angle)); R = np.eye(d)
    R[i, i] = R[j, j] = np.cos(t)
    R[i, j] = np.sin(t); R[j, i] = -np.sin(t)
    if not np.allclose(R.T @ R, np.eye(d), atol=1e-12, rtol=0):
        raise AssertionError('rotation not orthogonal')
    if not np.isclose(np.linalg.det(R), 1, atol=1e-12):
        raise AssertionError('rotation does not have determinant +1')
    return R


def complexity(strength: np.ndarray):
    s = np.asarray(strength, float)
    energy = np.sum(s * s)
    if not np.isfinite(energy) or energy <= 1e-15:
        return {'module_l21': None, 'module_effective_partners': None}
    w = s / np.sqrt(energy)
    # L21 counts nonzero axis--stimulus modules, normalized by invariant energy.
    col = np.sum(w * w, axis=0)
    probs = np.divide(w * w, col[None, :], out=np.zeros_like(w), where=col[None, :] > 1e-15)
    h = -np.sum(probs * np.log(np.maximum(probs, 1e-20)), axis=0)
    return {'module_l21': float(np.sum(w)),
            'module_effective_partners': float(np.sum(np.where(col > 1e-15, np.exp(h), 0))),
            'per_axis_strength': np.linalg.norm(s, axis=1).tolist(),
            'module_matrix': s.tolist()}
