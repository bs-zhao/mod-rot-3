"""M3 catalogue: scalar complexity of a derivative view B of shape (n, d, C).

n = aggregation points, d = rotated U axes, C = context cells.  Every loss is
invariant to per-axis sign, axis permutation and (because B is already in
z-scored candidate coordinates) per-axis scale.  Lower = simpler.
"""
from __future__ import annotations

import numpy as np
import torch

EPS = 1e-12


def _E2(B):
    return (B * B).mean(0)  # (d, C)


def _ssqrt(E2):
    # smooth sqrt: finite gradient at exactly-zero cells (relative epsilon keeps scale invariance)
    return (E2 + 1e-12 * E2.mean().detach() + 1e-30).sqrt()


def rmsL1(B):
    E2 = _E2(B)
    return _ssqrt(E2).sum() / (E2.sum().sqrt() + EPS)


def absL1(B):
    A = (B * B + 1e-24).sqrt().mean(0)
    return A.sum() / ((A * A).sum().sqrt() + EPS)


def colL1L2(B):
    M = _E2(B).clamp_min(EPS).sqrt()
    col2 = (M * M).sum(0)
    w = col2 / col2.sum()
    return (w * M.sum(0) / col2.sqrt()).sum()


def rowNeff(B):
    E2 = _E2(B) + EPS
    row = E2.sum(1)
    q = E2 / row[:, None]
    ent = -(q * q.log()).sum(1)
    w = row / row.sum()
    return (w * ent.exp()).sum()


def cellNeff(B):
    E2 = _E2(B) + EPS
    q = E2 / E2.sum()
    return (-(q * q.log()).sum()).exp()


def softL0(B):
    E2 = _E2(B)
    tau2 = 0.01 * E2.mean()
    return (E2 / (E2 + tau2 + EPS)).sum()


def quartimax(B):
    E2 = _E2(B)
    return -(E2 * E2).sum() / (E2.sum() ** 2 + EPS)


def localL1(B):
    return B.abs().sum((1, 2)).mean() / ((B * B).sum((1, 2)).mean().sqrt() + EPS)


def localHoyer(B):
    l1 = B.abs().sum((1, 2))
    l2 = (B * B).sum((1, 2)).sqrt()
    return (l1 * l2).sum() / ((l2 * l2).sum() + EPS)


def _rowkurt(B):
    m2 = (B * B).mean((0, 2))
    m4 = (B ** 4).mean((0, 2))
    w = m2 / (m2.sum() + EPS)
    return (w * m4 / (m2 * m2 + EPS)).sum()


def kurtMax(B):
    return -_rowkurt(B)


def kurtMin(B):
    return _rowkurt(B)


def varL1(B):
    v = B.var(0, unbiased=False)
    return _ssqrt(v).sum() / (v.sum().sqrt() + EPS)


def _softL0k(k):
    def f(B):
        E2 = _E2(B)
        tau2 = (k * k) * E2.mean()
        return (E2 / (E2 + tau2 + EPS)).sum()
    f.__name__ = f"softL0_{int(round(k * 100)):02d}"
    return f


def logSp(B):
    E2 = _E2(B)
    return torch.log(E2 / (E2.mean() + EPS) + 1e-3).mean()


def _colLog(delta):
    def f(B):
        E2 = _E2(B)                       # (d, C) axis x context energies
        col = E2.sum(0)                   # context energy
        q = E2 / (col + EPS)              # share of each axis in that context
        w = col / (col.sum() + EPS)
        return (w * torch.log(q + delta).sum(0)).sum()
    f.__name__ = f"colLog{int(round(-np.log10(delta)))}"
    return f


LOSSES = {f.__name__: f for f in (rmsL1, absL1, colL1L2, rowNeff, cellNeff, softL0, quartimax,
                                  localL1, localHoyer, kurtMax, kurtMin, varL1, logSp,
                                  _softL0k(0.03), _softL0k(0.1), _softL0k(0.3),
                                  _colLog(1e-1), _colLog(1e-2), _colLog(1e-3))}
