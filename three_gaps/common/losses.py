"""M3 catalogue: scalar complexity of a derivative view B of shape (n, d, C).

n = aggregation points, d = rotated U axes, C = context cells.  Every loss is
invariant to per-axis sign, axis permutation and (because B is already in
z-scored candidate coordinates) per-axis scale.  Lower = simpler.
"""
from __future__ import annotations

import torch

EPS = 1e-12


def _E2(B):
    return (B * B).mean(0)  # (d, C)


def rmsL1(B):
    E2 = _E2(B)
    return E2.clamp_min(0).sqrt().sum() / (E2.sum().sqrt() + EPS)


def absL1(B):
    A = B.abs().mean(0)
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
    return v.clamp_min(0).sqrt().sum() / (v.sum().sqrt() + EPS)


LOSSES = {f.__name__: f for f in (rmsL1, absL1, colL1L2, rowNeff, cellNeff, softL0, quartimax,
                                  localL1, localHoyer, kurtMax, kurtMin, varL1)}
