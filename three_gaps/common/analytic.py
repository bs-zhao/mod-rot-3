"""Exact analytic logit functions of the four generators, written in torch.

These are re-implementations of the generator formulas (C1/C2/C3 design
matrices and the CR3 CPT-like value) so that torch.func can differentiate
them to any order.  `check_against_dataset` verifies them numerically against
the stored `predictive_true` column of S1.

CR3 conventions at non-differentiable boundary points (Q08):
  * (amount/100)**alpha with amount == 0 is implemented with a `where` so its
    value is 0 and every derivative through the zero branch is 0 (instead of
    +inf for d/d amount).
  * probabilities are clipped to [1e-12, 1-1e-12] exactly like the generator;
    derivatives w.r.t. p at the clipped boundary are therefore 0.
"""
from __future__ import annotations

import numpy as np
import torch

S3 = 3.0 ** 0.5
Q5 = 5.0 ** 0.5 / 2.0

PARAM_NAMES = {
    "C1": ["b1", "b2", "b3", "b4"],
    "C2": ["b1", "b2", "b3", "b4"],
    "C3": ["b1", "b2", "b3", "b4", "b5", "b6"],
    "CR3": ["alpha", "lambda", "gamma", "tau"],
}
X_DIM = {"C1": 3, "C2": 4, "C3": 4, "CR3": 6}


def eta_c1(T, X):
    b1, b2, b3, b4 = T.unbind(-1)
    x1, x2, x3 = X.unbind(-1)
    return b1 + S3 * (b2 * x1 + b3 * x2 + b4 * x3)


def eta_c2(T, X):
    b1, b2, b3, b4 = T.unbind(-1)
    x1a, x1b, x2a, x2b = X.unbind(-1)
    return (b1 + S3 * b2 * (x1a - x1b) + Q5 * b3 * (3 * x1a ** 2 - 3 * x1b ** 2)
            + S3 * b4 * (x2a - x2b))


def eta_c3(T, X):
    b1, b2, b3, b4, b5, b6 = T.unbind(-1)
    x1a, x1b, x2a, x2b = X.unbind(-1)
    return (b1 + S3 * b2 * (x1a - x1b) + Q5 * b3 * (3 * x1a ** 2 - 3 * x1b ** 2)
            + S3 * b4 * (x2a - x2b) + Q5 * b5 * (3 * x2a ** 2 - 3 * x2b ** 2)
            + 3.0 * b6 * (x1a * x2a - x1b * x2b))


def _spow(a, alpha):
    pos = a > 0
    safe = torch.where(pos, a, torch.ones_like(a))
    return torch.where(pos, torch.exp(alpha * torch.log(safe)), torch.zeros_like(a))


def _w(p, g):
    pc = p.clamp(1e-12, 1 - 1e-12)
    return torch.sigmoid(g * torch.log(pc / (1 - pc)))


def eta_cr3(T, X):
    a, lam, g, tau = T.unbind(-1)
    ga, la, pa, gb, lb, pb = X.unbind(-1)
    va = _w(pa, g) * _spow(ga / 100.0, a) - lam * _w(1 - pa, g) * _spow(la / 100.0, a)
    vb = _w(pb, g) * _spow(gb / 100.0, a) - lam * _w(1 - pb, g) * _spow(lb / 100.0, a)
    return tau * (va - vb)


ETA = {"C1": eta_c1, "C2": eta_c2, "C3": eta_c3, "CR3": eta_cr3}


def make_h(model: str, channel: str, mu, sd):
    """Return h(t, x) on STANDARDISED parameters t = (T - mu)/sd."""
    eta = ETA[model]
    mu = torch.as_tensor(mu)
    sd = torch.as_tensor(sd)

    def h(t, x):
        T = mu.to(t) + sd.to(t) * t
        e = eta(T, x)
        if channel == "logit":
            return e
        if channel == "prob":
            return torch.sigmoid(e)
        raise ValueError(channel)

    return h


def check_against_dataset(model, params, x, subject_index, predictive_true, n_subjects=5):
    worst = 0.0
    for s in range(n_subjects):
        m = subject_index == s
        T = torch.tensor(params[s], dtype=torch.float64)
        X = torch.tensor(x[m], dtype=torch.float64)
        p = torch.sigmoid(ETA[model](T.expand(len(X), -1), X)).numpy()
        worst = max(worst, float(np.abs(p - np.asarray(predictive_true[m]).reshape(-1)).max()))
    return worst
