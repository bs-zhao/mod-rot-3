"""Rotation-equivariant numerical estimators, not behavioral model equations.

Polynomial kernel basis and Nystrom RBF kernel ridge use all W/Y rows.
Subject-heldout prediction error is computed before the all-subject refit.
"""
from __future__ import annotations

import math
from itertools import combinations_with_replacement

import numpy as np
from scipy.linalg import solve


def powers_for(d, degree):
    powers = [np.zeros(d, dtype=int)]
    for k in range(1, degree + 1):
        for ids in combinations_with_replacement(range(d), k):
            powers.append(np.bincount(ids, minlength=d))
    return np.asarray(powers)


def weights_for(powers):
    return np.asarray([math.sqrt(math.factorial(int(p.sum())) /
                               math.prod(math.factorial(int(v)) for v in p))
                       for p in powers])


def monomials(x, powers, weights):
    out = np.ones((len(x), len(powers)))
    for j in range(x.shape[1]):
        out *= x[:, j, None] ** powers[None, :, j]
    return out * weights


class Scaler:
    def __init__(self, w, d, chunk=65536, row_mask=None):
        # One U scale is essential: per-axis restandardization would change
        # the geometry when rotating non-spherical empirical U.
        total = np.zeros(w.shape[1]); squares = total.copy(); count = 0
        for a in range(0, len(w), chunk):
            x = np.asarray(w[a:a + chunk], dtype=float)
            if row_mask is not None:
                x = x[row_mask[a:a + chunk]]
            total += x.sum(0); squares += np.sum(x*x, axis=0)
            count += len(x)
        if count < 2:
            raise ValueError("Not enough training rows to scale coordinates")
        self.mean = total / count
        variance = np.maximum(squares / count - self.mean**2, 0.)
        self.scale = np.sqrt(variance)
        self.scale[:d] = math.sqrt(float(variance[:d].mean()))
        self.scale[self.scale < 1e-10] = 1.

    def __call__(self, x):
        return (np.asarray(x, dtype=float) - self.mean) / self.scale


class Polynomial:
    def __init__(self, scaler, degree):
        self.scaler = scaler
        self.degree = degree
        self.powers = powers_for(len(scaler.mean), degree)
        self.weights = weights_for(self.powers)
        self.name = f"poly{degree}"

    def features(self, w):
        return monomials(self.scaler(w), self.powers, self.weights)

    def set_coef(self, coef):
        self.coef = coef
        d = len(self.scaler.mean)
        self.dp = powers_for(d, max(0, self.degree - 1))
        self.hp = powers_for(d, max(0, self.degree - 2))
        self.dw = weights_for(self.dp); self.hw = weights_for(self.hp)
        di = {tuple(p): i for i, p in enumerate(self.dp)}
        hi = {tuple(p): i for i, p in enumerate(self.hp)}
        self.dc = np.zeros((len(self.dp), d, coef.shape[1]))
        self.hc = np.zeros((len(self.hp), d, d, coef.shape[1]))
        for row, p in enumerate(self.powers):
            for i in range(d):
                if p[i] == 0:
                    continue
                q = p.copy(); q[i] -= 1
                r = di[tuple(q)]
                self.dc[r, i] += coef[row] * self.weights[row] * p[i] / self.dw[r]
                for j in range(d):
                    if q[j] == 0:
                        continue
                    v = q.copy(); v[j] -= 1
                    s = hi[tuple(v)]
                    self.hc[s, i, j] += (coef[row] * self.weights[row] * p[i] * q[j] /
                                          self.hw[s])

    def derivatives(self, w):
        x = self.scaler(w)
        g = np.einsum("np,pic->nci", monomials(x, self.dp, self.dw), self.dc)
        h = np.einsum("np,pijc->ncij", monomials(x, self.hp, self.hw), self.hc)
        return (g / self.scaler.scale,
                h / self.scaler.scale[None, None, :, None] /
                self.scaler.scale[None, None, None, :])


class RBF:
    """Nystrom approximation to RBF kernel ridge; no GP uncertainty claim."""
    def __init__(self, scaler, centers, lengthscale):
        self.scaler = scaler
        self.centers = scaler(centers)
        self.lengthscale = float(lengthscale)
        self.name = f"rbf{lengthscale:g}"
        k = self.kernel(self.centers)
        values, vectors = np.linalg.eigh(k)
        good = values > max(float(values.max()) * 1e-10, 1e-12)
        self.transform = vectors[:, good] / np.sqrt(values[good])

    def kernel(self, x):
        ds = np.maximum(np.sum(x*x, axis=1)[:, None] +
                        np.sum(self.centers**2, axis=1)[None, :] -
                        2*x @ self.centers.T, 0.)
        return np.exp(-ds / (2*self.lengthscale**2))

    def features(self, w):
        x = self.scaler(w)
        return np.column_stack([np.ones(len(x)), self.kernel(x) @ self.transform])

    def set_coef(self, coef):
        self.coef = coef
        self.beta = self.transform @ coef[1:]
        self.first = self.centers[:, :, None] * self.beta[:, None, :]
        self.second = (self.centers[:, :, None, None] *
                       self.centers[:, None, :, None] * self.beta[:, None, None, :])

    def derivatives(self, w):
        x = self.scaler(w)
        k = self.kernel(x)
        n, d = x.shape; c = self.beta.shape[1]; l2 = self.lengthscale**2
        s = k @ self.beta
        t = (k @ self.first.reshape(len(self.centers), -1)).reshape(n, d, c)
        v = (k @ self.second.reshape(len(self.centers), -1)).reshape(n, d, d, c)
        g = (t - x[:, :, None]*s[:, None, :]) / l2
        h = (v - x[:, :, None, None]*t[:, None, :, :] -
             x[:, None, :, None]*t[:, :, None, :] +
             x[:, :, None, None]*x[:, None, :, None]*s[:, None, None, :]) / l2**2
        h -= np.eye(d)[None, :, :, None] * s[:, None, None, :] / l2
        g = np.moveaxis(g, -1, 1) / self.scaler.scale
        h = np.moveaxis(h, -1, 1) / self.scaler.scale[None, None, :, None]
        h = h / self.scaler.scale[None, None, None, :]
        return g, h


def fit_all_rows(model, w, y, sid, train_subjects, chunk=8192,
                 ridges=(1e-6, 1e-4, 1e-2)):
    """Sufficient statistics include EVERY row; no trial/subject subsampling.

    A subject split selects regularization using prediction alone. Frozen
    chosen regularization is then used for the all-subject refit. Candidate
    case identities and their metric rankings never select regularization.
    """
    p = model.features(w[:1]).shape[1]
    ga = np.zeros((p, p)); ca = np.zeros((p, y.shape[1]))
    gt = ga.copy(); ct = ca.copy(); ya = np.zeros(y.shape[1]); yt = ya.copy()
    train_mask = np.zeros(int(sid.max()) + 1, dtype=bool)
    train_mask[train_subjects] = True
    nt = 0
    for a in range(0, len(w), chunk):
        b = min(a + chunk, len(w))
        phi = model.features(w[a:b]); z = y[a:b]
        mask = train_mask[sid[a:b]]
        ga += phi.T @ phi; ca += phi.T @ z; ya += np.sum(z*z, axis=0)
        pt = phi[mask]; zt = z[mask]
        gt += pt.T @ pt; ct += pt.T @ zt; yt += np.sum(zt*zt, axis=0)
        nt += int(mask.sum())
        if not np.isfinite(ga).all():
            raise ArithmeticError("Numerically unstable feature basis")
    if nt == 0 or nt == len(w):
        raise ValueError("Need disjoint nonempty subject train/validation sets")
    gv, cv, yv = ga - gt, ca - ct, ya - yt
    ytr = y[train_mask[sid]]
    variance = np.maximum(ytr.var(axis=0), 1e-12)
    penalty = np.eye(p); penalty[0, 0] = 0.

    def estimate(g, cross, ridge):
        scale = max(float(np.trace(g[1:, 1:])) / max(p - 1, 1), 1e-12)
        return solve(g + ridge*scale*penalty, cross, assume_a="sym")

    trials = []
    for ridge in ridges:
        coef = estimate(gt, ct, ridge)
        sse = yv - 2*np.sum(coef*cv, axis=0) + np.sum(coef*(gv @ coef), axis=0)
        nmse = np.maximum(sse, 0.) / (len(w) - nt) / variance
        trials.append((float(nmse.mean()), ridge, nmse))
    _, ridge, nmse = min(trials, key=lambda t: t[0])
    model.set_coef(estimate(ga, ca, ridge))
    return dict(estimator=model.name, feature_count=p, ridge=ridge,
                validation_nmse=nmse.tolist(), train_subjects=len(train_subjects),
                validation_subjects=len(train_mask) - len(train_subjects),
                final_fit_rows=len(w), final_fit_subjects=len(train_mask),
                derivative_accuracy_verified=False)
