"""Data-only nonparametric/sieve regression and Hessian moments for CR3.

NO imports from the analytic generator or other experiment folders.

Fit a tensor-product polynomial sieve by regularized least squares:
 m(t,x) ~= phi_T(t)^T B phi_X(x).
The polynomial's Hessian is differentiated algebraically. This is an
ESTIMATOR under smoothness/model-approximation assumptions, not a direct
identification of an arbitrary function from finitely many observations.
X-derivatives are extrapolative because the bank is a fixed finite set.
"""
from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np


def exponent_table(dim: int, degree: int) -> np.ndarray:
    rows = [np.zeros(dim, dtype=int)]
    for k in range(1, degree + 1):
        for comb in combinations_with_replacement(range(dim), k):
            rows.append(np.bincount(comb, minlength=dim))
    return np.asarray(rows, dtype=int)


def feature_matrix(points, powers, center, scale, derivatives=()):
    """Monomials and exact partial derivatives in ORIGINAL input units."""
    points = np.asarray(points, dtype=np.float64)
    z = (points - center) / scale
    d = points.shape[1]
    counts = np.bincount(derivatives, minlength=d)
    matrix = np.ones((len(points), len(powers)), dtype=float)
    for k in range(d):
        exponents = powers[:, k]
        r = int(counts[k])
        factors = np.ones(len(powers), dtype=float)
        for j in range(r):
            factors *= np.maximum(exponents - j, 0)
        exponents_new = np.maximum(exponents - r, 0)
        matrix *= (z[:, k, None] ** exponents_new[None, :])
        matrix *= (factors / scale[k] ** r)[None, :]
    return matrix


class TensorPolynomial:
    """Shared X/T design, one independent regression fit per response channel."""

    def __init__(self, T, X, degree_t=4, degree_x=3,
                 ridge_t=0.01, ridge_x=0.01, seed=20261106,
                 train_fraction=0.8):
        T, X = np.asarray(T, float), np.asarray(X, float)
        self.T, self.X = T, X
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(T))
        n_train = int(np.floor(len(T) * train_fraction))
        self.train_idx, self.test_idx = indices[:n_train], indices[n_train:]
        if n_train < 10 or not len(self.test_idx):
            raise ValueError("Need at least 10 train and one test subject")
        self.t_powers = exponent_table(4, degree_t)
        self.x_powers = exponent_table(6, degree_x)
        self.t_center = (T[self.train_idx].max(0) + T[self.train_idx].min(0)) / 2
        self.t_scale = np.maximum(
            (T[self.train_idx].max(0) - T[self.train_idx].min(0)) / 2, 1e-10)
        self.x_center = (X.max(0) + X.min(0)) / 2
        self.x_scale = np.maximum((X.max(0) - X.min(0)) / 2, 1e-10)
        self.phi_t = feature_matrix(
            T, self.t_powers, self.t_center, self.t_scale)
        self.phi_x = feature_matrix(
            X, self.x_powers, self.x_center, self.x_scale)
        train = self.phi_t[self.train_idx]
        self.t_norm = np.maximum(np.linalg.norm(train, axis=0), 1e-12)
        self.x_norm = np.maximum(np.linalg.norm(self.phi_x, axis=0), 1e-12)
        self.Ut, self.St, self.Vt = np.linalg.svd(
            train / self.t_norm, full_matrices=False)
        self.Ux, self.Sx, self.Vx = np.linalg.svd(
            self.phi_x / self.x_norm, full_matrices=False)
        self.ridge_t, self.ridge_x = ridge_t, ridge_x

    def fit(self, Y):
        """Input ONLY numeric (T,X,Y); no true generating formula consulted."""
        Y = np.asarray(Y, float)
        if Y.shape != (len(self.T), len(self.X)):
            raise ValueError(f"Y must have shape {(len(self.T), len(self.X))}")
        C = self.Ut.T @ (Y[self.train_idx] @ self.Ux)
        shrink_t = self.St / (self.St**2 + self.ridge_t)
        shrink_x = self.Sx / (self.Sx**2 + self.ridge_x)
        coef = self.Vt.T @ (
            shrink_t[:, None] * C * shrink_x[None, :]) @ self.Vx
        return coef / self.t_norm[:, None] / self.x_norm[None, :]

    def holdout_metrics(self, Y, coefficient):
        prediction = self.phi_t[self.test_idx] @ coefficient @ self.phi_x.T
        observed = np.asarray(Y, float)[self.test_idx]
        return dict(
            n_train=len(self.train_idx),
            n_test=len(self.test_idx),
            rmse=float(np.sqrt(np.mean((prediction - observed)**2))),
            mae=float(np.mean(np.abs(prediction - observed))),
            predicted_min=float(np.min(prediction)),
            predicted_max=float(np.max(prediction)),
        )

    def derivative_moments(self, points, powers, center, scale,
                           derivatives, weights):
        f = feature_matrix(points, powers, center, scale, derivatives)
        weights = np.asarray(weights, float)
        weights = weights / np.sum(weights)
        return weights @ f, f.T @ (f * weights[:, None])

    def expected_hessian(self, coefficient, T_eval, t_weights):
        """All 10x10 derivative moments, including RMS and SD.

        Uses factorization of the empirical T x fixed-X product measure.
        Undefined oracle amount derivatives are tracked via 'fraction_defined'
        and 'strict_mean'; the polynomial itself is smooth everywhere.
        """
        T_eval = np.asarray(T_eval, float)
        tw = np.asarray(t_weights, float)
        tw = tw / tw.sum()
        B = np.asarray(coefficient, float)
        cache_t, cache_x = {}, {}

        def moments_t(derivatives):
            key = tuple(sorted(derivatives))
            if key not in cache_t:
                cache_t[key] = self.derivative_moments(
                    T_eval, self.t_powers, self.t_center, self.t_scale,
                    key, tw)
            return cache_t[key]

        def moments_x(derivatives, valid):
            key = (tuple(sorted(derivatives)), valid.tobytes())
            if key not in cache_x:
                w = valid.astype(float)
                cache_x[key] = self.derivative_moments(
                    self.X, self.x_powers, self.x_center, self.x_scale,
                    key[0], w)
            return cache_x[key]

        mean = np.empty((10, 10))
        rms = np.empty((10, 10))
        std = np.empty((10, 10))
        fraction = np.empty((10, 10))
        mean_unmasked = np.empty((10, 10))
        all_valid = np.ones(len(self.X), dtype=bool)
        for i in range(10):
            for j in range(i, 10):
                dt = tuple(v for v in (i, j) if v < 4)
                dx = tuple(v - 4 for v in (i, j) if v >= 4)
                valid = all_valid.copy()
                for x_idx in dx:
                    if x_idx in (0, 1, 3, 4):
                        valid &= (self.X[:, x_idx] > 0.)
                mt, qt = moments_t(dt)
                mx, qx = moments_x(dx, valid)
                value = float(mt @ B @ mx)
                squared = float(np.sum((B.T @ qt @ B) * qx))
                spread = max(squared - value**2, 0.)
                frac = float(valid.mean())
                # Raw model-implied value, even when the TRUE derivative
                # is undefined at zero amounts (separate diagnostic only).
                xu, _ = moments_x(dx, all_valid)
                raw = float(mt @ B @ xu)
                for a, b in ((i, j), (j, i)):
                    mean[a, b] = value
                    rms[a, b] = np.sqrt(max(squared, 0.))
                    std[a, b] = np.sqrt(spread)
                    fraction[a, b] = frac
                    mean_unmasked[a, b] = raw
        strict = np.where(fraction > 1. - 1e-12, mean, np.nan)
        return dict(
            strict_mean=strict,
            conditional_mean=mean,
            conditional_rms=rms,
            conditional_std=std,
            fraction_defined=fraction,
            model_unmasked_mean=mean_unmasked,
        )
