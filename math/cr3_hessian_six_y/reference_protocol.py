"""Verified copy of original CR3 sample/prior analytic Hessian comparator.

This module belongs solely to cr3_hessian_six_y, and never computes
any neural-network derivative. It is copied without changing the
mathematical definitions of the earlier baseline experiment.
"""
from __future__ import annotations
import csv
import json
import hashlib
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import beta, gamma as gamma_distribution, truncnorm
from oracle import NAMES, oracle_hessians, oracle_outputs

def to_builtin(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_builtin(v) for v in value]
    return value


def save_json(path, obj):
    Path(path).write_text(json.dumps(to_builtin(obj), indent=2,
                                     ensure_ascii=False, allow_nan=False),
                          encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cr3(path):
    with np.load(path, allow_pickle=False) as d:
        required = ("parameters", "trial_counts", "subject_index",
                    "trial_index", "x", "y", "predictive_true")
        missing = [key for key in required if key not in d.files]
        if missing:
            raise ValueError(f"Dataset missing arrays: {missing}")
        T = np.asarray(d["parameters"], dtype=np.float64)
        counts = np.asarray(d["trial_counts"], dtype=int)
        if T.ndim != 2 or T.shape[1] != 4 or len(counts) != len(T):
            raise ValueError("Expected CR3-4P with T shape (subjects,4)")
        if not np.all(counts == 500):
            raise ValueError("CR3 experiment requires 500 trials per subject")
        n, m = len(T), 500
        xall = np.asarray(d["x"], dtype=np.float64).reshape(n, m, 6)
        X = xall[0].copy()
        if not np.all(xall == X[None, :, :]):
            raise ValueError("The 500 X stimuli must be identical for all subjects")
        if not np.array_equal(d["subject_index"],
                              np.repeat(np.arange(n), m)):
            raise ValueError("Subject order differs from CR3 generator")
        if not np.array_equal(d["trial_index"],
                              np.tile(np.arange(m), n)):
            raise ValueError("Trial order differs from CR3 generator")
        binary = np.asarray(d["y"], float).reshape(n, m)
        probability = np.asarray(d["predictive_true"], float).reshape(n, m)
        if not np.all(np.isin(binary, [0., 1.])):
            raise ValueError("binary choice must be 0/1")
        if not np.isfinite(probability).all():
            raise ValueError("Non-finite probability in saved dataset")
    return T, X, binary, probability


def prior_quadrature(nodes_per_dim):
    """4D independent marginal Gauss-Legendre integration of ACTUAL CR3 prior.

    alpha ~ Beta(12,3); lambda ~ Gamma(9,scale=.22) truncated [.3,5];
    gamma ~ Beta(9,4); tau ~ Normal(2.2,.55) truncated [.5,4].
    The fixed 500 X points are never re-sampled or continuously integrated.
    """
    legx, legw = leggauss(nodes_per_dim)
    norm_lambda = gamma_distribution.cdf(5., 9., scale=.22) - \
                  gamma_distribution.cdf(.3, 9., scale=.22)
    norm_tau = truncnorm((.5 - 2.2)/.55, (4. - 2.2)/.55,
                         loc=2.2, scale=.55)

    specifications = [
        (0., 1., lambda x: beta.pdf(x, 12, 3)),
        (.3, 5., lambda x: gamma_distribution.pdf(x, 9, scale=.22) / norm_lambda),
        (0., 1., lambda x: beta.pdf(x, 9, 4)),
        (.5, 4., lambda x: norm_tau.pdf(x)),
    ]
    nodes, weights, errors = [], [], []
    for lo, hi, density in specifications:
        xi = (hi - lo) * .5 * legx + (hi + lo) * .5
        wi = (hi - lo) * .5 * legw * density(xi)
        raw_mass = float(wi.sum())
        errors.append(raw_mass - 1.)
        nodes.append(xi)
        weights.append(wi / raw_mass)
    T = np.stack(np.meshgrid(*nodes, indexing="ij"), axis=-1).reshape(-1, 4)
    W = np.prod(np.stack(np.meshgrid(*weights, indexing="ij"), axis=-1),
                axis=-1).ravel()
    W /= W.sum()
    return T, W, errors


class HessianMoments:
    """Weighted moments, preserving the domain of non-existent derivatives."""
    def __init__(self):
        self.mass = np.zeros((10, 10))
        self.total = np.zeros((10, 10))
        self.total_squared = np.zeros((10, 10))

    def add(self, h, row_weights):
        # Each theta point's mass is uniformly split among the 500 trials.
        w = np.asarray(row_weights)[:, None, None, None] / h.shape[1]
        good = np.isfinite(h)
        clean = np.where(good, h, 0.)
        self.mass += np.sum(good * w, axis=(0, 1))
        self.total += np.sum(clean * w, axis=(0, 1))
        self.total_squared += np.sum((clean**2) * w, axis=(0, 1))

    def result(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = self.total / self.mass
            sq = self.total_squared / self.mass
        mu[self.mass == 0] = np.nan
        sq[self.mass == 0] = np.nan
        std = np.sqrt(np.maximum(sq - mu**2, 0.))
        return dict(
            strict_mean=np.where(self.mass >= 1 - 1e-10, mu, np.nan),
            conditional_mean=mu,
            conditional_rms=np.sqrt(sq),
            conditional_std=std,
            fraction_defined=self.mass,
        )


def oracle_moments(T_eval, X, weights, batch_size=32):
    acc_eta, acc_p = HessianMoments(), HessianMoments()
    n = len(T_eval)
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        H_eta, H_p = oracle_hessians(T_eval[lo:hi], X)
        acc_eta.add(H_eta, weights[lo:hi])
        acc_p.add(H_p, weights[lo:hi])
        if lo == 0 or hi == n or (lo // batch_size) % 50 == 0:
            print(f"  oracle Hessian batch {hi}/{n}", flush=True)
    p_res = acc_p.result()
    return dict(binary=p_res, logit=acc_eta.result(), probability=p_res)


def comparison(oracle, fitted):
    result = {}
    for block, rows, cols in (
        ("theta_theta", slice(0, 4), slice(0, 4)),
        ("theta_x", slice(0, 4), slice(4, 10)),
        ("x_x", slice(4, 10), slice(4, 10)),
        ("all", slice(0, 10), slice(0, 10)),
    ):
        a = oracle["conditional_mean"][rows, cols]
        b = fitted["conditional_mean"][rows, cols]
        valid = (oracle["fraction_defined"][rows, cols] > 0) & \
                np.isfinite(a) & np.isfinite(b)
        diff = (a - b)[valid]
        result[block] = dict(
            comparable_entries=int(valid.sum()),
            mae=float(np.mean(np.abs(diff))) if len(diff) else None,
            rmse=float(np.sqrt(np.mean(diff**2))) if len(diff) else None,
            max_abs=float(np.max(np.abs(diff))) if len(diff) else None,
        )
    return result


def write_matrix(path, matrix):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + list(NAMES))
        for name, row in zip(NAMES, matrix):
            writer.writerow([name] + ["" if not np.isfinite(v) else
                                      f"{v:.12g}" for v in row])


