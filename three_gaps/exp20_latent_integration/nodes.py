"""EXP20B integration nodes: Gaussian measures on the physical latent space and MC / scrambled-Sobol (RQMC) / Gauss-Hermite integrators.

All nodes are in epsilon space (standard normal) and mapped to the physical latent Z = mu + eps @ L.T (or other reference measure);
the decoder is queried at V = Z - mu_train.  No labels / T are used.  Sobol here is an INTEGRATION sequence, not Sobol sensitivity.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import qmc

U_CLIP = 1e-12


def mc_eps(N, d, seed):
    """independent standard-normal nodes; nested prefixes (first n rows are the n-node sample of the same seed)."""
    return np.random.default_rng(seed).standard_normal((N, d))


def rqmc_eps(N, d, seed):
    """randomised (scrambled) Sobol nodes mapped through the inverse normal CDF.  N must be a power of two (prefixes 2^m are nested).

    returns eps and the number of clipped uniforms (clipping only guards exact 0/1)."""
    if N & (N - 1):
        raise ValueError("RQMC needs N = 2^m")
    u = qmc.Sobol(d=d, scramble=True, seed=seed).random(N)
    bad = int(((u < U_CLIP) | (u > 1 - U_CLIP)).sum())
    return stats.norm.ppf(np.clip(u, U_CLIP, 1 - U_CLIP)), bad


def gh_eps(m, d):
    """tensor-product Gauss-Hermite (probabilists' normalisation) nodes and weights for N(0, I_d): m^d nodes, weights sum to 1."""
    x, w = np.polynomial.hermite.hermgauss(m)
    x, w = np.sqrt(2.0) * x, w / np.sqrt(np.pi)
    grids = np.meshgrid(*([x] * d), indexing="ij")
    ws = np.meshgrid(*([w] * d), indexing="ij")
    eps = np.stack([g.ravel() for g in grids], 1)
    wt = np.prod(np.stack([g.ravel() for g in ws], 1), 1)
    return eps, wt


def cholesky_cov(V_train):
    """population covariance (ddof=0) of the TRAIN latent and its Cholesky factor (jitter is reported, never silent)."""
    Vc = V_train - V_train.mean(0)
    S = Vc.T @ Vc / len(V_train)
    w = np.linalg.eigvalsh(S)
    jitter = 0.0 if w[0] > 1e-10 * w[-1] else 1e-8 * w[-1]
    return S, np.linalg.cholesky(S + jitter * np.eye(len(S))), w, jitter


def measure_V(kind, eps, S, L, mu_z, c=1.0):
    """map epsilon nodes to V = Z - mu_train for the reference measures.

    P1: N(mu, Sigma)          P2: N(mu, diag Sigma)       P3: N(0, I) in the ORIGINAL Z coordinates (V = eps - mu)
    P4: N(mu, tr(Sigma)/d I)  P6: N(mu, c^2 Sigma)        P5 uses the P1 nodes with truncation weights (see truncation_weights)."""
    d = S.shape[0]
    if kind == "P1":
        return eps @ L.T
    if kind == "P2":
        return eps * np.sqrt(np.diag(S))
    if kind == "P3":
        return eps - mu_z
    if kind == "P4":
        return eps * np.sqrt(np.trace(S) / d)
    if kind == "P6":
        return c * (eps @ L.T)
    raise ValueError(kind)


def truncation_weights(eps, q=0.95):
    """P5: conditional Gaussian inside the Mahalanobis radius with chi-square_d quantile q; weights are 0/1 (never rotation dependent)."""
    d = eps.shape[1]
    r2 = (eps ** 2).sum(1)
    return (r2 <= stats.chi2.ppf(q, d)).astype(float)
