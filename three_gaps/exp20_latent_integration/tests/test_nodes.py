"""Selftests for the integrators: weights, constants, Gaussian moments, covariance mapping, prior-vs-V mean mapping, truncation.

python three_gaps/exp20_latent_integration/tests/test_nodes.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nodes as N  # noqa: E402


def test_gh():
    for d, m in ((2, 5), (4, 5), (4, 7)):
        eps, w = N.gh_eps(m, d)
        assert abs(w.sum() - 1) < 1e-12 and (w > 0).all()
        assert np.allclose((w[:, None] * eps).sum(0), 0, atol=1e-12)               # first moment
        cov = (w[:, None, None] * eps[:, :, None] * eps[:, None, :]).sum(0)
        assert np.allclose(cov, np.eye(d), atol=1e-12)                              # second moment
        assert abs((w * eps[:, 0] ** 4).sum() - 3) < 1e-10                          # fourth moment of N(0,1) = 3 (exact for m >= 3)
    print("PASS Gauss-Hermite: weights sum to 1, moments 1/2/4 exact")


def test_mc_qmc():
    e1 = N.mc_eps(4096, 4, 0); e2 = N.mc_eps(8192, 4, 0)
    assert np.array_equal(e1, e2[:4096]), "MC prefixes must be nested"
    q, bad = N.rqmc_eps(8192, 4, 3); q2, _ = N.rqmc_eps(1024, 4, 3)
    assert np.allclose(q[:1024], q2) and bad == 0, "RQMC prefixes must be nested"
    assert np.abs(q.mean(0)).max() < 0.02 and np.abs(np.cov(q.T) - np.eye(4)).max() < 0.03
    print("PASS MC / RQMC nested prefixes; RQMC mean/cov close to N(0,I) at 8192")


def test_measures():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 4)); S = A @ A.T + np.eye(4); L = np.linalg.cholesky(S); mu = rng.standard_normal(4)
    eps = N.mc_eps(200000, 4, 5)
    V1 = N.measure_V("P1", eps, S, L, mu)
    assert np.allclose(np.cov(V1.T), S, rtol=0.03, atol=0.05), "P1 covariance"
    V3 = N.measure_V("P3", eps, S, L, mu)
    assert np.allclose(V3.mean(0), -mu, atol=0.02), "P3: V = eps - mu_train"
    V4 = N.measure_V("P4", eps, S, L, mu)
    assert np.allclose(np.cov(V4.T), np.trace(S) / 4 * np.eye(4), rtol=0.03, atol=0.05)
    w = N.truncation_weights(eps, 0.95)
    assert abs(w.mean() - 0.95) < 0.003
    print("PASS measures: P1 covariance, P3 mean map, P4 spherical, P5 truncation rate")


if __name__ == "__main__":
    test_gh(); test_mc_qmc(); test_measures()
