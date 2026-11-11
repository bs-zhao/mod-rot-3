import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import profiled_links as pl


@pytest.mark.parametrize("method", ["l2", "l1"])
def test_local_link_profiling_removes_smooth_output_reparameterization(method):
    rng = np.random.default_rng(60)
    g = rng.normal(size=(200, 4))
    h = rng.normal(size=(200, 4, 4)); h = (h+h.swapaxes(-1, -2))/2
    a = rng.uniform(.3, 3, size=(200,))
    b = rng.normal(size=(200,))
    gt = a[:, None]*g
    ht = a[:, None, None]*h+b[:, None, None]*g[:, :, None]*g[:, None, :]
    r, _, _ = pl.residual(g, h, method)
    rt, _, _ = pl.residual(gt, ht, method)
    np.testing.assert_allclose(r, rt, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("method", ["l2", "l1"])
def test_profiled_signed_permutation_preserves_scores(method):
    rng = np.random.default_rng(61)
    g = rng.normal(size=(200, 4))
    h = rng.normal(size=(200, 4, 4)); h = (h+h.swapaxes(-1, -2))/2
    q = np.eye(4)[:, [2, 0, 3, 1]]*[-1, 1, -1, 1]
    a, _ = pl.scores(g, h, method)
    b, _ = pl.scores(g @ q, np.einsum("ij,njk,kl->nil", q.T, h, q), method)
    for key in a:
        assert b[key] == pytest.approx(a[key], abs=1e-10)


def test_additive_function_under_unknown_link_is_diagonal_after_profiling():
    rng = np.random.default_rng(62)
    g = rng.normal(size=(200, 4))
    diag = rng.uniform(.3, 2, size=(200, 4))
    h = np.zeros((200, 4, 4)); h[:, np.arange(4), np.arange(4)] = diag
    h += rng.normal(size=(200, 1, 1))*g[:, :, None]*g[:, None, :]
    base, _ = pl.scores(g, h, "l2")
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    rotated, _ = pl.scores(g @ q, np.einsum("ij,njk,kl->nil", q.T, h, q), "l2")
    assert base["profiled_edge_fraction"] < 1e-20
    assert rotated["profiled_edge_fraction"] > .01
