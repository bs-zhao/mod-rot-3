import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import predictive_compression as pc
import surrogates as sur


def compress_matrix(x, coef):
    y = x @ coef; g = x.T @ x; ct = x.T @ y; ss = float(y @ y)
    return pc.compress(g, g, ct, ct, ss, ss, max_terms=16)


def test_user_sparsity_example_uses_relative_prediction_tail():
    x = np.eye(4)
    a, _ = compress_matrix(x, np.array([2., 2., .01, .01]))
    b, _ = compress_matrix(x, np.array([1., .5, .1, .1]))
    assert a['prediction_tail2_l2'] == pytest.approx(.0049999375)
    assert b['prediction_tail2_l2'] == pytest.approx(np.sqrt(.02/1.27))
    assert a['prediction_count_eps010'] < b['prediction_count_eps010']


def test_aliases_and_feature_scales_do_not_inflate_support():
    rng = np.random.default_rng(402)
    x = rng.normal(size=(500, 4)); beta = np.array([2., .7, .03, .01])
    expected, _ = compress_matrix(x, beta)
    changed, _ = compress_matrix(x*np.array([.01, 1., 30., 2.]), beta/np.array([.01, 1., 30., 2.]))
    alias, _ = compress_matrix(np.c_[x, x[:, 0]], np.r_[0., beta[1:], beta[0]])
    for key in expected:
        assert changed[key] == pytest.approx(expected[key], abs=1e-7)
        assert alias[key] == pytest.approx(expected[key], abs=1e-7)


def test_nuisance_and_gram_transport_match_direct_rotated_rows():
    rng = np.random.default_rng(403); d = 3
    w = rng.normal(size=(300, 5)); powers = sur.powers_for(5, 3); weights = sur.weights_for(powers)
    x = sur.monomials(w, powers, weights); coef = rng.normal(size=(len(powers), 2))
    g, v = x[:200].T @ x[:200], x[200:].T @ x[200:]
    transport = pc.Transport(powers, d); system = pc.nuisance_system(g, v, coef, transport.structure.has_u)
    q, _ = np.linalg.qr(rng.normal(size=(d, d))); wr = w.copy(); wr[:, :d] = w[:, :d] @ q
    xr = sur.monomials(wr, powers, weights)
    cr = transport.structure.rotate(coef, q)
    actual = pc.nuisance_system(xr[:200].T @ xr[:200], xr[200:].T @ xr[200:], cr, transport.structure.has_u)
    blocks = transport.blocks(q)
    for key in ('c', 'v'):
        np.testing.assert_allclose(transport.square(system[key], blocks), actual[key], atol=1e-8)
    for key in ('ct', 'cv'):
        np.testing.assert_allclose(transport.vector(system[key], blocks), actual[key], atol=1e-8)
    np.testing.assert_allclose(system['ss_validation'], actual['ss_validation'], atol=1e-8)


def test_each_selected_prefix_matches_direct_train_least_squares_on_heldout():
    rng = np.random.default_rng(404)
    x = rng.normal(size=(250, 12)); x[:, 1] += .95*x[:, 0]
    beta = rng.normal(size=12); y = x @ beta
    xt, xv = x[:170], x[170:]; yt, yv = y[:170], y[170:]
    _, audit = pc.compress(xt.T @ xt, xv.T @ xv, xt.T @ yt, xv.T @ yv,
                           yt @ yt, yv @ yv, max_terms=9)
    for k, error in enumerate(audit['raw_relative_mse'][1:], 1):
        ids = audit['selected'][:k]
        coefficient = np.linalg.lstsq(xt[:, ids], yt, rcond=None)[0]
        expected = np.sum((yv-xv[:, ids] @ coefficient)**2)/np.sum(yv**2)
        assert error == pytest.approx(expected, abs=1e-10)
