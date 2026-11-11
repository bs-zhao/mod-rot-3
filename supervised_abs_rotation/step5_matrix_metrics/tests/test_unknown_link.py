import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unknown_link as ul


def test_monotone_link_cannot_collapse_and_handles_ties():
    y = np.repeat(np.linspace(-2, 2, 20), 3)[:, None]
    train = np.arange(len(y))
    transformed, links = ul.calibrate(y, np.zeros_like(y), ul.orders(y, train))
    assert transformed.std() == pytest.approx(1.)
    assert np.all(np.diff(links[0][1]) > 0)
    np.testing.assert_allclose(ul.inverse_link(transformed, links), y)


def test_unknown_link_improves_nonlinear_response_without_formula():
    rng = np.random.default_rng(50)
    u = rng.uniform(-1, 1, size=(1000, 3))
    y = np.exp(1.5*u[:, 0]+u[:, 1]**3)[:, None]
    split = (np.arange(600), np.arange(600, 800), np.arange(800, 1000))
    scores, fit = ul.fit_case(u, y, split, "poly3", iterations=12)
    assert fit["best_iteration"] > 0
    assert scores["unknown_link_test_nmse"] < scores["identity_link_test_nmse"]*.5


def test_unknown_link_signed_permutation_invariance():
    rng = np.random.default_rng(51)
    u = rng.normal(size=(500, 3))
    y = np.tanh(u[:, 0] + .3*u[:, 1]**2)[:, None]
    split = (np.arange(300), np.arange(300, 400), np.arange(400, 500))
    a, _ = ul.fit_case(u, y, split, "rbf6", iterations=4)
    b, _ = ul.fit_case(u[:, [2, 0, 1]]*[-1, 1, -1], y, split, "rbf6", iterations=4)
    for key in a:
        assert b[key] == pytest.approx(a[key], rel=1e-7, abs=1e-7)


def test_cached_link_interpolation_preserves_heldout_predictions_and_ties():
    rng = np.random.default_rng(52)
    u = rng.normal(size=(300, 3))
    y = np.round(np.column_stack([np.tanh(u[:, 0]), np.exp(.3*u[:, 1])]), 2)
    design = np.column_stack([np.ones(300), u, u**2])
    split = (np.arange(180), np.arange(180, 240), np.arange(240, 300))
    cached = ul.chain(design, y, *split, iterations=5, use_cache=True)
    direct = ul.chain(design, y, *split, iterations=5, use_cache=False)
    assert cached["best_iteration"] == direct["best_iteration"]
    np.testing.assert_allclose(cached["test_nmse_by_output"], direct["test_nmse_by_output"], atol=1e-12)
    np.testing.assert_allclose(cached["coef"], direct["coef"], atol=1e-12)


def test_validation_patience_does_not_use_test_error():
    history = [dict(iteration=i, validation_nmse=v, test_nmse=t)
               for i, (v, t) in enumerate(zip([.5, .3, .31, .32, .33, .1], [1., 2., 1., 0., -1., -2.]))]
    stop, selected = ul.validation_checkpoint(history, 3)
    assert stop == 4
    assert selected == 1
    assert ul.validation_checkpoint(history, None) == (5, 5)
