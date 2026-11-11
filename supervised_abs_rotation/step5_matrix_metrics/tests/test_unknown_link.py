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
