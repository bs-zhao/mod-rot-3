import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import finite_effects as fe


def test_fixed_stimulus_signs_are_allowed_to_differ():
    e = np.ones((100, 4, 3)); e[:, [0, 2]] *= -1
    assert fe.direction_loss(e) == 0.
    e[:50] *= -1
    assert fe.direction_loss(e) == pytest.approx(1.)
    assert fe.direction_loss(e, np.repeat([0, 1], 50)) == 0.


def test_sign_permutation_and_context_invariance():
    rng = np.random.default_rng(31)
    u = rng.normal(size=(200, 3)); e = rng.normal(size=(200, 12, 3))
    a = fe.effect_scores(e, fe.contexts(u, (8,)))
    p = [2, 0, 1]; signs = [-1, 1, -1]
    b = fe.effect_scores(e[:, :, p]*signs, fe.contexts(u[:, p]*signs, (8,)))
    for key in a:
        assert b[key] == pytest.approx(a[key], abs=1e-12)


def test_support_intersection_ignores_sign_permutation():
    rng = np.random.default_rng(32)
    u = rng.normal(size=(300, 3))
    p = np.eye(3)[:, [2, 0, 1]]*[-1, 1, -1]
    a = fe.support_mask(u, [np.eye(3)])
    b = fe.support_mask(u, [p])
    np.testing.assert_array_equal(a, b)
    assert 0 < a.mean() <= 1


def test_stimulus_contrast_tail_distinguishes_one_and_two_axes():
    rng = np.random.default_rng(33)
    x = rng.normal(size=(20, 2))
    g = np.zeros((40, 20, 3)); g[:, :, 0] = x[:, 0]
    assert fe.ux_contrast_scores(g, x)["UX_contrast_tail1_l2"] == 0.
    g[:, :, 1] = x[:, 0]
    assert fe.ux_contrast_scores(g, x)["UX_contrast_tail1_l2"] == pytest.approx(np.sqrt(.5))
