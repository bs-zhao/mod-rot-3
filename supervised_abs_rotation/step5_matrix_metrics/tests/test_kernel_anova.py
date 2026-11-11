import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kernel_anova as ka
import surrogates as sur


def test_analytic_kernel_moments_match_monte_carlo():
    rng = np.random.default_rng(70)
    centers = rng.normal(size=(5, 2)); anchor = np.array([.3, -.2])
    m, c, second = ka.moments(centers, anchor, 1.2, .4)
    z = rng.normal(anchor[0], .4, size=200000)
    phi = np.exp(-(z[:, None]-centers[None, :, 0])**2/(2*1.2**2))
    np.testing.assert_allclose(phi.T @ phi/len(z), second[0], atol=.002, rtol=.004)
    np.testing.assert_allclose(m[0]+c[0], second[0], atol=1e-14)


def test_component_variances_sum_to_function_variance():
    rng = np.random.default_rng(71)
    centers = rng.normal(size=(8, 3)); beta = rng.normal(size=(8, 2))
    _, values, total, audit = ka.variances(centers, beta @ beta.T, np.zeros(3), 1.1, .5)
    assert np.all(values >= 0)
    assert values.sum() == pytest.approx(total, rel=1e-12)
    assert audit["negative_relative_energy"] < 1e-12


def test_signed_permutation_and_total_variance_rotation_controls():
    rng = np.random.default_rng(72)
    centers = rng.normal(size=(10, 3)); anchors = rng.normal(size=(3, 3))
    beta = rng.normal(size=(10, 4)); gram = beta @ beta.T
    a, _ = ka.scores(centers, gram, anchors, 1.2, .5)
    p = np.eye(3)[:, [2, 0, 1]]*[-1, 1, -1]
    b, _ = ka.scores(centers @ p, gram, anchors @ p, 1.2, .5)
    for key in a:
        assert b[key] == pytest.approx(a[key], abs=1e-12)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rot, _ = ka.scores(centers @ q, gram, anchors @ q, 1.2, .5)
    assert rot["control_total_local_variance"] == pytest.approx(a["control_total_local_variance"], abs=1e-12)


def test_polynomial_gaussian_variance_matches_closed_form():
    powers = sur.powers_for(2, 2); weights = sur.weights_for(powers)
    coef = np.zeros((len(powers), 1))
    coef[np.flatnonzero(np.all(powers == [2, 0], axis=1))[0], 0] = 1.
    anchor = np.array([[.7, .2]])
    scores, _ = ka.polynomial_scores(powers, weights, coef @ coef.T, anchor, .4)
    assert scores["control_total_local_variance"] == pytest.approx(4*.7**2*.4**2+2*.4**4)
    assert scores["interaction_fraction"] == pytest.approx(0.)


def test_polynomial_rotation_preserves_local_variance():
    rng = np.random.default_rng(73)
    powers = sur.powers_for(3, 5); weights = sur.weights_for(powers)
    coef = rng.normal(size=(len(powers), 3))
    gram = coef @ coef.T
    anchors = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    t = ka.rotation_representation(powers, weights, q, rng.normal(size=(4*len(powers), 3)))
    a, _ = ka.polynomial_scores(powers, weights, gram, anchors, .3)
    b, _ = ka.polynomial_scores(powers, weights, t @ gram @ t.T, anchors @ q, .3)
    assert b["control_total_local_variance"] == pytest.approx(a["control_total_local_variance"], rel=1e-10)
