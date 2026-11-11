import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kernel_separation as ks
import kernel_anova as ka
import surrogates as sur


def test_dense_multiplicative_interaction_has_low_separation_rank():
    # A product has interactions, but needs one separable term.
    powers = sur.powers_for(3, 5); weights = sur.weights_for(powers)
    coef = np.zeros((len(powers), 1))
    for p, value in (([1, 1, 0], 2.), ([1, 2, 0], 2.), ([1, 1, 2], 1.), ([1, 2, 2], 1.)):
        i = np.flatnonzero(np.all(powers == p, axis=1))[0]
        coef[i] = value/weights[i]
    gram = coef @ coef.T; anchors = np.array([[.2, -.3, .4]])
    base, _ = ks.polynomial_scores(powers, weights, gram, anchors, .5)
    assert base["axis_mean_effective_support_p1"] == pytest.approx(1.)
    rng = np.random.default_rng(90); q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    transform = ka.rotation_representation(powers, weights, q, rng.normal(size=(4*len(powers), 3)))
    rotated, _ = ks.polynomial_scores(powers, weights, transform @ gram @ transform.T, anchors @ q, .5)
    assert rotated["axis_mean_effective_support_p1"] > 1.05


def test_rbf_mode_spectrum_matches_direct_function_quadrature():
    rng = np.random.default_rng(91)
    centers = rng.normal(size=(8, 2)); beta = rng.normal(size=(8, 2))
    anchor = np.array([.2, -.4]); sigma, length = .4, 1.2
    spectra, audit = ks.rbf_spectra(centers, beta @ beta.T, anchor, length, sigma, quadrature=16)
    nodes, weights = np.polynomial.hermite_e.hermegauss(20); weights /= np.sqrt(2*np.pi)
    a, b = np.meshgrid(nodes, nodes, indexing="ij")
    x = anchor+sigma*np.column_stack([a.ravel(), b.ravel()])
    kernel = np.exp(-np.sum((x[:, None]-centers[None])**2, axis=-1)/(2*length**2))
    response = (kernel @ beta).reshape(20, 20, 2)
    response -= np.einsum("i,ijc->jc", weights, response)[None, :, :]
    matrix = response*np.sqrt(weights)[:, None, None]*np.sqrt(weights)[None, :, None]
    direct = np.linalg.svd(matrix.reshape(20, -1), compute_uv=False)**2
    np.testing.assert_allclose(np.sort(spectra[0])[-3:], np.sort(direct)[-3:], rtol=1e-6, atol=1e-10)
    assert audit["quadrature_relative_error"] < 1e-7


def test_rbf_axis_summary_ignores_sign_permutation_and_output_scale():
    rng = np.random.default_rng(92)
    centers = rng.normal(size=(10, 3)); beta = rng.normal(size=(10, 3))
    anchors = rng.normal(size=(2, 3)); p = np.eye(3)[:, [2, 0, 1]]*[-1, 1, -1]
    a, _ = ks.rbf_scores(centers, beta @ beta.T, anchors, 1.3, .3)
    b, _ = ks.rbf_scores(centers @ p, 10000*(beta @ beta.T), anchors @ p, 1.3, .3)
    for key in a:
        assert b[key] == pytest.approx(a[key], rel=1e-6, abs=1e-7)
