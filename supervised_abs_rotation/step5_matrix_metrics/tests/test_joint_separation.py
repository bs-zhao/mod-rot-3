import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joint_separation as js
import coefficient_support as cs
import surrogates as sur


def test_empirical_basis_allows_binary_support_without_false_extra_degrees():
    x = np.array([-1., 1.]); features = x[:, None]**np.arange(5)
    g = features.T @ features/2
    moment = js.marginal_basis(g)
    assert np.count_nonzero(np.diag(moment)) == 2
    np.testing.assert_allclose(moment[:, 2:], 0., atol=1e-12)


def test_mixed_block_spectrum_distinguishes_rotated_bilinear_coupling():
    rng = np.random.default_rng(411); w = rng.normal(size=(3000, 4))
    powers = sur.powers_for(4, 2); weights = sur.weights_for(powers)
    phi = sur.monomials(w, powers, weights)
    model = js.JointSeparation(powers, weights, 2, phi.T @ phi, len(w))
    structure = cs.Structure(powers, 2)
    coef = np.zeros((len(powers), 1))
    for p in ([1, 0, 1, 0], [0, 1, 0, 1]):
        i = np.flatnonzero(np.all(powers == p, axis=1))[0]; coef[i] = 1/weights[i]
    q = np.array([[1., -1.], [1., 1.]])/np.sqrt(2)
    moment = model.moment_map(1.)
    a, _ = model.scores(moment.T @ (weights*coef[:, 0]))
    b, _ = model.scores(moment.T @ (weights*structure.rotate(coef, q)[:, 0]))
    assert a['UX_block_full_mean_effective_support_p1'] < b['UX_block_full_mean_effective_support_p1']
    assert b['control_product_reference_norm2'] == pytest.approx(a['control_product_reference_norm2'])


def test_gaussian_product_norm_is_rotation_invariant_and_scale_free():
    rng = np.random.default_rng(412); w = rng.normal(size=(5000, 5))
    powers = sur.powers_for(5, 3); weights = sur.weights_for(powers)
    phi = sur.monomials(w, powers, weights)
    model = js.JointSeparation(powers, weights, 3, phi.T @ phi, len(w))
    structure = cs.Structure(powers, 3); coef = rng.normal(size=(len(powers), 1))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    for sigma in (.5, 1., 2.):
        moment = model.moment_map(sigma)
        a, _ = model.scores(moment.T @ (weights*coef[:, 0]))
        b, _ = model.scores(moment.T @ (weights*structure.rotate(coef, q)[:, 0]))
        assert b['control_product_reference_norm2'] == pytest.approx(a['control_product_reference_norm2'])
        scaled, _ = model.scores(moment.T @ (weights*coef[:, 0])*100.)
        for key in a:
            if not key.startswith('control_'):
                assert scaled[key] == pytest.approx(a[key], abs=1e-9)
