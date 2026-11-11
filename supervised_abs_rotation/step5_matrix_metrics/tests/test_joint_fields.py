import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joint_fields as jf
import local_joint as lj
import surrogates as sur


def test_lowered_coefficient_derivatives_match_original_full_derivative_routine():
    rng = np.random.default_rng(451); z = rng.normal(size=(70, 6))
    scaler = sur.Scaler(z, 4); model = sur.Polynomial(scaler, 4)
    model.set_coef(rng.normal(size=(len(model.powers), 2)))
    g, h = jf.evaluate(scaler(z), model.powers, model.weights, model.coef)
    original_g, original_h = model.derivatives(z)
    np.testing.assert_allclose(g/scaler.scale, original_g, atol=1e-9)
    np.testing.assert_allclose(h/scaler.scale[None, None, :, None]/scaler.scale[None, None, None, :],
                               original_h, atol=1e-9)


def test_local_ux_derivatives_and_controls_are_signed_permutation_invariant():
    rng = np.random.default_rng(452); g = rng.normal(size=(15, 5, 7))
    h = rng.normal(size=(15, 5, 7, 7)); h = (h+h.swapaxes(-1, -2))/2
    q = np.eye(4)[:, [2, 0, 3, 1]]*[-1., 1., 1., -1.]
    a, b = lj.evaluate(g, h, np.eye(4), 4), lj.evaluate(g, h, q, 4)
    for key in a:
        assert b[key] == pytest.approx(a[key], abs=1e-9)
