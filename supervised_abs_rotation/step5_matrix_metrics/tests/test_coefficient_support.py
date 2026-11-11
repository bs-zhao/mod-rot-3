import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coefficient_support as cs
import surrogates as sur


def test_joint_polynomial_coordinate_change_is_exact():
    rng = np.random.default_rng(110)
    powers = sur.powers_for(7, 4); weights = sur.weights_for(powers)
    structure = cs.Structure(powers, 4)
    coef = rng.normal(size=(len(powers), 2))
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    rotated_coef = structure.rotate(coef, q)
    w = rng.normal(size=(100, 7)); wr = w.copy(); wr[:, :4] = w[:, :4] @ q
    np.testing.assert_allclose(sur.monomials(w, powers, weights) @ coef,
                               sur.monomials(wr, powers, weights) @ rotated_coef, atol=1e-9)
    np.testing.assert_allclose(np.sum(coef**2, axis=0), np.sum(rotated_coef**2, axis=0), atol=1e-8)


def test_joint_support_identifies_sparse_bilinear_structure():
    powers = sur.powers_for(4, 2); weights = sur.weights_for(powers)
    structure = cs.Structure(powers, 2)
    coef = np.zeros((len(powers), 1))
    for p in ([1, 0, 1, 0], [0, 1, 0, 1]):
        i = np.flatnonzero(np.all(powers == p, axis=1))[0]; coef[i] = 1/weights[i]
    q = np.array([[1., -1.], [1., 1.]])/np.sqrt(2)
    a = structure.scores(coef[:, 0]); b = structure.scores(structure.rotate(coef, q)[:, 0])
    assert a["UX_terms_effective_support_p1"] == pytest.approx(2.)
    assert b["UX_terms_effective_support_p1"] == pytest.approx(4.)


def test_sign_permutation_and_overall_output_scale_cannot_change_support():
    rng = np.random.default_rng(111)
    powers = sur.powers_for(6, 3); structure = cs.Structure(powers, 4)
    coef = rng.normal(size=(len(powers), 1))
    p = np.eye(4)[:, [2, 0, 3, 1]]*[-1, 1, -1, 1]
    a = structure.scores(coef[:, 0]); b = structure.scores(structure.rotate(coef, p)[:, 0]*100.)
    for key in a:
        if not key.startswith("control_"):
            assert b[key] == pytest.approx(a[key], abs=1e-10)
