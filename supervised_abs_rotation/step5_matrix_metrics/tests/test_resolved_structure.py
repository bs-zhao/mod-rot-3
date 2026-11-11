import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import resolved_structure as rs
import coefficient_support as cs
import joint_separation as js
import surrogates as sur


def test_degree_support_is_scale_and_signed_permutation_invariant():
    rng = np.random.default_rng(421)
    powers = sur.powers_for(6, 4); structure = cs.Structure(powers, 4)
    coef = rng.normal(size=(len(powers), 1))
    q = np.eye(4)[:, [3, 1, 0, 2]]*[-1., 1., -1., 1.]
    a, _ = rs.degree_support(powers, coef[:, 0], 4)
    b, _ = rs.degree_support(powers, structure.rotate(coef, q)[:, 0]*20., 4)
    for key in a:
        assert a[key] == pytest.approx(b[key], abs=1e-9)


def test_nuisance_terms_do_not_change_pure_ux_scores():
    rng = np.random.default_rng(422); powers = sur.powers_for(4, 3); weights = sur.weights_for(powers)
    w = rng.normal(size=(400, 4)); phi = sur.monomials(w, powers, weights)
    model = js.JointSeparation(powers, weights, 2, phi.T @ phi, len(w))
    coef = rng.normal(size=len(powers)); changed = coef.copy()
    only_u_or_x = (powers[:, :2].sum(1) == 0) | (powers[:, 2:].sum(1) == 0)
    changed[only_u_or_x] += rng.normal(size=int(only_u_or_x.sum()))*1000.
    a, _ = rs.scores(model, coef); b, _ = rs.scores(model, changed)
    for key in a:
        if key.startswith('UX_coupling_'):
            assert a[key] == pytest.approx(b[key], abs=1e-10)
