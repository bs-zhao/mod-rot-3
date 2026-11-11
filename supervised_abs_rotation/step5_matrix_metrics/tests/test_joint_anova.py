import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joint_anova as ja
import coefficient_support as cs
import joint_separation as js
import surrogates as sur


def test_joint_component_variance_and_bilinear_ux_support():
    import json
    rng = np.random.default_rng(501); w = rng.normal(size=(1000, 4)); w = np.r_[w, -w]
    powers = sur.powers_for(4, 2); weights = sur.weights_for(powers); phi = sur.monomials(w, powers, weights)
    model = js.JointSeparation(powers, weights, 2, phi.T @ phi, len(w)); moment = model.moment_map(1.)
    coef = np.zeros((len(powers), 1))
    for p in ([1, 0, 1, 0], [0, 1, 0, 1]):
        i = np.flatnonzero(np.all(powers == p, axis=1))[0]; coef[i] = 1/weights[i]
    structure = cs.Structure(powers, 2); q = np.array([[1., -1.], [1., 1.]])/np.sqrt(2)
    a, audit = ja.scores(model, moment.T @ (weights*coef[:, 0]))
    b, _ = ja.scores(model, moment.T @ (weights*structure.rotate(coef, q)[:, 0]))
    assert a['UX_components_effective_support_p1'] < b['UX_components_effective_support_p1']
    assert sum(audit['component_energy']) == pytest.approx(a['control_joint_variance'])
    assert a['control_joint_variance'] == pytest.approx(b['control_joint_variance'])
    json.dumps(audit, allow_nan=False)


def test_joint_anova_support_is_signed_permutation_and_scale_invariant():
    rng = np.random.default_rng(502); powers = sur.powers_for(6, 3); weights = sur.weights_for(powers)
    w = rng.normal(size=(1000, 6)); phi = sur.monomials(w, powers, weights)
    model = js.JointSeparation(powers, weights, 4, phi.T @ phi, len(w))
    moment = model.moment_map(.5); coef = rng.normal(size=(len(powers), 1))
    structure = cs.Structure(powers, 4); q = np.eye(4)[:, [2, 0, 3, 1]]*[-1., 1., 1., -1.]
    a, _ = ja.scores(model, moment.T @ (weights*coef[:, 0]))
    b, _ = ja.scores(model, moment.T @ (weights*structure.rotate(coef, q)[:, 0])*20.)
    for key in a:
        if not key.startswith('control_'):
            assert a[key] == pytest.approx(b[key], abs=1e-9)
