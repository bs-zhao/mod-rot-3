import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ux_objective as ux
import local_joint as lj


@pytest.mark.parametrize('kind,eps,label', [('fractional', .1, 'fractional_count_eps010'),
                                          ('soft', .03, 'soft_count_eps003')])
@pytest.mark.parametrize('aggregation', ['mean', 'median'])
def test_covariance_objective_matches_full_local_field_rule_and_gradient(kind, eps, label, aggregation):
    rng = np.random.default_rng(461); d = 4
    g = rng.normal(size=(53, 8, 7)); h = rng.normal(size=(53, 8, 7, 7)); h = (h+h.swapaxes(-1, -2))/2
    objective = ux.UXObjective(h[..., :d, d:], eps, aggregation, kind)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    expected = lj.evaluate(g, h, q, d)[f'UX_columns_subject_{aggregation}_{label}']
    actual, _ = objective.values_gradient(q)
    assert actual == pytest.approx(expected, abs=1e-10)
    theta = rng.normal(size=6)*.05
    loss, gradient = ux.parameter_objective(theta, q, objective)
    for i in range(6):
        perturb = np.eye(6)[i]*1e-6
        numerical = (ux.parameter_objective(theta+perturb, q, objective)[0] -
                     ux.parameter_objective(theta-perturb, q, objective)[0])/2e-6
        assert gradient[i] == pytest.approx(numerical, abs=1e-6)
