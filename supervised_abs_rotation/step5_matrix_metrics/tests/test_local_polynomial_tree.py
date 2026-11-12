import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import local_polynomial_tree as model
import surrogates as sur


def test_complete_local_polynomial_predictions_are_rotation_equivariant():
    rng = np.random.default_rng(661); u = rng.normal(size=(200, 4)); target = rng.normal(size=(200, 8))
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    powers = sur.powers_for(4, 2); weights = sur.weights_for(powers)
    with threadpool_limits(2):
        a, b, _ = model.local_fit(u[:140], target[:140], u[140:], powers, weights)
        c, d, _ = model.local_fit(u[:140] @ q, target[:140], u[140:] @ q, powers, weights)
    np.testing.assert_allclose(a, c, atol=1e-12)
    np.testing.assert_allclose(b, d, atol=1e-12)


def test_global_quadratic_is_exact_and_needs_no_partition():
    rng = np.random.default_rng(662); u = rng.normal(size=(240, 3))
    y = np.column_stack((u[:, 0]*u[:, 1]+u[:, 2]**2, 3*u[:, 1]-.4*u[:, 0]**2))
    train, query = np.arange(180), np.arange(180, 240)
    with threadpool_limits(2):
        leaves, errors, audit = model.fit_path(u, y, train, query, degree=2)
    assert errors.max() < 1e-20
    assert audit['minimum_node_design_rank'] == 10
    assert audit['minimum_leaf_rows'] >= 20


def test_local_quadratics_reduce_heldout_bias_relative_to_constant_tree():
    from sklearn.tree import DecisionTreeRegressor
    rng = np.random.default_rng(663); u = rng.uniform(-2., 2., size=(500, 3))
    y = (np.sin(u[:, 0])+u[:, 1]*u[:, 2]+.1*u[:, 0]**3)[:, None]
    train, query = np.arange(375), np.arange(375, 500)
    with threadpool_limits(2):
        _, errors, audit = model.fit_path(u, y, train, query, degree=2)
    other = DecisionTreeRegressor(random_state=424).fit(u[train], y[train])
    mse = np.sum((other.predict(u[query])[:, None]-y[query])**2)/np.sum(y[query]**2)
    assert errors[1, -1] < mse*.2
    assert audit['train_subjects'] == 375 and np.all(np.diff(errors[0]) <= 1e-10)
