import sys
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import conditional_ux as cu
import ux_predictive_complexity as uc
import prepare_response_bank as prep
import run_conditional_ux as runner
from test_matrix_metrics import make_cases


def test_balanced_latent_compression_preserves_each_omitted_X_task():
    rng = np.random.default_rng(551)
    phi = rng.normal(size=(150, 12)); coef = rng.normal(size=(12, 2, 40))
    # Deliberately different X-column and channel strengths.
    maps = {'test': [np.eye(40), np.eye(40)*1e-4]}
    z, weights, grams, audit = cu.balanced_latents(phi, coef, maps)
    residual = rng.normal(size=(80, len(z)))
    h = residual.T @ residual/len(residual)
    assert weights.sum() == pytest.approx(1.)
    assert audit['maximum_task_relative_squared_reconstruction_error'] < 1e-10
    expected = np.einsum('cjaa->cj', grams['test'])
    centered = (phi-phi.mean(0)) @ (coef-coef.mean(2, keepdims=True)).reshape(12, -1)
    direct = np.mean(centered.reshape(150, 2, 40)**2, axis=(0, 2))
    np.testing.assert_allclose(expected[:, 0], direct, rtol=1e-10)
    np.testing.assert_allclose(expected[:, 1], direct*1e-8, rtol=1e-10)


def test_conditional_omission_removes_affine_complement_without_X_independence():
    rng = np.random.default_rng(552); z = rng.normal(size=(240, 3))
    z[:, 1] += .6*z[:, 0]
    direction = np.array([1., 0., 0.])
    target = np.column_stack((2+z[:, 1]-3*z[:, 2], z[:, 0]))
    with threadpool_limits(limits=2):
        covariance, audit = cu.omission_covariance(z, direction, target, np.array([.5, .5]),
            degree=3, centers=24, folds=4)
    assert covariance[0, 0] < 1e-20
    assert covariance[1, 1] > .3
    assert audit['all_subjects_scored'] == 240
    assert sum(r['scored_subjects'] for r in audit['history']) == 240


def test_UX_column_support_ignores_arbitrary_column_scales_and_obeys_user_example():
    a = np.tile(np.array([2., 2., .01, .01])[:, None]**2, (1, 6))
    b = np.tile(np.array([1., .5, .1, .1])[:, None]**2, (1, 6))
    sa, sb = cu.sparsity(a), cu.sparsity(b)
    changed = cu.sparsity(a*np.array([1e-10, 1e10, 2., 3., 7., 8.]))
    for key in sa:
        if key.startswith('UX_columns') or key.startswith('equal_X'):
            assert changed[key] == pytest.approx(sa[key])
            assert sa[key] < sb[key]


def test_W_only_direction_cache_identifies_signed_permutations():
    q = np.array([[1., -1.], [1., 1.]])/2**.5
    directions, membership = cu.direction_classes([np.eye(2), q, -q[:, ::-1]])
    assert len(directions) == 4
    np.testing.assert_array_equal(membership[1], membership[2, ::-1])


def test_complete_conditional_UX_pipeline_only_reads_WY(tmp_path):
    import json
    source = tmp_path/'source'; make_cases(source)
    folder, output = tmp_path/'bank', tmp_path/'out'
    with threadpool_limits(limits=2):
        prep.prepare(source, folder, estimators=('poly3',), centers=10)
        rows = runner.analyze(source, output, folder, capacities=((2, 8),), folds=2)
    assert rows
    assert json.loads((output/'protocol.json').read_text())['status'] == 'COMPLETE'
    audit = json.loads((output/'read_audit.json').read_text())
    assert audit['all_subjects_in_fit_and_crossfit_scoring'] == 32
    assert audit['original_rows_in_source_fit'] == 160
    assert audit['forbidden_files_read'] == []
