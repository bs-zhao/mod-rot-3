import sys
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import conditional_ux as conditional
import subset_ux as subset
import run_subset_ux as runner
import prepare_response_bank as prep
from test_matrix_metrics import make_cases


def test_W_only_subspace_cache_is_signed_permutation_invariant():
    rng = np.random.default_rng(571); q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    p = np.eye(4)[:, [2, 0, 3, 1]]*[-1, 1, 1, -1]
    bases, members = subset.subspace_classes([q, q @ p])
    assert len(bases) == 15
    assert {r['class_id'] for r in members[0]} == {r['class_id'] for r in members[1]}


def test_coordinate_subset_budget_detects_UX_sparsity_without_coefficient_amplitudes():
    rng = np.random.default_rng(572); z = rng.normal(size=(180, 3))
    target = z[:, :2]-z[:, :2].mean(0); target /= target.std(0)
    q = np.eye(3); q[0, 0]=q[2, 2]=2**-.5; q[0, 2]=-2**-.5; q[2, 0]=2**-.5
    bases, membership = subset.subspace_classes([np.eye(3), q])
    errors = []
    with threadpool_limits(limits=2):
        for basis in bases:
            covariance, _ = conditional.omission_covariance(z, None, target, np.array([.5, .5]),
                basis=basis, degree=2, centers=12, folds=3)
            errors.append(np.diag(covariance)[None])
    curves = subset.budget_curves(np.asarray(errors), membership, 3)
    assert subset.scores(curves[0, :, 0])['UX_minimum_U_subset_tail1_mean_l2'] < 1e-7
    assert subset.scores(curves[1, :, 0])['UX_minimum_U_subset_tail1_mean_l2'] > .1
    np.testing.assert_allclose(curves[:, -1], 0., atol=1e-20)
    assert np.all(np.diff(curves, axis=1) <= 0.)


def test_subspace_predictor_is_invariant_to_orthonormal_basis_rotation():
    rng = np.random.default_rng(573); z = rng.normal(size=(150, 4)); y = rng.normal(size=(150, 3))
    basis, _ = np.linalg.qr(rng.normal(size=(4, 2))); rotate, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    with threadpool_limits(limits=2):
        a, _ = conditional.omission_covariance(z, None, y, np.ones(3)/3, basis=basis, degree=3, centers=20, folds=3)
        b, _ = conditional.omission_covariance(z, None, y, np.ones(3)/3, basis=basis @ rotate, degree=3, centers=20, folds=3)
    np.testing.assert_allclose(a, b, atol=1e-7, rtol=1e-7)


def test_full_UX_subset_pipeline_scores_all_subjects_and_checks_full_model_error(tmp_path):
    import json
    source = tmp_path/'source'; make_cases(source)
    folder, output = tmp_path/'bank', tmp_path/'out'
    with threadpool_limits(limits=2):
        prep.prepare(source, folder, estimators=('poly3',), centers=10)
        rows = runner.analyze(source, output, folder, estimators=('poly3',), degree=2, centers=8, folds=2)
    assert rows and all('full_U_relative_mse' in r for r in rows)
    audit = json.loads((output/'read_audit.json').read_text())
    assert audit['all_subjects'] == 32
    assert audit['all_original_rows_in_source_fit'] == 160
    assert audit['forbidden_files_read'] == []
    assert json.loads((output/'protocol.json').read_text())['status'] == 'COMPLETE'
