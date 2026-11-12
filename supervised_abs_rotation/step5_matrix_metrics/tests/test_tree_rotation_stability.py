import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tree_rotation_stability as stability
import tree_function_complexity as trees


def test_fast_training_count_matches_explicit_prediction_path_with_output_tail():
    rng = np.random.default_rng(671); u = rng.normal(size=(200, 4)); target = rng.normal(size=(200, 9))
    discarded = rng.uniform(0., 1e-5, size=len(u)); rows = np.arange(len(u))
    with threadpool_limits(2):
        leaves, errors, _ = trees.fit_path(u, target, rows, rows, discarded)
        for eps in (.01, .03, .1):
            assert stability.training_count(u, target, discarded, eps) == trees.minimum_leaves(leaves, errors[0], eps)


def test_subject_resampling_recenters_retained_and_discarded_function_modes():
    rng = np.random.default_rng(672); response = rng.normal(size=(50, 2, 12))
    with threadpool_limits(2):
        targets, residual = stability.coupling_modes(response, [np.eye(12)], tolerance=.1)
    ids = rng.integers(50, size=50)
    values, discarded = stability.sample_functions(targets, residual, ids)
    np.testing.assert_allclose(values[0].mean(0), 0., atol=1e-15)
    centered_tail = residual[0][ids]-residual[0][ids].mean(0)
    np.testing.assert_allclose(discarded[0], np.sum(centered_tail**2, axis=1))
    np.testing.assert_allclose(targets[0].T @ residual[0], 0., atol=1e-12)


def test_random_and_exponential_rotations_stay_in_SO_without_case_labels():
    rng = np.random.default_rng(673)
    for d in (4, 6):
        q = stability.random_rotation(d, rng) @ stability.increment(rng.normal(size=d*(d-1)//2), d)
        np.testing.assert_allclose(q.T @ q, np.eye(d), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(q), 1., atol=1e-12)


def test_complete_blind_and_bootstrap_pipeline_uses_frozen_matrix_only_candidate(tmp_path):
    import matrix_function_profiles as profiles
    import run_tree_function_complexity as tree_runner
    import run_tree_rotation_stability as runner
    from test_matrix_metrics import make_cases
    source = tmp_path/'source'; make_cases(source)
    parent = tmp_path/'step5_matrix_metrics'; parent.mkdir()
    folder = parent/'matrix_response_table_seed383'; original = parent/'original'; output = parent/'stability'
    with threadpool_limits(2):
        profiles.prepare(source, folder)
        tree_runner.analyze(source, original, folder)
        result = runner.analyze(source, original, output, random_rotations=2, bootstrap=2, optimizer_starts=0)
    assert result['bootstrap_replicates'] == 2
    assert json.loads((output/'protocol.json').read_text())['status'] == 'COMPLETE'
    audit = json.loads((output/'read_audit.json').read_text())
    assert audit['forbidden_files_read'] == [] and audit['all_subjects'] == 32
    assert json.loads((output/'reproduction.json').read_text())['all_counts_identical'] is True
    assert (output/'blind_freeze.json').exists()
