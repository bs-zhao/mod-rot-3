import itertools
import sys
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tree_function_complexity as trees


def test_every_best_first_distortion_matches_explicit_truncated_tree_predictions():
    rng = np.random.default_rng(651); u = rng.normal(size=(120, 3))
    target = np.column_stack((u[:, 0]**2+.2*u[:, 1], np.sin(u[:, 1])+.1*u[:, 2]))
    train, query = np.arange(90), np.arange(90, 120)
    leaves, errors, audit = trees.fit_path(u, target, train, query)
    other = DecisionTreeRegressor(random_state=421).fit(u[train], target[train])
    for count in (2, 4, 8, 16, 32):
        # Independent pointwise traversal of the same grown tree. Re-fitting
        # sklearn's best-first builder can choose different tied splits.
        expanded = set(audit['split_node_ids'][:count-1])
        for k, rows in enumerate((train, query)):
            predictions = []
            for row in u[rows]:
                node = 0
                while node in expanded:
                    node = (other.tree_.children_left[node] if row[other.tree_.feature[node]] <= other.tree_.threshold[node]
                            else other.tree_.children_right[node])
                predictions.append(other.tree_.value[node, :, 0])
            expected = np.sum((np.array(predictions)-target[rows])**2)/np.sum(target[rows]**2)
            np.testing.assert_allclose(errors[k, count-1], expected, atol=1e-12)
    assert audit['train_subjects'] == 90
    assert errors[0, -1] < 1e-12


def test_small_residuals_need_fewer_rules_in_the_users_explicit_sparsity_example():
    u = np.array(list(itertools.product((-1., 1.), repeat=4)))
    rows = np.arange(len(u)); counts = []
    for coefficients in ([2., 2., .01, .01], [1., .5, .1, .1]):
        y = (u @ coefficients)[:, None]; y /= np.sqrt(np.mean(y*y))
        leaves, errors, _ = trees.fit_path(u, y, rows, rows)
        counts.append(trees.minimum_leaves(leaves, errors[0], .03))
    assert counts == [4, 16]


def test_tree_error_includes_discarded_modes_and_is_invariant_to_output_basis():
    rng = np.random.default_rng(652); u = rng.normal(size=(80, 3)); target = rng.normal(size=(80, 4))
    q, _ = np.linalg.qr(rng.normal(size=(4, 4))); rows = np.arange(len(u)); discarded = np.full(len(u), .001)
    leaves, a, _ = trees.fit_path(u, target, rows, rows, discarded)
    _, b, _ = trees.fit_path(u, target @ q, rows, rows, discarded)
    np.testing.assert_allclose(a, b, atol=1e-12)
    expected = discarded.sum()/(np.sum(target*target)+discarded.sum())
    np.testing.assert_allclose(a[:, -1], expected, atol=1e-12)
    assert np.all(np.diff(a[0]) <= 1e-12)


def test_complete_tree_pipeline_retains_all_original_matrix_rows(tmp_path):
    import json
    from threadpoolctl import threadpool_limits
    import matrix_function_profiles as profiles
    import run_tree_function_complexity as runner
    from test_matrix_metrics import make_cases
    source = tmp_path/'source'; make_cases(source)
    folder, output = tmp_path/'bank', tmp_path/'out'
    with threadpool_limits(limits=2):
        profiles.prepare(source, folder)
        rows = runner.analyze(source, output, folder)
    assert rows and max(r['endpoint_error'] for r in rows) < 1e-12
    protocol = json.loads((output/'protocol.json').read_text())
    assert protocol['status'] == 'COMPLETE' and protocol['all_original_rows'] == 160
    audit = json.loads((output/'read_audit.json').read_text())
    assert audit['forbidden_files_read'] == [] and audit['all_subjects_in_tree_fit'] == 32


def test_fitted_polynomial_tree_pipeline_uses_the_same_matrix_reference(tmp_path):
    import json
    from threadpoolctl import threadpool_limits
    import matrix_function_profiles as profiles
    import prepare_response_bank
    import run_tree_function_complexity as runner
    from test_matrix_metrics import make_cases
    source = tmp_path/'source'; make_cases(source)
    matrix, fitted, output = tmp_path/'matrix', tmp_path/'fitted', tmp_path/'out'
    with threadpool_limits(limits=2):
        profiles.prepare(source, matrix)
        prepare_response_bank.prepare(source, fitted, estimators=('poly5',), centers=10)
        rows = runner.analyze(source, output, matrix, source_estimator='poly5', fitted_folder=fitted)
    assert rows and all(r['estimator'] == 'poly5_conditional_tree' for r in rows)
    protocol = json.loads((output/'protocol.json').read_text())
    assert protocol['direct_raw_response_table'] is False and protocol['fitted_source_manifest_sha256']
    assert max(r['endpoint_error'] for r in rows) < 1e-12
