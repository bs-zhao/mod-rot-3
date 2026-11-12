import itertools
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import conditional_ui_witness as witness


def test_context_features_retain_multiplicative_U_interactions():
    u = np.array(list(itertools.product((-1., 1.), repeat=3)))
    target = (u[:, 0]*u[:, 1])[:, None]
    first = u[:, 0:1]
    context = first*np.column_stack((np.ones(len(u)), u[:, 1]))
    assert witness.moments(first, target)[0] == 0.
    assert witness.moments(context, target)[0] == 1.


def test_W_direction_cache_matches_antipodes_despite_roundoff_in_nominal_zeros():
    import conditional_ux
    angle = 1e-8; q = np.eye(4)
    q[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    perm = [2, 0, 3, 1]; signs = np.array([-1., 1., -1., 1.])
    directions, membership = conditional_ux.direction_classes([np.eye(4), q, q[:, perm]*signs])
    assert len(directions) == 4
    np.testing.assert_array_equal(membership[1], membership[0])
    np.testing.assert_array_equal(membership[2], membership[0][perm])


def test_diagonal_removed_moment_matches_explicit_cross_subject_products():
    rng = np.random.default_rng(691); a, b = rng.normal(size=(18, 7)), rng.normal(size=(18, 5))
    raw, corrected = witness.moments(a, b)
    gram_product = (a @ a.T)*(b @ b.T)
    expected = (gram_product.sum()-np.trace(gram_product))/(len(a)*(len(a)-1))
    np.testing.assert_allclose(corrected, expected, atol=1e-12)
    np.testing.assert_allclose(raw, gram_product.mean(), atol=1e-12)


def test_conditional_Ui_features_only_use_W_and_are_reproducible():
    rng = np.random.default_rng(692); z = rng.normal(size=(80, 4)); v = np.array([1., 0., 0., 0.])
    with threadpool_limits(2):
        a, audit = witness.features(z, v, centers=8)
        b, _ = witness.features(z, v, centers=8)
    assert audit['response_values_used'] is False and audit['all_subjects'] == 80
    for name in a:
        np.testing.assert_allclose(a[name], b[name], atol=0.)
        np.testing.assert_allclose(np.mean(np.sum(a[name]**2, axis=1)), 1., atol=1e-12)


def test_packed_tensor_moments_match_each_explicit_numpy_target():
    import torch
    import run_conditional_kernel_moments as runner
    rng = np.random.default_rng(693); b = rng.normal(size=(45, 9)); g = rng.normal(size=(45, 3, 7))
    raw, corrected = runner.moments(torch.tensor(b), torch.tensor(g.reshape(45, -1)),
        torch.tensor(np.sum(g*g, axis=-1)), 7)
    for i in range(3):
        expected = witness.moments(b, g[:, i])
        np.testing.assert_allclose([raw[i], corrected[i]], expected, atol=1e-12)


def test_complete_conditional_feature_pipeline_uses_all_case_rows_and_no_derivatives(tmp_path):
    import json
    import matrix_function_profiles as profiles
    import run_conditional_kernel_moments as runner
    from test_matrix_metrics import make_cases
    source = tmp_path/'source'; make_cases(source)
    folder, output = tmp_path/'bank', tmp_path/'out'
    with threadpool_limits(2):
        profiles.prepare(source, folder)
        rows = runner.analyze(source, output, folder, centers=8, device='cpu')
    assert rows and max(row['endpoint_error'] for row in rows) < 1e-12
    protocol = json.loads((output/'protocol.json').read_text())
    assert protocol['all_subjects'] == 32 and protocol['response_derivatives_used'] is False
    assert json.loads((output/'read_audit.json').read_text())['all_original_rows'] == 160
