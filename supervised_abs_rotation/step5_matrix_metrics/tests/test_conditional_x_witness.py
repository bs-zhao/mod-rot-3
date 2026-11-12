import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import conditional_x_witness as witness
import matrix_function_profiles as profiles
import run_matrix_gp_ard as runner
from test_approximate_ux_function import shared_fixture


def test_dual_kernel_regression_matches_primal_feature_ridge_with_free_intercept():
    rng = np.random.default_rng(641); phi = rng.normal(size=(40, 7))
    train, query = np.arange(28), np.arange(28, 40)
    target = rng.normal(size=(28, 4)); ridge = .03
    mapping = witness.regression_maps(phi @ phi.T, train, query, (ridge,))[0]
    centered = phi[train]-phi[train].mean(0)
    penalty = ridge*np.sum(centered*centered)/len(train)
    beta = np.linalg.solve(centered.T @ centered+penalty*np.eye(phi.shape[1]), centered.T @ (target-target.mean(0)))
    expected = target.mean(0)+(phi[query]-phi[train].mean(0)) @ beta
    np.testing.assert_allclose(mapping @ target, expected, atol=1e-12)
    np.testing.assert_allclose(mapping.sum(1), 1., atol=1e-12)


def test_context_product_kernel_matches_explicit_tensor_witness_and_preserves_mixed_effects():
    # Balanced independent signs: E[X1|X2]=0 exactly, but E[X1*X2*X1]=0.
    # Adding the X2 context detects this interaction that marginal covariance misses.
    x1 = np.array([-1., -1., 1., 1.]); x2 = np.array([-1., 1., -1., 1.])
    context = np.column_stack((np.ones(4), x2))
    explicit = x1[:, None]*context
    gram = x1[:, None]*x1[None]*(context @ context.T)
    np.testing.assert_allclose(gram, explicit @ explicit.T)
    product = x1*x2
    assert abs(product @ x1) == 0.
    assert product @ gram @ product == 16.
    root, audit = witness.features(gram)
    np.testing.assert_allclose(root @ root.T, gram, atol=1e-12)
    assert audit['relative_negative_eigenvalue'] < 1e-12


def test_W_only_witnesses_are_PSD_centered_and_invariant_to_affine_X_units():
    rng = np.random.default_rng(642); x = rng.normal(size=(55, 3)); x[:, 1] += .6*x[:, 0]
    with threadpool_limits(limits=2):
        a, audit = witness.operators(x)
        b, _ = witness.operators(x*np.array([1e-5, -30., 200.])+np.array([.1, 3., -20.]))
    assert audit['W_only_tuning'] is True and audit['response_values_used'] is False
    for left, right in zip(a, b):
        np.testing.assert_allclose(left.T @ left, right.T @ right, atol=1e-10)
        np.testing.assert_allclose(left.sum(1), 0., atol=1e-12)
        assert abs(np.sum(left*left)-1.) < 1e-12


def test_conditional_witness_GP_complete_pipeline_uses_actual_matrix_responses(tmp_path):
    source = tmp_path/'source'; shared_fixture(source)
    # Five X nodes in the older fixture cannot support nested five-fold CV.
    # Expand the actual synthetic matrices, retaining all signed/rotated U.
    x = np.random.default_rng(643).normal(size=(15, 2))
    for path in (source/'cases').glob('*/W.npy'):
        old = np.load(path)
        np.save(path, np.column_stack((np.repeat(old[::5, :4], 15, axis=0), np.tile(x, (32, 1)))))
    u = np.load(source/'cases/baseline/W.npy')[::15, :4]
    y = (u[:, 0, None]*x[None, :, 0]+u[:, 1, None]**2*x[None, :, 1]).ravel()
    np.savez(source/'Y.npz', logit_nn=y, p_nn=1/(1+np.exp(-y)))
    np.save(source/'subject_index.npy', np.repeat(np.arange(32), 15))
    np.save(source/'trial_index.npy', np.tile(np.arange(15), 32))
    np.save(source/'trial_counts.npy', np.full(32, 15))
    index = json.loads((source/'index.json').read_text()); index['n_trials'] = 480
    (source/'index.json').write_text(json.dumps(index))
    folder, output = tmp_path/'bank', tmp_path/'out'
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        profiles.prepare(source, folder)
        rows = runner.analyze(source, output, folder, centers=8, training_steps=1, refit_steps=1,
            starts=1, model_chunk=8, device='cpu', X_contrast='conditional_witness')
    assert rows and max(r['endpoint_error'] for r in rows) < 1e-12
    protocol = json.loads((output/'protocol.json').read_text())
    assert protocol['status'] == 'COMPLETE' and protocol['X_contrast'] == 'conditional_witness'
    assert protocol['all_subject_refit_steps'] == 1 and protocol['n_subjects'] == 32
    audit = json.loads((output/'read_audit.json').read_text())
    assert audit['forbidden_files_read'] == [] and audit['original_U_model_coefficients_read'] is False
    assert audit['original_rows_in_response_preparation'] == 480
