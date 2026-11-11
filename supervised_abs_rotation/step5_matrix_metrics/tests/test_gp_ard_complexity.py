import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gp_ard_complexity as gp
import run_gp_ard_complexity as runner
import prepare_response_bank as prep
from test_matrix_metrics import make_cases


def model_fixture(device='cpu'):
    rng = np.random.default_rng(621)
    u = rng.normal(size=(40, 3)); q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    targets = rng.normal(size=(2, 40, 5))
    targets /= np.sqrt(np.mean(np.sum(targets**2, axis=-1), axis=1))[:, None, None]
    return gp.Model(torch.tensor(np.array([u, u @ q]), dtype=torch.float64, device=device),
        torch.tensor(targets, dtype=torch.float64, device=device), np.arange(8), starts=1)


def test_Woodbury_GP_evidence_and_posterior_match_dense_covariance():
    torch.set_num_threads(2); model = model_fixture()
    ids, rows = torch.arange(4), torch.arange(40)
    actual = model.nll(ids, rows)
    phi = model.features(ids, rows)
    for k in range(4):
        y = model.targets[model.task_index[k]]
        covariance = torch.exp(2*model.log_amplitude[k])*(phi[k] @ phi[k].T)+torch.exp(2*model.log_noise[k])*torch.eye(40, dtype=torch.float64)
        chol = torch.linalg.cholesky(covariance)
        expected = torch.log(torch.diagonal(chol)).sum()+.5*(y*torch.cholesky_solve(y, chol)).sum()
        torch.testing.assert_close(actual[k], expected, rtol=1e-9, atol=1e-8)
    # Isotropic starts on W-derived rotated cases must be identical.
    torch.testing.assert_close(actual[0], actual[1], rtol=1e-10, atol=1e-8)
    training, query = torch.arange(30), torch.arange(30, 40)
    error = model.prediction_error(ids, training, query)
    for k in range(4):
        amplitude = torch.exp(2*model.log_amplitude[k]); noise = torch.exp(2*model.log_noise[k])
        covariance = amplitude*(phi[k, :30] @ phi[k, :30].T)+noise*torch.eye(30, dtype=torch.float64)
        y = model.targets[model.task_index[k]]
        predicted = amplitude*phi[k, 30:] @ phi[k, :30].T @ torch.linalg.solve(covariance, y[:30])
        expected = (predicted-y[30:]).square().sum()/y[30:].square().sum()
        torch.testing.assert_close(error[k], expected, rtol=1e-9, atol=1e-9)


def test_GP_hyperparameter_gradients_are_independent_across_models():
    model = model_fixture()
    model.nll(torch.tensor([0]), torch.arange(40)).sum().backward()
    for parameter in model.parameters():
        assert torch.count_nonzero(parameter.grad[1:]) == 0
    assert torch.count_nonzero(model.eta.grad[0]) > 0


def test_function_mode_reconstruction_has_an_explicit_relative_error_budget():
    rng = np.random.default_rng(622)
    a = rng.normal(size=(100, 8)); a -= a.mean(0)
    z, _ = np.linalg.qr(a); z *= 10.
    gram = np.diag(np.geomspace(1., 1e-9, 8))
    target, audit = gp.target_modes(z, gram, tolerance=1e-6)
    assert audit['relative_squared_reconstruction_error'] <= 1e-6
    assert np.mean(np.sum(target**2, axis=1)) == pytest.approx(audit['retained_fraction'], abs=1e-12)


def test_ARD_learns_single_variable_support_and_its_rotated_two_variable_version():
    import conditional_ux
    rng = np.random.default_rng(623); u = rng.normal(size=(180, 3))
    q = np.eye(3); q[:2, :2] = np.array([[1., -1.], [1., 1.]])/np.sqrt(2.)
    y = np.sin(.7*u[:, 0]); y -= y.mean(); y /= y.std()
    torch.set_num_threads(2)
    model = gp.Model(torch.tensor(np.array([u, u @ q]), dtype=torch.float64),
        torch.tensor(y[None, :, None], dtype=torch.float64), np.arange(24), starts=1)
    gp.optimize(model, torch.arange(len(u)), steps=120, chunk=2)
    error = model.prediction_error(torch.arange(2), torch.arange(len(u)), torch.arange(len(u)))
    assert float(error.max()) < .01
    eta = model.eta.detach().numpy()
    sparse = conditional_ux.sparsity(np.tile((eta[0]**2)[:, None], (1, 6)))
    rotated = conditional_ux.sparsity(np.tile((eta[1]**2)[:, None], (1, 6)))
    assert sparse['UX_columns_mean_effective_support_p1'] < rotated['UX_columns_mean_effective_support_p1']-.4


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_CPU_CUDA_GP_evidence_and_gradient_agree():
    a, b = model_fixture(), model_fixture('cuda')
    la = a.nll(torch.arange(4), torch.arange(40)).sum()
    lb = b.nll(torch.arange(4, device='cuda'), torch.arange(40, device='cuda')).sum()
    la.backward(); lb.backward()
    torch.testing.assert_close(la, lb.cpu(), rtol=1e-8, atol=1e-6)
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa.grad, pb.grad.cpu(), rtol=1e-6, atol=1e-5)


def test_full_GP_ARD_pipeline_refits_every_subject_using_only_case_matrices(tmp_path):
    import json
    torch.set_num_threads(2)
    source = tmp_path/'source'; make_cases(source)
    folder, output = tmp_path/'bank', tmp_path/'out'
    with threadpool_limits(limits=2):
        prep.prepare(source, folder, estimators=('poly3',), centers=10)
        rows = runner.analyze(source, output, folder, estimators=('poly3',), centers=8,
            training_steps=1, refit_steps=1, starts=1, model_chunk=8, device='cpu')
    assert rows
    assert max(r['endpoint_error'] for r in rows) < 1e-12
    audit = json.loads((output/'read_audit.json').read_text())
    assert audit['all_subjects_in_GP_refit'] == 32
    assert audit['all_original_rows_in_source_fit'] == 160
    assert audit['forbidden_files_read'] == []
    assert json.loads((output/'protocol.json').read_text())['status'] == 'COMPLETE'
