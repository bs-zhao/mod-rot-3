import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isotropic_gp_rotation as radial
import gp_ard_complexity as gp
import gp_posterior_effects as posterior
from test_gp_ard_complexity import model_fixture


def test_projected_radial_GP_stays_isotropic_and_has_rotation_equivariant_fits():
    torch.set_num_threads(2); old = model_fixture()
    model = radial.model(old.u, old.targets, np.arange(8), starts=1)
    gp.optimize(model, torch.arange(40), steps=4, chunk=4)
    eta = model.eta.detach().numpy()
    np.testing.assert_allclose(eta, np.repeat(eta[:, :1], 3, axis=1), atol=0., rtol=0.)
    np.testing.assert_allclose(eta[::2], eta[1::2], atol=1e-10)


def test_covariance_diagonal_matches_actual_GP_derivative_energy():
    model = model_fixture(); ids, rows = torch.arange(4), torch.arange(40)
    cov, error = radial.covariances(model, ids, rows)
    actual, other_error = posterior.effects(model, ids, rows)
    np.testing.assert_allclose(np.diagonal(cov, axis1=-2, axis2=-1), actual['derivative'], atol=1e-12)
    np.testing.assert_allclose(error, other_error, atol=1e-12)


def test_rotated_function_covariance_matches_explicit_gradient_vectors():
    rng = np.random.default_rng(681); gradients = rng.normal(size=(30, 6, 4, 8))
    cov = np.einsum('nxir,nxjr->nxij', gradients, gradients)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    expected = np.einsum('ai,nxar->nxir', q, gradients)
    expected = np.sum(expected*expected, axis=-1).transpose(0, 2, 1)
    np.testing.assert_allclose(radial.energies(cov, q), expected, atol=1e-12)
    expected_primary = radial.scores(cov, q)['point_UX_columns_Xmean_subject_mean_soft_count_eps010']
    np.testing.assert_allclose(radial.primary_value(cov, q), expected_primary, atol=1e-12)


def test_full_radial_GP_pipeline_fits_once_and_preserves_all_rows(tmp_path):
    import json
    from threadpoolctl import threadpool_limits
    import matrix_function_profiles as profiles
    import run_isotropic_gp_rotation as runner
    from test_matrix_metrics import make_cases
    source = tmp_path/'source'; make_cases(source)
    folder, output = tmp_path/'bank', tmp_path/'out'
    torch.set_num_threads(2)
    with threadpool_limits(2):
        profiles.prepare(source, folder)
        rows = runner.analyze(source, output, folder, centers=8, training_steps=1, refit_steps=1,
            random_count=2, optimizer_starts=0, device='cpu')
        import run_norm_lift_profiles as lift_runner
        lift_runner.analyze(source,output,folder,tmp_path/'lift',device='cpu')
    assert rows and max(row['endpoint_error'] for row in rows) < 1e-12
    assessment = json.loads((output/'assessment.json').read_text())
    assert assessment['source_GP_refitted_per_rotation'] is False
    assert json.loads((output/'protocol.json').read_text())['all_subjects'] == 32
    audits=json.loads((tmp_path/'lift'/'gradient_replay_and_trace_audits.json').read_text())
    assert max(r['covariance_replay_relative_L2_error'] for r in audits)<1e-12
    assert json.loads((tmp_path/'lift'/'protocol.json').read_text())['all_subjects']==32


def test_fixed_function_pipeline_accepts_an_audited_ordinal_bank_and_explicit_primary(tmp_path):
    import json
    from threadpoolctl import threadpool_limits
    import matrix_function_profiles as profiles
    import rank_response_profiles as ranked
    import run_isotropic_gp_rotation as runner
    from test_matrix_metrics import make_cases
    source = tmp_path/'source'; make_cases(source)
    raw, folder, output = tmp_path/'raw', tmp_path/'ranked', tmp_path/'out'
    metric = 'pooled_UX_columns_worst_soft_count_eps003'
    torch.set_num_threads(2)
    with threadpool_limits(2):
        profiles.prepare(source, raw)
        ranked.prepare(source, folder, raw)
        runner.analyze(source, output, folder, centers=8, training_steps=1, refit_steps=1,
            random_count=2, optimizer_starts=0, device='cpu', bank_factory=ranked.MatrixBank, primary_metric=metric)
    protocol = json.loads((output/'protocol.json').read_text())
    assert protocol['global_output_transform'] == 'normal_rank'
    assert json.loads((output/'assessment.json').read_text())['primary_metric'] == metric
