import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warped_gp as warp


def fixture():
    rng = np.random.default_rng(701); y = np.sort(rng.normal(size=45)); knots = np.quantile(y, np.linspace(0., 1., 8))
    phi = rng.normal(size=(45, 10)); h, bins = warp.hat_basis(y, knots)
    model = warp.Model(phi.T @ phi, phi.T @ h, h.T @ h, h.sum(0), np.bincount(bins, minlength=7), knots)
    return phi, h, model


def test_warped_GP_sufficient_density_and_posterior_match_dense_covariance():
    torch.set_num_threads(2); phi, h, model = fixture()
    with torch.no_grad():
        model.log_slopes.copy_(torch.linspace(-.4, .3, 7, dtype=torch.float64))
    p, basis = torch.tensor(phi), torch.tensor(h)
    values = basis @ model.heights()
    covariance = torch.exp(2*model.log_amplitude)*(p @ p.T)+torch.exp(2*model.log_noise)*torch.eye(len(p), dtype=torch.float64)
    chol = torch.linalg.cholesky(covariance)
    expected = torch.log(torch.diagonal(chol)).sum()+.5*values @ torch.cholesky_solve(values[:, None], chol)[:, 0]
    expected -= model.counts @ torch.log(torch.diff(model.heights())/torch.diff(model.knots))
    expected += .5*torch.diff(model.log_slopes).square().sum()
    torch.testing.assert_close(model.nll(), expected, rtol=1e-9, atol=1e-7)
    posterior = torch.exp(2*model.log_amplitude)*p.T @ torch.linalg.solve(covariance, values)
    torch.testing.assert_close(model.coefficients(), posterior, rtol=1e-9, atol=1e-8)


def test_positive_global_warp_and_linear_extrapolation_are_invertible():
    _, _, model = fixture()
    heights = model.heights().detach().numpy(); knots = model.knots.numpy()
    assert np.all(np.diff(heights) > 0.)
    y = np.linspace(knots[0]-1., knots[-1]+1., 100)
    basis, _ = warp.hat_basis(y, knots)
    transformed = basis @ heights
    np.testing.assert_allclose(warp.inverse(transformed, knots, heights), y, atol=1e-12)


def test_warp_training_improves_its_defined_density_without_flat_output():
    _, _, model = fixture(); start = float(model.nll().detach())
    history = warp.fit(model, steps=10)
    assert float(model.nll().detach()) < start
    h = model.heights().detach().numpy()
    assert np.all(np.diff(h) > 0.) and np.isfinite(h).all()
    assert history[-1]['step'] == 10


def test_joint_W_kernel_dictionary_is_exactly_U_rotation_equivariant():
    from joint_warp_source import Dictionary
    rng = np.random.default_rng(702); w = rng.normal(size=(60, 7)); q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    rotated = w.copy(); rotated[:, :4] = w[:, :4] @ q
    a = Dictionary(np.zeros(7), np.ones(7), w[:12], 4, 'cpu')
    b = Dictionary(np.zeros(7), np.ones(7), rotated[:12], 4, 'cpu')
    np.testing.assert_allclose(a(w).numpy() @ a(w).numpy().T,
                               b(rotated).numpy() @ b(rotated).numpy().T, atol=1e-12)


def test_global_warp_source_streams_every_actual_row_and_refits_all_subjects(tmp_path):
    import json
    from threadpoolctl import threadpool_limits
    import joint_warp_source as source_runner
    from test_matrix_metrics import make_cases
    source = tmp_path/'source'; make_cases(source); output = tmp_path/'out'
    torch.set_num_threads(2)
    with threadpool_limits(2):
        rows = source_runner.analyze(source, output, centers=8, knot_count=9, steps=3, chunk=71, device='cpu')
    assert rows and all(np.isfinite(row['learned_original_Y_validation_nmse']) for row in rows)
    protocol = json.loads((output/'protocol.json').read_text())
    assert protocol['status'] == 'COMPLETE' and protocol['jacobian_included'] is True
    assert protocol['all_original_rows'] == 160 and protocol['all_subjects'] == 32
    audit = json.loads((output/'read_audit.json').read_text())
    assert audit['forbidden_files_read'] == [] and audit['all_actual_rows_in_sufficient_statistics'] == 160
