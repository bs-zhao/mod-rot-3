import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gp_posterior_effects as posterior
from test_gp_ard_complexity import model_fixture


def test_posterior_derivative_energies_match_autograd_at_noncenter_queries():
    torch.set_num_threads(2); model = model_fixture()
    ids, rows = torch.tensor([0]), torch.arange(40)
    energy, fidelity = posterior.effects(model, ids, rows, steps=(1e-4,))
    with torch.no_grad():
        _, _, _, ratio, _, _, solved = model.system(ids, rows)
        beta = ratio[:, None, None]*solved
    model.u.requires_grad_(True)
    predicted = model.features(ids, torch.arange(30, 40)) @ beta
    expected = torch.zeros(10, 3, dtype=torch.float64)
    for c in range(predicted.shape[-1]):
        derivative, = torch.autograd.grad(predicted[0, :, c].sum(), model.u, retain_graph=True)
        expected += derivative[0, 30:40].square()
    np.testing.assert_allclose(energy['derivative'][0, 30:40], expected.numpy(), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(energy['finite_0.0001'], energy['derivative'], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(fidelity, model.prediction_error(ids, rows, rows).numpy(), rtol=1e-10, atol=1e-12)


def test_posterior_UX_support_obeys_the_user_example_and_independent_X_units():
    a = np.broadcast_to((np.array([2., 2., .01, .01])**2)[None, :, None], (20, 4, 3)).copy()
    b = np.broadcast_to((np.array([1., .5, .1, .1])**2)[None, :, None], (20, 4, 3)).copy()
    sa, sb = posterior.scores(a), posterior.scores(b)
    changed = posterior.scores(a*np.array([1e-100, 1., 1e100])[None, None])
    for metric in sa:
        if metric.startswith('point_'):
            np.testing.assert_allclose(sa[metric], changed[metric], atol=1e-12)
            assert sa[metric] <= sb[metric]+1e-12


def test_full_frozen_GP_posterior_pipeline_reproduces_original_fit_and_all_subjects(tmp_path):
    import json
    from threadpoolctl import threadpool_limits
    import matrix_function_profiles as profiles
    import run_matrix_gp_ard as gp_runner
    import run_gp_posterior_support as runner
    from test_matrix_metrics import make_cases
    source = tmp_path/'source'; make_cases(source)
    parent = tmp_path/'step5'; parent.mkdir()
    bank, original, output = parent/'matrix_response_table_seed383', parent/'original', parent/'posterior'
    torch.set_num_threads(2)
    with threadpool_limits(2):
        profiles.prepare(source, bank)
        gp_runner.analyze(source, original, bank, centers=8, training_steps=1, refit_steps=1,
                          starts=1, device='cpu', X_contrast='conditional_witness')
        rows = runner.analyze(source, original, output, device='cpu')
    assert rows and max(r['endpoint_error'] for r in rows) < 1e-12
    protocol = json.loads((output/'protocol.json').read_text())
    assert protocol['GP_refitted'] is False and protocol['all_subjects'] == 32
    assert protocol['original_GP_prediction_error_reproduction_maximum'] < 1e-10
    assert json.loads((output/'read_audit.json').read_text())['forbidden_files_read'] == []
