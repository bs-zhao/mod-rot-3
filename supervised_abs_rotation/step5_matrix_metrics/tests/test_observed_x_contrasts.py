import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import observed_x_contrasts as ox
import fitted_bank
import prepare_response_bank as prep
import run_observed_x_contrasts as runner
from test_matrix_metrics import make_cases


def test_local_contrasts_recover_affine_effects_on_correlated_observed_design():
    rng = np.random.default_rng(511); x = rng.normal(size=(120, 4)); x[:, 1] += .8*x[:, 0]
    scale = x.std(0); operator, audit = ox.contrast_operator(x, scale, neighbors=24)
    recovered = (operator @ (x/scale)).reshape(len(x), 4, 4)
    np.testing.assert_allclose(recovered, np.broadcast_to(np.eye(4), recovered.shape), atol=1e-10)
    np.testing.assert_allclose(operator @ np.ones(len(x)), 0., atol=1e-12)
    assert audit['full_rank_node_fraction'] == 1.
    gradient = rng.normal(size=(12, 2, len(x), 3))
    moments = ox.moments(gradient, operator, 4, chunk=5)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated = ox.moments(gradient @ q, operator, 4, chunk=7)
    for key in moments:
        expected = q.T @ moments[key] @ q
        np.testing.assert_allclose(rotated[key], expected, atol=1e-9)


def test_gradient_only_kernel_reconstruction_matches_saved_function_derivatives(tmp_path):
    source = tmp_path/'source'; make_cases(source)
    folder = tmp_path/'bank'
    with threadpool_limits(limits=2):
        prep.prepare(source, folder, estimators=('rbf1',), centers=12)
        bank = fitted_bank.FittedBank(source, folder)
        actual = ox.source_gradient(bank, 'rbf1')
        with np.load(folder/'model_rbf1.npz', allow_pickle=False) as saved:
            scaler = object.__new__(ox.wave.sur.Scaler); scaler.mean, scaler.scale = saved['mean'], saved['scale']
            model = ox.wave.sur.RBF(scaler, saved['centers'], 1.); model.set_coef(saved['coef'])
        expected, _ = model.derivatives(bank.u)
    np.testing.assert_allclose(actual, expected.reshape(actual.shape), atol=1e-9)


def test_observed_contrast_screen_completes_with_explicit_node_coverage(tmp_path):
    import json
    source = tmp_path/'source'; make_cases(source)
    folder = tmp_path/'bank'; output = tmp_path/'out'
    with threadpool_limits(limits=2):
        prep.prepare(source, folder, estimators=('rbf1',), centers=12)
        rows = runner.analyze(source, output, folder, neighbors=(16,))
    assert rows
    assert json.loads((output/'protocol.json').read_text())['status'] == 'COMPLETE'
    audit = json.loads((output/'read_audit.json').read_text())
    assert audit['all_original_rows_in_source_fit'] == 160
    assert audit['scored_X_nodes'] == 128
    assert audit['forbidden_files_read'] == []
