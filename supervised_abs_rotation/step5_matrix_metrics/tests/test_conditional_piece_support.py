import sys
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import conditional_piece_support as pieces
import conditional_ux
import prepare_response_bank as prep
import run_conditional_piece_support as runner
import surrogates
from test_matrix_metrics import make_cases


def test_piece_support_respects_user_sparsity_example_and_whole_field_scaling():
    nodes = np.linspace(-1., 1., 17)
    def table(amplitudes):
        values = np.asarray(amplitudes)[None, None, None, :, None, None]*nodes[None, None, None, None, :, None]
        return torch.tensor(np.broadcast_to(values, (5, 2, 3, 4, 17, 6)).copy(), dtype=torch.float64)
    a, b = table([2., 2., .01, .01]), table([1., .5, .1, .1])
    widths = np.full((5, 4), 2.)
    sa, sb = pieces.piece_costs(a, widths), pieces.piece_costs(b, widths)
    changed = pieces.piece_costs(a*torch.tensor([1e-200, 1., 1e200], dtype=torch.float64)[None, None, :, None, None, None], widths)
    for key in sa:
        assert np.all(sa[key] < sb[key])
        np.testing.assert_allclose(sa[key], changed[key], atol=1e-12)


def test_conditional_pieces_do_not_penalize_sign_changes_caused_by_other_U_variables():
    rng = np.random.default_rng(641); z = rng.normal(size=(60, 2))
    powers = surrogates.powers_for(2, 2)
    coef = np.zeros((len(powers), 2)); coef[np.flatnonzero(np.all(powers == [1, 1], axis=1))[0]] = [1., -1.]
    evaluator = pieces.Evaluator(dict(kind='poly', powers=powers), coef, device='cpu')
    bounds = np.tile(np.array([-1., 1.]), (60, 2, 1)); widths = np.full((60, 2), 2.)
    q = np.array([[1., -1.], [1., 1.]])/np.sqrt(2.)
    costs = []
    for transform in (np.eye(2), q):
        probes = pieces.probes_for(z @ transform, bounds, transform, 17)
        values = evaluator(probes).reshape(60, 2, 17, 1, 1, 2).permute(0, 3, 4, 1, 2, 5)
        costs.append(pieces.piece_costs(values, widths))
    # U1*U2 is monotone in either own coordinate with the other fixed,
    # even though each derivative changes sign across different people.
    np.testing.assert_allclose(costs[0]['eps030'], costs[0]['active_eps030'], atol=1e-12)
    assert costs[0]['eps030'].mean() < costs[1]['eps030'].mean()-.5


def test_RBF_value_evaluator_matches_the_verified_source_feature_definition():
    rng = np.random.default_rng(642)
    meta = dict(kind='rbf', centers=rng.normal(size=(9, 3)), lengthscale=1.7)
    coef = rng.normal(size=(10, 7)); probes = rng.normal(size=(40, 3))
    actual = pieces.Evaluator(meta, coef, device='cpu')(probes).numpy()
    np.testing.assert_allclose(actual, conditional_ux.source_features(probes, meta) @ coef, atol=1e-12)


def test_full_conditional_piece_pipeline_covers_every_subject_and_X_profile(tmp_path):
    import json
    torch.set_num_threads(2)
    source = tmp_path/'source'; make_cases(source)
    folder, output = tmp_path/'bank', tmp_path/'out'
    with threadpool_limits(limits=2):
        prep.prepare(source, folder, estimators=('poly3',), centers=10)
        rows = runner.analyze(source, output, folder, estimators=('poly3',), nodes=5, neighbors=16,
                              subjects_per_chunk=8, device='cpu')
    assert rows
    assert max(r['endpoint_error'] for r in rows) < 1e-12
    audit = json.loads((output/'read_audit.json').read_text())
    assert audit['all_subjects'] == 32 and audit['all_original_rows_in_source_fit'] == 160
    assert audit['all_source_X_contexts_scored'] > 1 and audit['forbidden_files_read'] == []
    assert json.loads((output/'protocol.json').read_text())['status'] == 'COMPLETE'


def test_each_X_context_can_change_units_without_changing_raw_monotonicity_cost():
    nodes = np.linspace(-1., 1., 17)
    values = np.array([2., 2., .01, .01])[None, None, None, :, None, None]*nodes[None, None, None, None, :, None]
    values = torch.tensor(np.broadcast_to(values, (5, 2, 1, 4, 17, 3)).copy(), dtype=torch.float64)
    scales = torch.tensor([1e-200, 1., 1e200], dtype=torch.float64)
    a = pieces.piece_costs(values, np.full((5, 4), 2.), reference='context')
    b = pieces.piece_costs(values*scales, np.full((5, 4), 2.), reference='context')
    for key in a:
        np.testing.assert_allclose(a[key], b[key], atol=1e-12)


def test_raw_response_monotonicity_pipeline_does_not_use_X_omission(tmp_path):
    import json
    torch.set_num_threads(2)
    source = tmp_path/'source'; make_cases(source)
    folder, output = tmp_path/'bank', tmp_path/'out'
    with threadpool_limits(limits=2):
        prep.prepare(source, folder, estimators=('poly3',), centers=10)
        rows = runner.analyze(source, output, folder, estimators=('poly3',), nodes=5, neighbors=16,
            subjects_per_chunk=8, device='cpu', effect_view='raw_response')
    assert rows and all(row['metric'].startswith('U_raw_conditional_') for row in rows)
    assert any('monotonicity_excess' in row['metric'] for row in rows)
    assert json.loads((output/'X_crossfit_audit.json').read_text())['X_omission_performed'] is False
    assert max(row['endpoint_error'] for row in rows) < 1e-12
