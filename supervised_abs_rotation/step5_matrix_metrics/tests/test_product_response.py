import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import product_response as pr
import matrix_io
import prepare_response_bank as prep
import fitted_bank
import run_product_response as runner
from test_matrix_metrics import make_cases


def test_x_dictionary_standardized_derivatives_and_product_all_row_holdout(tmp_path):
    source = tmp_path/'source'; make_cases(source)
    data = matrix_io.MatrixCases(source, ('logit_nn', 'p_nn'), 383)
    dictionary = pr.XDictionary(data, centers=12)
    x = np.asarray(data.w[50:56, data.d:], dtype=float)
    phi, gradient = dictionary.evaluate(x, gradient=True)
    for j in range(data.dx):
        delta = np.zeros(data.dx); delta[j] = 1e-5*dictionary.scaler.scale[j]
        numerical = (dictionary.evaluate(x+delta)-dictionary.evaluate(x-delta))/(2e-5)
        np.testing.assert_allclose(gradient[:, :, j], numerical, atol=1e-7)
    folder = tmp_path/'function_bank'
    with threadpool_limits(limits=2):
        prep.prepare(source, folder, estimators=('rbf1',), centers=12)
        bank = fitted_bank.FittedBank(source, folder)
        with np.load(folder/'response_bank.npz', allow_pickle=False) as saved:
            probes = saved['x']
        mapping, quality = pr.x_projection(bank, dictionary, probes)
        moments, audit = pr.moments(bank, dictionary, 'rbf1', mapping)
    assert mapping.shape[0] == len(probes)
    assert quality['validation_u_subjects_used_to_select_ridge'] == 0
    assert audit['all_original_rows'] == 160
    assert audit['raw_validation_rows'] == 40
    assert audit['all_subjects'] == 32
    assert set(moments) == set(pr.PROFILES)
    assert all(np.isfinite(v).all() and v.shape[0] == 32 for v in moments.values())


def test_complete_product_kernel_screen_preserves_raw_holdout_and_no_forbidden_inputs(tmp_path):
    import json
    source = tmp_path/'source'; make_cases(source)
    folder = tmp_path/'bank'; output = tmp_path/'out'
    with threadpool_limits(limits=2):
        prep.prepare(source, folder, estimators=('rbf1',), centers=12)
        rows = runner.analyze(source, output, folder)
    assert rows
    protocol = json.loads((output/'protocol.json').read_text())
    assert protocol['status'] == 'COMPLETE'
    assert protocol['all_original_rows_used_in_metric'] is True
    audit = json.loads((output/'read_audit.json').read_text())
    assert audit['all_original_rows_in_fit_and_metric'] == 160
    assert audit['forbidden_files_read'] == []
    prediction = json.loads((output/'raw_prediction_audit.json').read_text())['rbf1']
    assert prediction['raw_validation_rows'] == 40
