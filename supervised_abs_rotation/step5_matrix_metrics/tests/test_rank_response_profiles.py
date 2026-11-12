import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rank_response_profiles as ranked
import matrix_function_profiles as raw
import run_matrix_gp_ard as runner
from test_approximate_ux_function import shared_fixture
from test_matrix_metrics import make_cases


@pytest.mark.parametrize('transform', ['normal_rank', 'uniform_rank'])
def test_global_output_calibration_preserves_order_ties_and_monotone_units(transform):
    y = np.array([[-3., .2], [1., .5], [1., 2.], [4., 7.], [6., -2.]])
    a = ranked.global_scores(y, transform)
    b = ranked.global_scores(np.exp(y*.2)+4., transform)
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a).all() and a[1, 0] == a[2, 0]
    for c in range(2):
        assert np.all(np.diff(a[np.argsort(y[:, c]), c]) >= 0.)


def test_ranked_shared_bank_is_exactly_transformed_actual_Y_and_detects_changes(tmp_path):
    source = tmp_path/'source'; shared_fixture(source)
    raw_folder, folder = tmp_path/'raw', tmp_path/'ranked'
    with threadpool_limits(2):
        raw.prepare(source, raw_folder)
        ranked.prepare(source, folder, raw_folder)
        bank = ranked.MatrixBank(source, folder)
    np.testing.assert_array_equal(bank.responses.transpose(0, 2, 1).reshape(160, 2), ranked.global_scores(bank.data.y))
    assert bank.protocol['all_original_rows'] == 160
    with (folder/'response_quality.json').open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='artifact changed'):
        ranked.MatrixBank(source, folder)


def test_unshared_ranked_profiles_transform_actual_Y_before_X_interpolation(tmp_path):
    source = tmp_path/'source'; make_cases(source)
    raw_folder, folder = tmp_path/'raw', tmp_path/'ranked'
    with threadpool_limits(2):
        raw.prepare(source, raw_folder)
        ranked.prepare(source, folder, raw_folder)
        bank = ranked.MatrixBank(source, folder)
    assert bank.quality['mode'] == 'generic X dictionary evaluated at pooled observed X probes'
    assert bank.quality['calibration_original_rows'] == 160
    assert bank.protocol['calibration_includes_all_subject_Y'] is True


def test_complete_GP_pipeline_declares_frozen_global_calibration_and_all_subject_refit(tmp_path):
    # The five-context shared fixture cannot support nested conditional-X
    # regression. The unshared fixture supplies 128 observed X probes.
    source = tmp_path/'source'; make_cases(source)
    raw_folder, folder, output = tmp_path/'raw', tmp_path/'ranked', tmp_path/'out'
    torch.set_num_threads(2)
    with threadpool_limits(2):
        raw.prepare(source, raw_folder)
        ranked.prepare(source, folder, raw_folder)
        rows = runner.analyze(source, output, folder, centers=8, training_steps=1, refit_steps=1,
            starts=1, device='cpu', bank_factory=ranked.MatrixBank, X_contrast='conditional_witness')
    assert rows and max(row['endpoint_error'] for row in rows) < 1e-12
    protocol = json.loads((output/'protocol.json').read_text())
    assert protocol['global_output_transform'] == 'normal_rank'
    assert 'all-data rank calibration' in protocol['heldout_interpretation']
    assert json.loads((output/'read_audit.json').read_text())['all_subjects_in_GP_refit'] == 32
