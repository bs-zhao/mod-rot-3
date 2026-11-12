import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matrix_function_profiles as profiles
import run_matrix_gp_ard as runner
from test_approximate_ux_function import shared_fixture
from test_matrix_metrics import make_cases


def test_raw_shared_table_is_identical_to_every_original_Y_row(tmp_path):
    source = tmp_path/'source'; shared_fixture(source); folder = tmp_path/'bank'
    with threadpool_limits(limits=2):
        profiles.prepare(source, folder)
        bank = profiles.MatrixBank(source, folder)
    np.testing.assert_array_equal(bank.responses.transpose(0, 2, 1).reshape(160, 2), bank.data.y)
    assert bank.quality['validation_nmse'] == [0., 0.]
    assert bank.protocol['U_function_approximation_before_GP'] is False


def test_matrix_targets_remove_pure_U_X_and_normalize_each_contrast_amplitude():
    rng = np.random.default_rng(631); bank = rng.normal(size=(60, 2, 20))
    maps = [np.eye(20), np.eye(20)*3.]
    tasks, a, tail, audits = profiles.task_modes(bank, maps, tolerance=1e-5)
    changed = bank+rng.normal(size=(60, 2, 1))+rng.normal(size=(1, 2, 20))
    _, b, _, _ = profiles.task_modes(changed, maps, tolerance=1e-5)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x @ x.T, y @ y.T, atol=1e-10)
    np.testing.assert_allclose(a[0] @ a[0].T, a[1] @ a[1].T, atol=1e-10)
    assert max(r.get('relative_squared_reconstruction_error', 0.) for r in audits) <= 1e-5
    for target, discarded, audit in zip(a, tail, audits):
        assert (np.sum(target*target)+discarded.sum())/len(bank) == pytest.approx(1., abs=1e-12)


def test_full_fidelity_includes_the_exact_discarded_output_energy():
    actual = profiles.full_error(np.array([.01, .03]), np.array([9., 6.]), np.array([1., 4.]))
    np.testing.assert_allclose(actual, [.109, .418])


def test_unshared_X_profile_fallback_declares_its_approximation(tmp_path):
    source = tmp_path/'source'; make_cases(source); folder = tmp_path/'bank'
    with threadpool_limits(limits=2):
        profiles.prepare(source, folder)
        bank = profiles.MatrixBank(source, folder)
    assert bank.quality['mode'] == 'generic X dictionary evaluated at pooled observed X probes'
    assert bank.quality['common_support_verified'] is False
    assert bank.protocol['all_original_rows'] == 160


def test_direct_matrix_GP_pipeline_uses_all_subjects_and_no_U_teacher(tmp_path):
    import json
    torch.set_num_threads(2)
    source = tmp_path/'source'; shared_fixture(source)
    folder, output = tmp_path/'bank', tmp_path/'out'
    with threadpool_limits(limits=2):
        profiles.prepare(source, folder)
        rows = runner.analyze(source, output, folder, centers=8, training_steps=1, refit_steps=1,
                              starts=1, model_chunk=8, device='cpu')
    assert rows
    protocol = json.loads((output/'protocol.json').read_text())
    assert protocol['status'] == 'COMPLETE' and protocol['exact_original_response_grid'] is True
    assert protocol['U_function_approximation_before_GP'] is False
    audit = json.loads((output/'read_audit.json').read_text())
    assert audit['all_subjects_in_GP_refit'] == 32
    assert audit['original_rows_in_response_preparation'] == 160
    assert audit['original_U_model_coefficients_read'] is False
    assert audit['forbidden_files_read'] == []
