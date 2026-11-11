import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cp_regression as cp
from cp_grid_als import GridALS
import run_cp_grid_als as runner
import joint_bank
import prepare_response_bank as prep
from test_approximate_ux_function import shared_fixture
from test_matrix_metrics import make_cases


def fixture(rank=2, previous=None):
    rng = np.random.default_rng(561)
    u = rng.normal(size=(2, 70, 3)); u[:, :, 1] += .7*u[:, :, 0]
    x = rng.normal(size=(45, 2)); x[:, 1] -= .5*x[:, 0]
    up, _, _, au = cp.whiten(cp.features(u))
    xp, _, _, ax = cp.whiten(cp.features(x))
    y = np.array([np.outer(u[0, :, 0], x[:, 0])+np.outer(u[0, :, 1]**2, x[:, 1]),
                  np.outer(u[0, :, 2], x[:, 0]*x[:, 1])])
    y -= y.mean((1, 2), keepdims=True); y /= y.std((1, 2), keepdims=True)
    args = [torch.tensor(a, dtype=torch.float64) for a in (up, xp, au, ax, y)]
    return GridALS(*args, groups=2, channels=2, starts=2, rank=rank, previous=previous)


def test_exact_grid_contractions_equal_explicit_loss_with_dependent_scalar_coordinates():
    torch.set_num_threads(2)
    model = fixture()
    contracted = model.loss(model_chunk=3)
    for k in range(len(model.u)):
        direct = (model.predict_grid(k)-model.targets[model.channel_index[k]]).square().mean()
        assert contracted[k].item() == pytest.approx(direct.item(), abs=1e-12)


def test_each_linear_ALS_block_decreases_actual_prediction_error():
    torch.set_num_threads(2); model = fixture()
    before = model.loss(); initial = before.clone()
    for block, count in (('u', 3), ('x', 2)):
        for axis in range(count):
            model.update(block, axis, ridge=1e-10, model_chunk=3)
            after = model.loss()
            assert torch.all(after <= before+1e-8)
            before = after
    assert before.mean() < initial.mean()-1e-6
    for k in range(len(model.u)):
        direct = (model.predict_grid(k)-model.targets[model.channel_index[k]]).square().mean()
        assert before[k].item() == pytest.approx(direct.item(), abs=1e-10)


def test_zero_amplitude_warm_components_can_enter_and_keep_initial_prediction():
    torch.set_num_threads(2); small = fixture(rank=1)
    small.update('u', 0)
    larger = fixture(rank=3, previous=small.snapshot())
    torch.testing.assert_close(small.loss(), larger.loss(), rtol=1e-9, atol=1e-10)
    assert torch.count_nonzero(larger.amplitude[:, 1:]) == 0
    larger.update('u', 1)
    assert torch.count_nonzero(larger.amplitude[:, 1:]) > 0


@pytest.mark.parametrize('teacher,shared', [('poly4_UX', True), ('rbf2_UX', True),
                                           ('poly4_UX', False), ('rbf2_UX', False)])
def test_full_grid_function_pipeline_retains_source_rows_and_declares_reference(tmp_path, teacher, shared):
    import json
    torch.set_num_threads(2)
    source = tmp_path/'step2'
    (shared_fixture if shared else make_cases)(source)
    parent = tmp_path/'step5_matrix_metrics'
    with threadpool_limits(limits=2):
        if teacher == 'poly4_UX':
            joint_bank.prepare(source, parent/'joint_bank_seed391', degrees=(4,), device='numpy')
        if teacher == 'rbf2_UX' or not shared:
            prep.prepare(source, parent/'UX_complexity_source_seed383', estimators=('rbf2',), centers=12)
        output = tmp_path/'out'
        rows = runner.analyze(source, output, teacher=teacher, ranks=(1, 2), sweeps=1,
                              starts=1, device='cpu', model_chunk=5)
    assert rows
    audit = json.loads((output/'read_audit.json').read_text())
    assert audit['forbidden_files_read'] == []
    assert audit['all_original_rows_in_source_fit'] == 160
    assert audit['all_subjects'] == 32
    assert audit['exact_original_grid'] == shared
    assert json.loads((output/'protocol.json').read_text())['status'] == 'COMPLETE'
