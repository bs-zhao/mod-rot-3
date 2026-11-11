import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import approximate_ux_function as teacher
import joint_bank
import prepare_response_bank as prep
import run_cp_regression as cp_runner
import matrix_io
from test_matrix_metrics import make_cases


def shared_fixture(folder):
    make_cases(folder)
    common = np.array([[-1., -.5], [0., 1.], [.5, -.5], [1., 0.], [1.5, 1.5]])
    base = np.load(folder/'cases/baseline/W.npy')
    u = base[::5, :4]
    logit = (u[:, 0, None]*common[None, :, 0] + u[:, 1, None]**2*common[None, :, 1]
             + .3*u[:, 2, None] + common[None, :, 0]**2).ravel()
    for path in (folder/'cases').glob('*/W.npy'):
        w = np.load(path); w[:, 4:] = np.tile(common, (32, 1)); np.save(path, w)
    np.savez(folder/'Y.npz', logit_nn=logit, p_nn=1/(1+np.exp(-logit)))


def test_polynomial_teacher_extracts_UX_only_and_trains_compression_on_all_rows(tmp_path):
    source = tmp_path/'step2'; shared_fixture(source)
    folder = tmp_path/'step5_matrix_metrics/joint_bank_seed391'
    with threadpool_limits(limits=2):
        joint_bank.prepare(source, folder, degrees=(4,), device='numpy')
        data = matrix_io.MatrixCases(source, ('logit_nn', 'p_nn'), 404)
        target, audit = teacher.polynomial(data)
        matrix = target.reshape(32, 5, 2)
        np.testing.assert_allclose(matrix.mean(0), 0., atol=1e-9)
        np.testing.assert_allclose(matrix.mean(1), 0., atol=1e-9)
        assert target.var() > 0
        rows = cp_runner.analyze(source, tmp_path/'compression', ranks=(1,), epochs=1, starts=1,
                                 batch=80, model_chunk=8, device='cpu', teacher='poly4_UX')
    assert rows and audit['all_original_rows_evaluated'] == 160


def test_kernel_teacher_shared_recorded_grid_has_no_U_only_or_X_only_means(tmp_path):
    source = tmp_path/'step2'; shared_fixture(source)
    folder = tmp_path/'step5_matrix_metrics/UX_complexity_source_seed383'
    with threadpool_limits(limits=2):
        prep.prepare(source, folder, estimators=('rbf2',), centers=12)
        data = matrix_io.MatrixCases(source, ('logit_nn', 'p_nn'), 404)
        target, audit = teacher.kernel(data)
    matrix = target.reshape(32, 5, 2)
    np.testing.assert_allclose(matrix.mean(0), 0., atol=1e-9)
    np.testing.assert_allclose(matrix.mean(1), 0., atol=1e-9)
    assert audit['shared_X_grid_verified'] is True
    assert audit['all_subjects'] == 32


def test_kernel_teacher_unshared_design_evaluates_original_heldout_rows(tmp_path):
    source = tmp_path/'step2'; make_cases(source)
    folder = tmp_path/'step5_matrix_metrics/UX_complexity_source_seed383'
    with threadpool_limits(limits=2):
        prep.prepare(source, folder, estimators=('rbf2',), centers=12)
        data = matrix_io.MatrixCases(source, ('logit_nn', 'p_nn'), 404)
        target, audit = teacher.kernel(data)
    assert target.shape == data.y.shape and np.isfinite(target).all()
    assert audit['shared_X_grid_verified'] is False
    assert audit['raw_validation_rows'] == 40
