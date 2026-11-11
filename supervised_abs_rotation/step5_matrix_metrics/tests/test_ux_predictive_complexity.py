import sys
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ux_predictive_complexity as uc
import prepare_response_bank as prep
import run_ux_predictive_complexity as runner
from test_matrix_metrics import make_cases


def test_crossfit_X_omission_preserves_affine_complement_and_detects_sparse_UX():
    rng = np.random.default_rng(531); x = rng.normal(size=(100, 2)); u = rng.normal(size=(40, 2))
    y = np.array([u[:, 0, None]*x[None, :, 0]+u[:, 1, None]*x[None, :, 1]])
    bank = np.moveaxis(y, 0, 1)
    with threadpool_limits(limits=2):
        operators, audit = uc.omission_operators(x, x.std(0), bank, np.arange(30), folds=3, degrees=(2,), centers=8)
    for j, residual in enumerate(operators['omit_d2']):
        np.testing.assert_allclose(residual @ np.ones(len(x)), 0., atol=1e-9)
        np.testing.assert_allclose(residual @ x[:, 1-j], 0., atol=1e-8)
    powers = uc.wave.sur.powers_for(2, 1)
    coefficients = np.zeros((len(powers), 1, len(x))); coefficients[1, 0] = x[:, 0]; coefficients[2, 0] = x[:, 1]
    gram = uc.coefficient_grams(coefficients, operators)['omit_d2'][0]
    meta = dict(kind='poly', powers=powers)
    a, _ = uc.energies(meta, gram, np.zeros((1, 2)), 1., np.eye(2))
    q = np.array([[1., -1.], [1., 1.]])/np.sqrt(2)
    b, _ = uc.energies(meta, gram, np.zeros((1, 2)), 1., q)
    sa, sb = uc.sparsity(a['U_total']), uc.sparsity(b['U_total'])
    assert sa['pooled_UX_columns_mean_effective_support_p1'] == pytest.approx(1., abs=1e-6)
    assert sb['pooled_UX_columns_mean_effective_support_p1'] == pytest.approx(2., abs=1e-6)


def test_kernel_U_reference_variance_and_first_total_decomposition():
    rng = np.random.default_rng(532); centers = rng.normal(size=(7, 2)); anchor = rng.normal(size=2)
    q, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    meta = dict(kind='rbf', centers=centers, lengthscale=1.2)
    f, t, v = uc.u_axis_matrices(meta, q, anchor, .3)
    np.testing.assert_allclose(v, .5*(f[0]+f[1]+t[0]+t[1]), atol=1e-12)
    _, _, original = uc.u_axis_matrices(meta, np.eye(2), anchor, .3)
    np.testing.assert_allclose(v, original, atol=1e-12)


def test_complete_polynomial_and_kernel_UX_complexity_pipeline(tmp_path):
    import json
    source = tmp_path/'source'; make_cases(source)
    folder = tmp_path/'bank'; output = tmp_path/'out'
    with threadpool_limits(limits=2):
        prep.prepare(source, folder, estimators=('poly3', 'rbf1'), centers=12)
        rows = runner.analyze(source, output, folder, anchors_count=2, sigmas=(.2,))
    assert rows
    assert {r['estimator'] for r in rows} == {'poly3', 'rbf1'}
    assert json.loads((output/'protocol.json').read_text())['status'] == 'COMPLETE'
    audit = json.loads((output/'read_audit.json').read_text())
    assert audit['all_subjects'] == 32
    assert audit['all_original_rows_in_source_fit'] == 160
    assert audit['forbidden_files_read'] == []
