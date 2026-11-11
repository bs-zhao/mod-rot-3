import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import global_output_link as link
import run_global_output_link as runner
import joint_bank
import joint_fields
from test_matrix_metrics import make_cases


def test_global_chain_rule_corrected_UX_is_output_link_and_U_rotation_equivariant():
    rng = np.random.default_rng(591); n, d, dx = 100, 4, 3
    y = rng.normal(size=n); g = rng.normal(size=(n, d+dx)); h = rng.normal(size=(n, dx, d)); k = rng.normal(size=n)
    a, b = np.exp(.3*y), .3*np.exp(.3*y)
    transformed_g = a[:, None]*g
    outer = g[:, None, :d]*g[:, d:, None]
    transformed_h = a[:, None, None]*h+b[:, None, None]*outer
    transformed_k = k/a-b/a**2
    r = link.corrected(h, g, d, k)
    rr = link.corrected(transformed_h, transformed_g, d, transformed_k)
    np.testing.assert_allclose(link.unit_columns(r), link.unit_columns(rr), atol=1e-12)
    q, _ = np.linalg.qr(rng.normal(size=(d, d))); rotated_g = g.copy(); rotated_g[:, :d] = g[:, :d] @ q
    np.testing.assert_allclose(link.corrected(h @ q, rotated_g, d, k), r @ q, atol=1e-12)


def test_conditional_link_is_a_single_function_of_output_and_integrates_monotonically():
    rng = np.random.default_rng(592); y = np.repeat(np.linspace(-2., 2., 30), 80)
    denominator = rng.uniform(.5, 2., len(y)); expected = .2*np.sin(y)
    numerator = -expected*denominator
    fitted = link.fit_link(y, numerator, denominator, bins=30)
    np.testing.assert_allclose(link.evaluate(fitted, y), expected, atol=1e-12)
    audit = link.link_audit(fitted, -2., 2.)
    assert audit['global_link_integrable'] and audit['global_link_monotone']
    assert audit['numerical_derivative_dynamic_range_representable']
    derivative = np.exp(audit['log_derivative'])
    assert np.all(derivative > 0.)


def test_global_link_pipeline_covers_all_rows_and_exact_derivative_cache(tmp_path):
    import json
    source = tmp_path/'source'; make_cases(source)
    folder, output = tmp_path/'bank', tmp_path/'out'
    with threadpool_limits(limits=2):
        joint_bank.prepare(source, folder, degrees=(3,), device='numpy')
        rows = runner.analyze(source, output, [folder], degrees=(3,), bins=(0, 4, 8))
        bank = joint_bank.JointBank(source, folder); saved = bank.load(3)
        z = (bank.data.w-saved['mean'])/saved['scale']
        g, h = joint_fields.evaluate(z, saved['powers'], saved['weights'], saved['coef'])
    np.testing.assert_allclose(np.load(output/'poly3/gradient.npy'), g, atol=1e-10)
    np.testing.assert_allclose(np.load(output/'poly3/mixed.npy'), h[:, :, :4, 4:].transpose(0, 1, 3, 2), atol=1e-10)
    assert rows and any(not r['is_control'] for r in rows)
    audit = json.loads((output/'read_audit.json').read_text())
    assert audit['all_subjects_in_link_refit_and_moments'] == 32
    assert audit['all_original_rows_in_source_fit_and_moments'] == 160
    assert audit['forbidden_files_read'] == []
