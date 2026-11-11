import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import equal_x_output_link as equal
import global_output_link as link
import run_global_output_link as preparation
import run_equal_x_output_link as runner
import joint_bank
from test_matrix_metrics import make_cases


def test_balanced_global_link_is_invariant_to_independent_X_column_strength_units():
    rng = np.random.default_rng(611); n, d, dx = 600, 3, 3
    y = rng.normal(size=(n, 1)); g = rng.normal(size=(n, 1, d+dx))
    outer = g[:, :, None, :d]*g[:, :, d:, None]
    h = -.3*np.sin(y)[..., None, None]*outer+.05*rng.normal(size=outer.shape)
    train = np.arange(n) % 4 != 0
    original = equal.column_invariants(g, h, d)
    scales = np.array([1e-40, 1., 1e40]); changed_g = g.copy(); changed_g[..., d:] *= scales
    changed_h = h*scales[None, None, :, None]
    changed = equal.column_invariants(changed_g, changed_h, d)
    a, selected_a, _ = equal.fit_all(y, *original, train, bins=(0, 8, 16), reference_bins=8)
    b, selected_b, _ = equal.fit_all(y, *changed, train, bins=(0, 8, 16), reference_bins=8)
    assert selected_a == selected_b
    for count in a[0]:
        np.testing.assert_allclose(link.evaluate(a[0][count], y[:, 0]), link.evaluate(b[0][count], y[:, 0]), atol=1e-12)


def test_histogram_response_link_integrates_exactly_across_every_interval():
    fitted = dict(kind='histogram', edges=[-2., -1., 0., 2.], k=[1., -2., .5], bins=3)
    np.testing.assert_array_equal(link.evaluate(fitted, np.array([-1.5, -.5, 1.])), [1., -2., .5])
    audit = link.link_audit(fitted, -2., 2.)
    actual = np.asarray(audit['log_derivative'])
    np.testing.assert_allclose(np.diff(actual), [1., -2., 1.])
    assert np.all(np.exp(actual) > 0.)


def test_balanced_global_link_reuses_verified_WY_fields_and_all_recorded_rows(tmp_path):
    import json
    source = tmp_path/'step2'; make_cases(source)
    folder = tmp_path/'step5_matrix_metrics/joint_bank_seed391'
    fields, output = tmp_path/'fields', tmp_path/'out'
    with threadpool_limits(limits=2):
        joint_bank.prepare(source, folder, degrees=(4,), device='numpy')
        preparation.analyze(source, fields, [folder], degrees=(4,), bins=(0, 4, 8))
        rows = runner.analyze(fields, output, bins=(0, 4, 8), reference_bins=4)
    assert rows and any(not r['is_control'] for r in rows)
    audit = json.loads((output/'read_audit.json').read_text())
    assert audit['all_subjects'] == 32
    assert audit['all_original_rows_in_source_fit_and_moments'] == 160
    assert len(audit['derived_field_files_sha256']) == 3
    assert audit['forbidden_files_read'] == []
