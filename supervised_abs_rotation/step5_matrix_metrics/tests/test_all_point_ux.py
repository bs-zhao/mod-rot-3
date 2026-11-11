import sys
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import all_point_ux as points
import joint_bank
import run_all_point_ux as runner
import joint_fields
from test_matrix_metrics import make_cases


def test_point_UX_support_has_extreme_column_scale_invariance_and_user_ordering():
    a = np.tile(np.array([2., 2., .01, .01]), (50, 2, 6, 1))
    b = np.tile(np.array([1., .5, .1, .1]), (50, 2, 6, 1))
    sa, sb = points.point_scores(a, np.eye(4)), points.point_scores(b, np.eye(4))
    changed = points.point_scores(a*np.geomspace(1e-200, 1e200, 6)[None, None, :, None], np.eye(4))
    for key in sa:
        np.testing.assert_allclose(changed[key], sa[key], rtol=1e-12)
        assert np.all(sa[key] < sb[key])


def test_normalizing_each_point_precedes_trial_pooling_with_unequal_subject_counts():
    field = np.zeros((7, 1, 2, 4)); field[:3, :, :, :2] = 1.; field[3:] = 1.
    offsets = np.array([0, 3, 7])
    statistics, keys = points.subject_statistics(field, offsets, np.eye(4), subjects_per_chunk=1)
    index = keys.index('effective_support_p1')
    np.testing.assert_allclose(statistics[0, :, :, :, index], 2.)
    np.testing.assert_allclose(statistics[1, :, :, :, index], 4.)
    assert points.scores(statistics, keys, 0)['UX_columns_Xmean_trialmean_subjectmean_effective_support_p1'] == pytest.approx(3.)


def test_full_point_pipeline_exact_field_and_all_recorded_rows(tmp_path):
    import json
    source = tmp_path/'source'; make_cases(source)
    folder, output = tmp_path/'bank', tmp_path/'out'
    with threadpool_limits(limits=2):
        joint_bank.prepare(source, folder, degrees=(3,), device='numpy')
        bank = joint_bank.JointBank(source, folder); saved = bank.load(3)
        rows = runner.analyze(source, output, [folder], degrees=(3,))
        _, h = joint_fields.evaluate((bank.data.w-saved['mean'])/saved['scale'], saved['powers'], saved['weights'], saved['coef'])
    actual = np.load(output/'UX_field_poly3.npy')
    np.testing.assert_allclose(actual, h[:, :, :4, 4:].transpose(0, 1, 3, 2), atol=1e-10)
    assert rows
    audit = json.loads((output/'read_audit.json').read_text())
    assert audit['all_original_rows_in_source_fit_and_point_scores'] == 160
    assert audit['all_subjects'] == 32 and audit['forbidden_files_read'] == []
