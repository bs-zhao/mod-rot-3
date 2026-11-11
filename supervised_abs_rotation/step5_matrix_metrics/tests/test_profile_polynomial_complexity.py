import sys
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import profile_polynomial_complexity as profile
import run_profile_polynomial_complexity as runner
import prepare_response_bank as prep
from test_matrix_metrics import make_cases


def test_group_OLS_matches_direct_arbitrary_multiresponse_fits_with_correlated_dictionary():
    rng = np.random.default_rng(581)
    a = rng.normal(size=(200, 16)); a[:, 3:] += .4*a[:, :1]
    a -= a.mean(0)
    y = a @ rng.normal(size=(16, 11))+rng.normal(size=(200, 11))*.1
    gram, cross = a.T @ a/len(a), a.T @ y/len(a)
    energy = float(np.mean(y*y)*y.shape[1])
    with threadpool_limits(limits=2):
        curve, audit = profile.compress(gram, cross @ cross.T, energy, maximum=16)
    for k in (2, 8, 16):
        design = a[:, audit['selected_term_indices'][:k]]
        prediction = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        direct = np.sum((y-prediction)**2)/np.sum(y*y)
        assert curve[k] == pytest.approx(direct, abs=1e-10)
    assert np.all(np.diff(curve) <= 1e-12)


def test_profile_support_respects_user_example_and_extreme_dictionary_units():
    def fit(amplitude, scales):
        gram = np.diag(scales**2)
        cross = np.diag(scales*np.asarray(amplitude))
        energy = float(np.sum(np.asarray(amplitude)**2))
        return profile.compress(gram, cross @ cross.T, energy, maximum=4)[0]
    a = fit([2., 2., .01, .01], np.ones(4))
    b = fit([1., .5, .1, .1], np.ones(4))
    changed = fit([2., 2., .01, .01], np.array([1e-80, 1e80, 1e-40, 1e40]))
    np.testing.assert_allclose(a, changed, atol=1e-12)
    sa, sb = profile.scores([a]*6, maximum=4), profile.scores([b]*6, maximum=4)
    assert all(sa[key] <= sb[key]+1e-12 for key in sa)
    assert sa['UX_U_profile_terms_tail2_mean_l2'] < sb['UX_U_profile_terms_tail2_mean_l2']


def test_full_profile_projection_and_compression_pipeline_uses_only_case_matrices(tmp_path):
    import json
    source = tmp_path/'source'; make_cases(source)
    folder, output = tmp_path/'bank', tmp_path/'out'
    with threadpool_limits(limits=2):
        prep.prepare(source, folder, estimators=('poly3',), centers=10)
        rows = runner.analyze(source, output, folder, estimators=('poly3',), degrees=(3,), maximum=16)
    assert rows
    assert json.loads((output/'protocol.json').read_text())['status'] == 'COMPLETE'
    audit = json.loads((output/'read_audit.json').read_text())
    assert audit['all_subjects_in_compression'] == 32
    assert audit['all_original_rows_in_source_fit'] == 160
    assert audit['forbidden_files_read'] == []
