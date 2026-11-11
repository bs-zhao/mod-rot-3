import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import all_trial_moments as am
import joint_fields as jf
import local_joint as lj
import surrogates as sur


def test_every_trial_moments_match_direct_rotated_fields_and_ux_column_rule():
    rng = np.random.default_rng(481); d, dx, n, m = 3, 2, 23, 13
    w = rng.normal(size=(n*m, d+dx)); powers = sur.powers_for(d+dx, 4); weights = sur.weights_for(powers)
    saved = dict(powers=powers, weights=weights, coef=rng.normal(size=(len(powers), 2)),
                 mean=np.zeros(d+dx), scale=np.ones(d+dx))
    data = SimpleNamespace(w=w, offsets=np.arange(n+1)*m, n=n, m=n*m, d=d, dx=dx)
    moments = am.moments(data, saved, subjects_per_chunk=7)
    g, h = jf.evaluate(w, powers, weights, saved['coef'])
    g, h = g.reshape(n, m, 2, d+dx), h.reshape(n, m, 2, d+dx, d+dx)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    for c in range(2):
        actual = am.scores(moments['UU'][:, c], moments['UX'][:, c], moments['gradient'][:, c], q)
        expected = lj.evaluate(g[:, :, c], h[:, :, c], q, d)
        for key in actual:
            # Total-median here is the median of each subject's total support;
            # R summed the separate family medians. Explicit different statistic.
            if key.startswith('UU_UX_subject_total_'):
                continue
            assert actual[key] == pytest.approx(expected[key], abs=1e-8)
    assert int(moments['rows_used']) == n*m


def test_variable_trial_counts_keep_all_rows_and_equal_subject_aggregation():
    rng = np.random.default_rng(482); counts = np.array([3, 9, 5, 17]); n = len(counts)
    offsets = np.r_[0, np.cumsum(counts)]; w = rng.normal(size=(offsets[-1], 4))
    powers = sur.powers_for(4, 2); weights = sur.weights_for(powers)
    saved = dict(powers=powers, weights=weights, coef=rng.normal(size=(len(powers), 1)), mean=np.zeros(4), scale=np.ones(4))
    data = SimpleNamespace(w=w, offsets=offsets, n=n, m=len(w), d=2, dx=2)
    moments = am.moments(data, saved, subjects_per_chunk=3)
    assert int(moments['rows_used']) == len(w)
    np.testing.assert_array_equal(moments['trial_counts'], counts)
    # Degree-two mixed derivatives are constant at all observed rows.
    np.testing.assert_allclose(moments['UX'], np.repeat(moments['UX'][:1], n, axis=0), atol=1e-10)
