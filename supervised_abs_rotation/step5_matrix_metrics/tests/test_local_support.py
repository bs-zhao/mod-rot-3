import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import local_support as ls
import tail_support as ts


def test_vectorized_local_support_matches_individual_definitions():
    rng = np.random.default_rng(441)
    e = np.r_[np.zeros((1, 8)), rng.lognormal(size=(40, 8)), [[4., 4., .0001, .0001, 0, 0, 0, 0]]]
    actual = ls.row_scores(e)
    for i, row in enumerate(e):
        expected = ts.scores(row)
        for key in actual:
            assert actual[key][i] == pytest.approx(expected[key], abs=1e-9)


def test_local_support_detects_switching_sparse_patterns_hidden_by_global_pooling():
    energy = np.zeros((12, 4, 4))
    for i in range(12):
        energy[i, :, i % 4] = 1.
    score = ls.pooled_scores(energy)
    assert score['point_mean_effective_support_p1'] == pytest.approx(1.)
    assert score['context_mean_effective_support_p1'] == pytest.approx(4.)


def test_local_fields_are_scale_and_signed_permutation_invariant():
    rng = np.random.default_rng(442); d = 4
    g = rng.normal(size=(20, 8, d)); h = rng.normal(size=(20, 8, d, d)); h = (h+h.swapaxes(-1, -2))/2
    delta = rng.normal(size=(20, 11, d)); q = np.eye(d)[:, [2, 3, 0, 1]]*[-1, 1, 1, -1]
    a = ls.evaluate(g, h, delta, np.eye(d)); b = ls.evaluate(g*30, h*30, delta*30, q)
    for key in a:
        if not key.startswith('control_'):
            assert a[key] == pytest.approx(b[key], abs=1e-9)
