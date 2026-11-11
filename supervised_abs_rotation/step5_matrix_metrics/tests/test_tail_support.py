import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tail_support as ts


def test_users_effective_support_example():
    a = ts.scores(np.array([2., 2., .01, .01])**2)
    b = ts.scores(np.array([1., .5, .1, .1])**2)
    for key in a:
        assert a[key] < b[key], key
    assert a["tail2_l2"] == pytest.approx(.0049999375)
    assert b["tail2_l2"] == pytest.approx(np.sqrt(.02/1.27))


def test_overall_rescaling_cannot_change_support():
    energy = np.array([4., 4., .0001, .0001])
    a = ts.scores(energy)
    for scale in (1e-12, .01, 100., 1e12):
        b = ts.scores(energy*scale**2)
        for key in a:
            assert b[key] == pytest.approx(a[key], rel=1e-12)


@pytest.mark.parametrize("pooling", ["global", "context", "point"])
def test_covariance_rotation_matches_full_field(pooling):
    rng = np.random.default_rng(80)
    h = rng.normal(size=(80, 12, 4, 4)); h = (h+h.swapaxes(-1, -2))/2
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    moment = ts.moment_covariance(h, pooling)
    expected = ts.uu_energy(moment, q)
    rotated = np.einsum("ij,nsjk,kl->nsil", q.T, h, q)
    direct = ts.moment_covariance(rotated, pooling).diagonal().reshape(4, 4)
    np.testing.assert_allclose(expected, direct, atol=1e-12)
