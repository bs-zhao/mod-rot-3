import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import local_ux_complexity as local


def test_each_reference_and_X_column_scale_cannot_change_UX_support():
    rng = np.random.default_rng(541); energy = rng.lognormal(size=(8, 4, 6))
    scale = 10.**rng.uniform(-100, 100, size=(8, 1, 6))
    a, b = local.scores(energy), local.scores(energy*scale)
    for key in a:
        assert a[key] == pytest.approx(b[key], abs=1e-10)


def test_user_sparsity_vectors_stay_ordered_after_reference_pooling():
    a = np.broadcast_to(np.array([2., 2., .01, .01])[None, :, None]**2, (8, 4, 6)).copy()
    b = np.broadcast_to(np.array([1., .5, .1, .1])[None, :, None]**2, (8, 4, 6)).copy()
    x, y = local.scores(a), local.scores(b)
    for key in x:
        assert x[key] < y[key]
