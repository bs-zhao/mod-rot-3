import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_input_independence as ri


def test_joint_entropy_is_orthogonal_invariant_and_independent_uniform_axes_preferred():
    rng = np.random.default_rng(431); u = rng.uniform(-1., 1., size=(5000, 2))
    q = np.array([[1., -1.], [1., 1.]])/np.sqrt(2)
    rotated = u @ q
    assert ri.entropy(rotated, 10) == pytest.approx(ri.entropy(u, 10), abs=1e-12)
    before = sum(ri.entropy(u[:, j], 10) for j in range(2))
    after = sum(ri.entropy(rotated[:, j], 10) for j in range(2))
    assert after > before+.1
