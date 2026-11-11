import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_ux_column as validation
import joint_fields as jf
import surrogates as sur


def test_recomputed_more_contexts_are_actual_joint_mixed_derivatives():
    rng = np.random.default_rng(471); w = rng.normal(size=(160, 6))
    data = SimpleNamespace(w=w, offsets=np.arange(0, 161, 5), n=32, d=4, dx=2)
    powers = sur.powers_for(6, 4); weights = sur.weights_for(powers)
    saved = dict(powers=powers, weights=weights, coef=rng.normal(size=(len(powers), 2)),
                 mean=np.zeros(6), scale=np.ones(6))
    actual, ids = validation.hux_fields(data, saved, 2)
    _, h = jf.evaluate(w[ids], powers, weights, saved['coef'])
    expected = h[:, 0, :4, 4:].reshape(32, 2, 4, 2)
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_paired_subject_bootstrap_preserves_uniform_known_margin():
    rng = np.random.default_rng(472); base = rng.uniform(1., 4., size=(53, 3))
    result = validation.bootstrap_scores(dict(baseline=base, a=base+.2, b=base+.4),
                                         'baseline', ['a', 'b'], 60, 473)
    assert result['simultaneous_minimum_margin_quantiles']['0.05'] == pytest.approx(.2)
    assert result['fraction_all_interior_margins_positive'] == 1.
