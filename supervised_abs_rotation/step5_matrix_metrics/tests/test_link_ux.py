import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import link_ux as lu


def test_profiled_local_fields_are_equivariant_and_invariant_to_unknown_output_link():
    rng = np.random.default_rng(491); d = 4; dimension = 7
    g = rng.normal(size=(70, 2, dimension)); h = rng.normal(size=(70, 2, dimension, dimension)); h = (h+h.swapaxes(-1, -2))/2
    original = lu.fields(g, h[..., :d, :], d)
    q, _ = np.linalg.qr(rng.normal(size=(d, d))); full = np.eye(dimension); full[:d, :d] = q
    gr = g @ full; hr = full.T @ h @ full
    rotated = lu.fields(gr, hr[..., :d, :], d)
    for key in original:
        np.testing.assert_allclose(rotated[key], q.T @ original[key], atol=1e-10)
    first = rng.uniform(.2, 3., size=(70, 2)); second = rng.normal(size=(70, 2))
    transformed_g = first[..., None]*g
    transformed_h = first[..., None, None]*h+second[..., None, None]*g[..., :, None]*g[..., None, :]
    linked = lu.fields(transformed_g, transformed_h[..., :d, :], d)
    for key in original:
        if not key.startswith('raw_'):
            np.testing.assert_allclose(linked[key], original[key], atol=1e-9)


def test_affine_scaling_and_numerical_zero_residual_are_handled_without_false_support():
    rng = np.random.default_rng(492); g = rng.normal(size=(30, 2, 6)); h = rng.normal(size=(30, 2, 4, 6))
    original = lu.fields(g, h, 4)
    for scale in (1e-120, 1e120):
        changed = lu.fields(g*scale, h*scale, 4)
        for key in original:
            np.testing.assert_allclose(changed[key], original[key], atol=1e-9)
    rank_one = g[..., :4, None]*g[..., None, :]
    zero = lu.fields(g, rank_one, 4)
    for key in zero:
        if not key.startswith('raw_'):
            np.testing.assert_allclose(zero[key], 0., atol=1e-12)
