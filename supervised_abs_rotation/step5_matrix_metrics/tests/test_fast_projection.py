import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fast_projection as fp
import surrogates as sur


def test_recurrent_features_preserve_isotropic_kernel():
    rng = np.random.default_rng(100)
    w = rng.normal(size=(160, 7)); scaler = sur.Scaler(w, 4)
    for degree in (2, 3, 4):
        fast = fp.Polynomial(scaler, degree)
        original = sur.Polynomial(scaler, degree)
        np.testing.assert_allclose(fast.features(w), original.features(w), atol=1e-12, rtol=1e-12)


def test_double_precision_backend_matches_cpu_all_row_fit():
    pytest.importorskip("torch")
    rng = np.random.default_rng(101)
    w = rng.normal(size=(600, 5)); sid = np.repeat(np.arange(40), 15)
    y = np.column_stack([w[:, 0]*w[:, 3] + .2*w[:, 1]**2, np.sin(w[:, 4])])
    scaler = sur.Scaler(w, 3, row_mask=sid < 30)
    model = fp.Polynomial(scaler, 3)
    a, qa = fp.fit(model, w, y, sid, np.arange(30), chunk=71, device="numpy")
    b, qb = fp.fit(model, w, y, sid, np.arange(30), chunk=71, device="cpu")
    np.testing.assert_allclose(model.features(w) @ a, model.features(w) @ b, atol=1e-10)
    np.testing.assert_allclose(qa["validation_nmse"], qb["validation_nmse"], atol=1e-10)
    assert qb["final_fit_rows"] == 600
    assert qb["final_fit_subjects"] == 40


def test_cuda_stats_match_numpy_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device")
    rng = np.random.default_rng(102)
    w = rng.normal(size=(700, 6)); sid = np.repeat(np.arange(50), 14)
    y = (w[:, 0]*w[:, 4]+np.sin(w[:, 5]))[:, None]
    model = fp.Polynomial(sur.Scaler(w, 4, row_mask=sid < 38), 3)
    a, qa = fp.fit(model, w, y, sid, np.arange(38), chunk=83, device="numpy")
    b, qb = fp.fit(model, w, y, sid, np.arange(38), chunk=83, device="cuda")
    np.testing.assert_allclose(model.features(w) @ a, model.features(w) @ b, atol=1e-9)
    np.testing.assert_allclose(qa["validation_nmse"], qb["validation_nmse"], atol=1e-9)
