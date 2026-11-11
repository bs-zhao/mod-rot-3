"""Tests for standalone supervised absolute rotation (no real data required)."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "run.py"
spec = importlib.util.spec_from_file_location("supervised_abs_run", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestFullSampleOracle(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(91)
        self.z = rng.normal(size=(850, 4))
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        self.t = (self.z - self.z.mean(0)) @ q.T * np.array([.5, -1.5, 2., .8])
        self.t += rng.normal(scale=.005, size=self.t.shape)
        self.zc, self.ts, self.zmean, self.tmean, self.tsd = module.prepare_all(
            self.z, self.t)

    def test_all_subject_normalization(self):
        np.testing.assert_allclose(self.zc.mean(0), 0, atol=1e-13)
        np.testing.assert_allclose(self.ts.mean(0), 0, atol=1e-13)
        np.testing.assert_allclose(self.ts.std(0), 1, atol=1e-13)
        np.testing.assert_allclose(self.zc, self.z - self.z.mean(0))

    def test_absolute_profile_formula(self):
        rng = np.random.default_rng(17)
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        loss, s = module.profile(self.zc, self.ts, q, "abs_mse")
        v = self.zc @ q.T
        brute = np.mean((np.abs(v) * s - np.abs(self.ts)) ** 2)
        self.assertAlmostEqual(loss, brute, places=12)
        self.assertTrue((s >= 0).all())
        for j in range(4):
            self.assertAlmostEqual(
                s[j], np.dot(np.abs(v[:, j]), np.abs(self.ts[:, j]))
                / np.dot(v[:, j], v[:, j]), places=12)

    def test_signed_profile_formula(self):
        q = np.eye(4)
        loss, s = module.profile(self.zc, self.ts, q, "signed_mse")
        v = self.zc @ q.T
        self.assertAlmostEqual(loss, np.mean((v * s - self.ts)**2), places=12)

    def test_recovered_rotation_orthogonal_and_prediction(self):
        result = module.fit(self.zc, self.ts, "abs_mse",
                            seed=31, restarts=1, maxiter=120)
        self.assertLess(result["loss"], 0.01)
        np.testing.assert_allclose(result["R"] @ result["R"].T, np.eye(4), atol=1e-8)
        np.testing.assert_allclose(result["prediction"], self.zc @ result["R"].T @ result["D"])
        np.testing.assert_allclose(result["DR"], result["D"] @ result["R"])
        self.assertAlmostEqual(
            result["loss"], np.mean((np.abs(result["prediction"]) - np.abs(self.ts))**2),
            places=10)

    def test_full_sample_writes_and_non_destructive(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "test_run"
            report = module.analyze(
                self.z, self.t, model="C1", output=out, objective="abs_mse",
                plots=False, restarts=0, maxiter=80)
            self.assertEqual(report["n_subjects"], len(self.z))
            self.assertEqual(report["sample_policy"], "ALL subjects supervised; no holdout")
            self.assertTrue((out / "R.csv").is_file())
            self.assertTrue((out / "D.csv").is_file())
            self.assertTrue((out / "DR.csv").is_file())
            self.assertTrue((out / "per_parameter.csv").is_file())
            self.assertTrue((out / "corr__prediction_Tstd_magnitude.csv").is_file())
            with np.load(out / "oracle_full_sample.npz", allow_pickle=False) as f:
                self.assertEqual(f["Z_centered"].shape, self.z.shape)
                self.assertEqual(f["prediction_Tstd"].shape, self.t.shape)
                self.assertEqual(len(f["full_sample_indices"]), len(self.z))
            with self.assertRaises(FileExistsError):
                module.analyze(self.z, self.t, model="C1", output=out,
                               plots=False, restarts=0, maxiter=20)

    def test_bad_geometry_rejected(self):
        with self.assertRaises(ValueError):
            module.prepare_all(self.z, self.t[:, :3])
        with self.assertRaises(ValueError):
            module.prepare_all(np.zeros_like(self.z), self.t)


if __name__ == "__main__":
    unittest.main()
