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
        self.zstd, self.ts, self.zmean, self.zsd, self.tmean, self.tsd = module.prepare_all(
            self.z, self.t)

    def test_all_subject_normalization(self):
        np.testing.assert_allclose(self.zstd.mean(0), 0, atol=1e-13)
        np.testing.assert_allclose(self.ts.mean(0), 0, atol=1e-13)
        np.testing.assert_allclose(self.ts.std(0), 1, atol=1e-13)
        np.testing.assert_allclose(self.zstd.mean(0), np.zeros(4), atol=1e-13)
        np.testing.assert_allclose(self.zstd.std(0), np.ones(4), atol=1e-13)
        np.testing.assert_allclose(self.zstd, (self.z - self.z.mean(0)) / self.z.std(0))
        np.testing.assert_allclose(self.zsd, self.z.std(0))
        np.testing.assert_allclose(self.tsd, self.t.std(0))

    def test_unequal_Z_scales_still_start_at_zscore(self):
        raw = self.z * np.array([0.02, 2.5, 7.0, 0.3])[None, :]
        rescaled, tstd, mean, sd, t_mean, t_sd = module.prepare_all(raw, self.t)
        np.testing.assert_allclose(rescaled, self.zstd, atol=1e-12)
        np.testing.assert_allclose(rescaled.std(axis=0), np.ones(4), atol=1e-12)
        np.testing.assert_allclose(sd, raw.std(axis=0), atol=1e-12)
        np.testing.assert_allclose(tstd, self.ts)
        np.testing.assert_allclose(t_mean, self.tmean)
        np.testing.assert_allclose(t_sd, self.tsd)

    def test_absolute_profile_formula(self):
        rng = np.random.default_rng(17)
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        loss, s = module.profile(self.zstd, self.ts, q, "abs_mse")
        v = self.zstd @ q.T
        brute = np.mean((np.abs(v) * s - np.abs(self.ts)) ** 2)
        self.assertAlmostEqual(loss, brute, places=12)
        self.assertTrue((s >= 0).all())
        for j in range(4):
            self.assertAlmostEqual(
                s[j], np.dot(np.abs(v[:, j]), np.abs(self.ts[:, j]))
                / np.dot(v[:, j], v[:, j]), places=12)

    def test_signed_profile_formula(self):
        q = np.eye(4)
        loss, s = module.profile(self.zstd, self.ts, q, "signed_mse")
        v = self.zstd @ q.T
        self.assertAlmostEqual(loss, np.mean((v * s - self.ts)**2), places=12)

    def test_recovered_rotation_orthogonal_and_prediction(self):
        result = module.fit(self.zstd, self.ts, "abs_mse",
                            seed=31, restarts=1, maxiter=120)
        self.assertLess(result["loss"], 0.01)
        np.testing.assert_allclose(result["R"] @ result["R"].T, np.eye(4), atol=1e-8)
        np.testing.assert_allclose(result["prediction"], self.zstd @ result["R"].T @ result["D"])
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
            self.assertTrue((out / "A_rawZ_to_Tstd.csv").is_file())
            self.assertTrue((out / "Z_zscore_stats.csv").is_file())
            self.assertTrue((out / "per_parameter.csv").is_file())
            self.assertTrue((out / "corr__prediction_Tstd_magnitude.csv").is_file())
            with np.load(out / "oracle_full_sample.npz", allow_pickle=False) as f:
                self.assertEqual(f["Z_centered"].shape, self.z.shape)
                np.testing.assert_allclose(f["Z_centered"], self.z - self.z.mean(0))
                np.testing.assert_allclose(f["Z_std"], (self.z - self.z.mean(0)) / self.z.std(0))
                np.testing.assert_allclose(f["z_std_all"], self.z.std(0))
                np.testing.assert_allclose(f["prediction_Tstd"], self.z @ f["A_rawZ_to_Tstd"].T
                                           + f["b_rawZ_to_Tstd"][None, :], atol=1e-9)
                np.testing.assert_allclose(f["prediction_Tstd"], f["Z_std"] @ f["DR"].T, atol=1e-9)
                self.assertEqual(f["prediction_Tstd"].shape, self.t.shape)
                self.assertEqual(len(f["full_sample_indices"]), len(self.z))
            with self.assertRaises(FileExistsError):
                module.analyze(self.z, self.t, model="C1", output=out,
                               plots=False, restarts=0, maxiter=20)

    def test_png_save_in_memory_and_short_name_fallback(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "simulated_bad_name.png"
            original = Path.write_bytes

            def simulated_write(path, payload):
                if path.name == "simulated_bad_name.png":
                    raise OSError(22, "Simulated Windows Invalid argument")
                return original(path, payload)

            with patch.object(Path, "write_bytes", simulated_write):
                actual, message = module.plot_heatmap(
                    output, np.eye(4), ["u1", "u2", "u3", "u4"],
                    ["t1", "t2", "t3", "t4"], "Fallback test")
            self.assertTrue(actual.is_file())
            self.assertNotEqual(actual, output)
            self.assertTrue(actual.name.startswith("fig_"))
            self.assertIn("Invalid argument", message)

    def test_plot_io_failure_keeps_numerical_results(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "failed_plots"
            with patch.object(module, "plot_heatmap",
                              side_effect=OSError(22, "Simulated PNG disk error")):
                summary = module.analyze(
                    self.z, self.t, model="C1", output=out,
                    plots=True, restarts=0, maxiter=50)
            self.assertTrue((out / "summary.json").is_file())
            self.assertTrue((out / "oracle_full_sample.npz").is_file())
            self.assertTrue((out / "R.csv").is_file())
            self.assertTrue((out / "corr__U_whitened_Tstd.csv").is_file())
            self.assertIn("corr__U_whitened_Tstd.png", summary["plot_errors"])

    def test_zca_whitening_full_rank_and_correlations(self):
        result = module.fit(self.zstd, self.ts, "abs_mse",
                            seed=31, restarts=0, maxiter=60)
        u = result["U"]
        w = module.zca_whiten_all(u)
        uw = w["U_whitened"]
        self.assertEqual(w["effective_rank"], 4)
        np.testing.assert_allclose(uw.mean(axis=0), np.zeros(4), atol=1e-10)
        np.testing.assert_allclose(uw.T @ uw / len(uw), np.eye(4), atol=1e-8)
        np.testing.assert_allclose(
            uw, (u - w["U_mean_all"]) @ w["whitening_matrix"], atol=1e-10)
        np.testing.assert_allclose(
            w["whitening_matrix"] @ w["U_cov_all"] @ w["whitening_matrix"],
            np.eye(4), atol=1e-8)
        self.assertEqual(module.corr(uw, self.ts).shape, (4, 4))

    def test_zca_rank_deficient_graceful(self):
        u = self.zstd.copy()
        u[:, 3] = u[:, 0] + u[:, 1]  # rank <= 3
        result = module.zca_whiten_all(u)
        self.assertEqual(result["effective_rank"], 3)
        covariance = result["U_whitened_cov_all"]
        np.testing.assert_allclose(covariance @ covariance, covariance, atol=1e-7)
        self.assertTrue(np.isfinite(result["U_whitened"]).all())

    def test_bad_geometry_rejected(self):
        with self.assertRaises(ValueError):
            module.prepare_all(self.z, self.t[:, :3])
        with self.assertRaises(ValueError):
            module.prepare_all(np.zeros_like(self.z), self.t)


if __name__ == "__main__":
    unittest.main()
