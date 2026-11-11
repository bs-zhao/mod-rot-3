"""Synthetic tests of the whole Stage-2 sweep; no frozen checkpoint needed."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "run.py"
spec = importlib.util.spec_from_file_location("pairwise_step2", SCRIPT)
mod = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class TestPairwiseStep2(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.stage1 = self.root / "stage1_data"
        self.stage1.mkdir()
        self.stage2 = self.root / "step2_pairwise_sweep"
        rng = np.random.default_rng(2026)
        self.n, self.dz, self.dx = 9, 4, 3
        counts = np.array([2, 3, 1, 4, 2, 3, 1, 2, 4], dtype=np.int64)
        self.sid = np.repeat(np.arange(self.n), counts)
        self.tid = np.concatenate([np.arange(k) for k in counts])
        z_raw = rng.normal(size=(self.n, self.dz)) * np.array([1.0, .2, 3.2, 4.1])
        self.zstd = (z_raw - z_raw.mean(0)) / z_raw.std(0)
        self.r, _ = np.linalg.qr(rng.normal(size=(self.dz, self.dz)))
        self.x = rng.normal(size=(len(self.sid), self.dx)).astype(np.float32)
        self.w0 = np.column_stack([self.zstd[self.sid], self.x]).astype(np.float32)
        self.choice = rng.integers(0, 2, size=len(self.sid))
        np.save(self.stage1 / "Zstd.npy", self.zstd)
        np.save(self.stage1 / "R.npy", self.r)
        np.save(self.stage1 / "W.npy", self.w0)
        np.save(self.stage1 / "subject_index.npy", self.sid)
        np.save(self.stage1 / "trial_index.npy", self.tid)
        np.save(self.stage1 / "trial_counts.npy", counts)
        np.savez_compressed(
            self.stage1 / "Y.npz",
            choice_observed=self.choice,
            logit_nn=np.zeros(len(self.sid)),
            p_nn=np.full(len(self.sid), .5),
            choice_nn=np.zeros(len(self.sid)),
            logits_nn_2class=np.zeros((len(self.sid), 2)))
        (self.stage1 / "manifest.json").write_text(
            json.dumps({"model": "C1", "w_shape": list(self.w0.shape)}),
            encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_case_count_names_and_angle_grid(self):
        cases4 = mod.all_cases(4)
        cases6 = mod.all_cases(6)
        self.assertEqual(len(cases4), 37)
        self.assertEqual(len(cases6), 91)
        self.assertEqual(cases4[0].name, "baseline")
        self.assertEqual(cases4[1].name, "u1_u2_015")
        self.assertEqual(cases4[-1].name, "u3_u4_090")
        self.assertEqual(len({c.name for c in cases6}), 91)

    def test_plane_rotation_convention_and_orthogonality(self):
        case = mod.Case("u1_u3_090", (0, 2), 90)
        g = mod.givens(4, case.pair, case.angle)
        np.testing.assert_allclose(g @ g.T, np.eye(4), atol=1e-14)
        np.testing.assert_allclose(g @ np.array([1., 2., 3., 4.]),
                                   np.array([-3., 2., 1., 4.]), atol=1e-14)
        np.testing.assert_array_equal(mod.orientation(self.r, mod.all_cases(4)[0]),
                                      self.r)
        np.testing.assert_allclose(mod.orientation(self.r, case),
                                   g @ self.r)

    def test_full_sweep_all_W_and_single_shared_Y(self):
        result = mod.run_sweep(self.stage1, self.stage2, model="C1",
                               chunk_rows=5)
        self.assertTrue(result["complete"])
        self.assertEqual(result["n_cases_all"], 37)
        self.assertEqual(len(list((self.stage2 / "cases").glob("*/W.npy"))), 37)
        self.assertEqual(len(list(self.stage2.glob("**/Y.npz"))), 1)
        np.testing.assert_array_equal(np.load(self.stage2 / "Zstd.npy"), self.zstd)
        np.testing.assert_array_equal(np.load(self.stage2 / "R_best.npy"), self.r)
        self.assertEqual(mod.sha256(self.stage1 / "Y.npz"),
                         mod.sha256(self.stage2 / "Y.npz"))

        for name, pair, angle in [
            ("baseline", None, 0),
            ("u1_u2_015", (0, 1), 15),
            ("u1_u3_090", (0, 2), 90),
            ("u3_u4_075", (2, 3), 75),
        ]:
            case_folder = self.stage2 / "cases" / name
            R_case = np.load(case_folder / "R.npy")
            G = mod.givens(self.dz, pair, angle)
            np.testing.assert_allclose(
                np.load(case_folder / "G_relative.npy"), G, atol=1e-14)
            np.testing.assert_allclose(R_case, G @ self.r, atol=1e-14)
            np.testing.assert_allclose(R_case @ R_case.T, np.eye(self.dz), atol=1e-14)
            W_case = np.load(case_folder / "W.npy")
            U_subject = self.zstd @ R_case.T
            np.testing.assert_allclose(W_case[:, :self.dz],
                                       U_subject[self.sid], atol=5e-7)
            np.testing.assert_allclose(W_case[:, self.dz:], self.x, atol=0)
            self.assertFalse((case_folder / "Y.npz").exists())
            np.testing.assert_allclose(np.load(case_folder / "U_subject.npy"),
                                       U_subject, atol=1e-14)

        # Reusing existing valid outputs does not alter Y.
        again = mod.run_sweep(self.stage1, self.stage2, model="C1", chunk_rows=7)
        self.assertTrue(again["complete"])

    def test_dry_run_and_partial_case(self):
        info = mod.run_sweep(self.stage1, self.stage2, model="C1",
                             dry_run=True)
        self.assertEqual(info["n_cases_all"], 37)
        self.assertFalse(self.stage2.exists())
        partial = mod.run_sweep(
            self.stage1, self.stage2, model="C1",
            only_case="u1_u4_060", chunk_rows=8)
        self.assertFalse(partial["complete"])
        self.assertTrue((self.stage2 / "cases" / "u1_u4_060" / "W.npy").exists())
        self.assertFalse((self.stage2 / "cases" / "baseline" / "W.npy").exists())
        finished = mod.run_sweep(self.stage1, self.stage2, model="C1",
                                 chunk_rows=8)
        self.assertTrue(finished["complete"])

    def test_source_change_requires_explicit_overwrite(self):
        mod.run_sweep(self.stage1, self.stage2, model="C1",
                      only_case="baseline", chunk_rows=5)
        with np.load(self.stage1 / "Y.npz", allow_pickle=False) as f:
            payload = {key: f[key] for key in f.files}
        payload["p_nn"] = np.full(len(self.sid), .3)
        np.savez_compressed(self.stage1 / "Y.npz", **payload)
        with self.assertRaisesRegex(RuntimeError, "Step 1 arrays changed"):
            mod.run_sweep(self.stage1, self.stage2, model="C1", dry_run=False)

    def test_invalid_step1_trial_alignment_rejected(self):
        np.save(self.stage1 / "subject_index.npy", np.roll(self.sid, 1))
        with self.assertRaisesRegex(ValueError, "subject mapping"):
            mod.run_sweep(self.stage1, self.stage2, model="C1", dry_run=True)


if __name__ == "__main__":
    unittest.main()
