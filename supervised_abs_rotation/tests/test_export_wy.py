"""Contract tests for the exact full-trial W/Y export; no real S1/S3/S5 needed."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "export_wy.py"
spec = importlib.util.spec_from_file_location("supervised_abs_export_wy", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestStage1DataExport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(808)
        self.n, self.dz, self.dx = 6, 4, 2
        self.zraw = rng.normal(size=(self.n, self.dz)) * [0.3, 1.4, 4.5, 0.9]
        self.zstd = (self.zraw - self.zraw.mean(0)) / self.zraw.std(0, ddof=0)
        self.r, _ = np.linalg.qr(rng.normal(size=(self.dz, self.dz)))
        self.counts = np.array([2, 3, 1, 4, 2, 3], dtype=np.int64)
        self.sid = np.repeat(np.arange(self.n), self.counts)
        self.tid = np.concatenate([np.arange(c) for c in self.counts])
        self.x = rng.normal(size=(len(self.sid), self.dx)).astype(np.float32)
        self.choice = rng.integers(0, 2, size=len(self.sid), dtype=np.int64)
        self.dataset = self.root / "dataset.npz"
        self.checkpoint = self.root / "decoder.pt"
        np.savez_compressed(
            self.dataset, trial_counts=self.counts,
            subject_index=self.sid, trial_index=self.tid,
            x=self.x, y=self.choice.reshape(-1, 1),
            parameters=np.zeros((self.n, self.dz), dtype=np.float32))
        if str(mod.NP_SRC) not in __import__("sys").path:
            __import__("sys").path.insert(0, str(mod.NP_SRC))
        from modlearn_rot.nn.np.point.discrete import (
            DiscreteNPConfig, DiscreteNeuralProcess)
        self.config = DiscreteNPConfig(x_dim=self.dx, latent_dim=self.dz)
        torch.manual_seed(177)
        self.teacher = DiscreteNeuralProcess(self.config).eval()
        torch.save({
            "model_state_dict": self.teacher.state_dict(),
            "model_config": self.config.to_dict(),
            "extra_state": {"dataset_sha256": mod.file_sha256(self.dataset)},
        }, self.checkpoint)

    def tearDown(self):
        self.temp.cleanup()

    def call_export(self, **kwargs):
        defaults = {
            "output": self.root / "output",
            "dataset_path": self.dataset,
            "checkpoint_path": self.checkpoint,
            "z_raw": self.zraw, "z_std": self.zstd, "r": self.r,
            "model_name": "C1", "device": "cpu",
            "batch_size": 4, "source_provenance": {"source": "synthetic"},
        }
        defaults.update(kwargs)
        return mod.export_stage1_data(**defaults)

    def test_flattened_geometry_and_exact_frozen_decode(self):
        result = self.call_export()
        folder = Path(result["directory"])
        self.assertEqual(result["n_subjects"], self.n)
        self.assertEqual(result["n_trials_total"], sum(self.counts))
        np.testing.assert_array_equal(np.load(folder / "Zstd.npy"), self.zstd)
        np.testing.assert_array_equal(np.load(folder / "R.npy"), self.r)
        np.testing.assert_array_equal(np.load(folder / "subject_index.npy"), self.sid)
        np.testing.assert_array_equal(np.load(folder / "trial_index.npy"), self.tid)
        np.testing.assert_array_equal(np.load(folder / "trial_counts.npy"), self.counts)
        W = np.load(folder / "W.npy")
        self.assertEqual(W.dtype, np.float32)
        self.assertEqual(W.shape, (sum(self.counts), self.dz + self.dx))
        np.testing.assert_allclose(W[:, :self.dz], self.zstd[self.sid], atol=2e-7)
        np.testing.assert_allclose(W[:, self.dz:], self.x, atol=0)

        with torch.inference_mode():
            for sid in range(self.n):
                indices = np.flatnonzero(self.sid == sid)
                xs = torch.from_numpy(self.x[indices][None])
                raw_z = torch.from_numpy(self.zraw[sid:sid+1].astype(np.float32))
                expected = self.teacher.decode(xs, raw_z)[0].numpy()
                with np.load(folder / "Y.npz", allow_pickle=False) as y:
                    np.testing.assert_allclose(
                        y["logits_nn_2class"][indices], expected, atol=5e-7)
        with np.load(folder / "Y.npz", allow_pickle=False) as y:
            self.assertEqual(set(y.files), {
                "choice_observed", "logit_nn", "p_nn",
                "choice_nn", "logits_nn_2class"})
            np.testing.assert_array_equal(y["choice_observed"], self.choice)
            np.testing.assert_allclose(
                y["logit_nn"],
                y["logits_nn_2class"][:, 1] - y["logits_nn_2class"][:, 0],
                atol=1e-6)
            np.testing.assert_allclose(
                y["p_nn"], 1/(1+np.exp(-y["logit_nn"])), atol=1e-7)
            np.testing.assert_array_equal(
                y["choice_nn"], y["logits_nn_2class"].argmax(axis=1))
        meta = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["w_shape"], [len(self.sid), self.dz+self.dx])
        self.assertEqual(meta["decoder_input"],
                         "hstack([X_original, Zraw[subject_index]]) (NOTE input order differs from W)")
        self.assertTrue(meta["labels_leaked_to_z"])
        self.assertEqual(meta["source_dataset_sha256"], mod.file_sha256(self.dataset))

    def test_export_is_non_destructive_without_overwrite(self):
        original = self.call_export()
        folder = Path(original["directory"])
        before = (folder / "Y.npz").read_bytes()
        with self.assertRaises(FileExistsError):
            self.call_export()
        self.assertEqual((folder / "Y.npz").read_bytes(), before)
        again = self.call_export(overwrite=True)
        self.assertEqual(again["n_trials_total"], sum(self.counts))
        self.assertTrue((folder / "manifest.json").is_file())

    def test_reject_wrong_Z_standardization_and_checkpoint_hash(self):
        with self.assertRaises(ValueError):
            self.call_export(z_std=self.zraw)
        torch.save({
            "model_state_dict": self.teacher.state_dict(),
            "model_config": self.config.to_dict(),
            "extra_state": {"dataset_sha256": "bad-dataset-hash"},
        }, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "another S1 dataset"):
            self.call_export()

    def test_reject_misaligned_trials(self):
        with np.load(self.dataset, allow_pickle=False) as f:
            data = {k: f[k] for k in f.files}
        data["subject_index"] = np.roll(self.sid, 1)
        np.savez_compressed(self.dataset, **data)
        with self.assertRaisesRegex(ValueError, "contiguous subject blocks"):
            self.call_export()


if __name__ == "__main__":
    unittest.main()
