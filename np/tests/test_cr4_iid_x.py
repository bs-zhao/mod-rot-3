"""Regression tests for CR4 IID subject/trial stimuli and CR5 compatibility.

Run from repository root:
  python -m unittest discover -s np/tests -p test_cr4_iid_x.py -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "np" / "reference" / "cr4"),
    str(ROOT / "np" / "reference"),
    str(ROOT / "np" / "src"),
]

from s0_s1_cr4 import (sample_subject_x, sample_theta, theta_as_cr3,
                      logits, verify_x)
from s0_cr2_design_recovery import logits as cr2_logits
from s2_oracle_cr4 import fisher_diagnostics
from modrot3 import pipeline


class TestCR4IIDX(unittest.TestCase):
    def test_each_subject_each_trial_has_new_x(self):
        x = sample_subject_x(5, seed=100, trials=500)
        self.assertEqual(x.shape, (5, 500, 6))
        self.assertTrue(verify_x(x))
        self.assertFalse(np.array_equal(x[0], x[1]))
        self.assertFalse(np.array_equal(x[0, 0], x[0, 1]))
        self.assertTrue(np.array_equal(x, sample_subject_x(5, seed=100)))
        self.assertTrue(np.array_equal(x[:2], sample_subject_x(2, seed=100)))
        self.assertFalse(np.array_equal(x, sample_subject_x(5, seed=101)))
        self.assertTrue(np.all((x[..., [0,1,3,4]] > 0)))

    def test_formula_and_parameters_unchanged(self):
        x = sample_subject_x(4, seed=200, trials=15)
        theta = sample_theta(4, np.random.default_rng(300))
        self.assertEqual(theta.shape, (4, 3))
        np.testing.assert_allclose(
            logits(x, theta[:, None, :]),
            cr2_logits(x, theta_as_cr3(theta)[:, None, :]),
            rtol=0, atol=1e-12)

    def test_scaler_uses_distribution_not_a_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "s1_generate__dataset.npz"
            manifest = data.with_name("s1_generate__manifest.json")
            manifest.write_text(json.dumps({
                "x_sampling_policy": "iid_subject_trial_uniform_v1",
                "amount_range": [5, 200],
                "probability_range": [0.05, 0.95],
            }), encoding="utf-8")
            with patch.object(pipeline, "dataset_path", return_value=data):
                mu, sd = pipeline.cr4_input_scaler()
            np.testing.assert_allclose(mu, [102.5,102.5,.5,102.5,102.5,.5])
            np.testing.assert_allclose(
                sd, [195/np.sqrt(12),195/np.sqrt(12),.9/np.sqrt(12),
                     195/np.sqrt(12),195/np.sqrt(12),.9/np.sqrt(12)])

    def test_oracle_fisher_accepts_iid_and_legacy_shared_x(self):
        x = sample_subject_x(4, seed=400, trials=500)
        t = sample_theta(4, np.random.default_rng(500))
        new = fisher_diagnostics(x, t)
        shared = fisher_diagnostics(x[0], t)
        self.assertEqual(new["reference_subjects"], 4)
        self.assertEqual(shared["reference_subjects"], 4)
        self.assertEqual(len(new["prior_sd"]), 3)
        self.assertTrue(np.isfinite(new["min_eigenvalue_median"]))
        # A repeated 3D X must agree numerically with the old 2D shared-X path.
        copied = np.broadcast_to(x[0], (4, 500, 6))
        duplicated = fisher_diagnostics(copied, t)
        self.assertAlmostEqual(shared["min_eigenvalue_median"],
                               duplicated["min_eigenvalue_median"], places=9)


if __name__ == "__main__":
    unittest.main()
