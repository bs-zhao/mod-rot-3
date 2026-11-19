"""Unit/finite-difference tests for CR4 analytic 9x9 Hessians.

Run: python -m unittest discover -s math/cr4_analytic_hessian -p test_cr4_hessian.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cr4_hessian import (analytic_logit, analytic_probability,
                        analytic_nll_hessian, fisher_information,
                        standardized_hessian, coordinate_scales,
                        monte_carlo, sample_theta, sample_subject_x)
from s0_s1_cr4 import logits


class TestCR4AnalyticHessian(unittest.TestCase):
    def test_exact_generator_and_finite_differences(self):
        rng = np.random.default_rng(31)
        for i in range(12):
            theta = sample_theta(1, np.random.default_rng(i + 30))[0]
            x = sample_subject_x(1, seed=i + 100, trials=1)[0, 0]
            eta, grad, h = analytic_logit(theta, x)
            self.assertAlmostEqual(eta, float(logits(x, theta)), places=12)
            self.assertAlmostEqual(float(expit(eta)),
                                   analytic_probability(theta,x)[0], places=12)
            point = np.r_[theta, x]
            for function in (analytic_logit, analytic_probability):
                hess = function(theta, x)[2]
                approx = np.zeros((9, 9))
                for j in range(9):
                    shift = np.zeros(9)
                    shift[j] = 2e-5
                    gp = function((point+shift)[:3], (point+shift)[3:])[1]
                    gm = function((point-shift)[:3], (point-shift)[3:])[1]
                    approx[:,j] = (gp-gm)/(4e-5)
                np.testing.assert_allclose(hess,approx,atol=1e-7,rtol=4e-6)
                np.testing.assert_allclose(hess,hess.T,atol=1e-13)
            self.assertTrue(np.array_equal(h[3:6,6:9],np.zeros((3,3))))
            self.assertTrue(np.array_equal(h[6:9,3:6],np.zeros((3,3))))
            self.assertEqual(h[0,0],0.0)
            self.assertEqual(h[2,2],0.0)

    def test_fisher_as_expected_negative_loglikelihood_hessian(self):
        theta = np.array([2.0,.65,2.5])
        x = np.array([100.,35.,.4,75.,90.,.7])
        q = analytic_probability(theta,x)[0]
        expected = ((1-q)*analytic_nll_hessian(theta,x,0)
                    + q*analytic_nll_hessian(theta,x,1))
        np.testing.assert_allclose(expected,fisher_information(theta,x),atol=1e-12)

    def test_standardization_and_mc(self):
        theta = [1.98,.7,2.2]
        x = [100.,40.,.75,80.,60.,.35]
        h = analytic_logit(theta,x)[2]
        sd = coordinate_scales()
        np.testing.assert_allclose(standardized_hessian(h),
                                   np.diag(sd)@h@np.diag(sd),atol=1e-12)
        result = monte_carlo(samples=20,seed=19)
        self.assertEqual(result['coordinate_system'],'standardized')
        for kind in ('logit','probability'):
            for stat in ('mean','mean_absolute','rms'):
                a = np.asarray(result['hessians'][kind][stat])
                self.assertEqual(a.shape,(9,9))
                self.assertTrue(np.isfinite(a).all())
        self.assertEqual(result,monte_carlo(samples=20,seed=19))


if __name__ == '__main__':
    unittest.main()
