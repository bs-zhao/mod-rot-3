"""Quick synthetic tests; do not require a 5000x500 generated dataset.

python -m unittest discover -s math/cr3_hessian_expectation -p 'test_*.py' -v
"""
import unittest

import numpy as np

from oracle import oracle_hessians, oracle_outputs
from data_only import TensorPolynomial
from run import prior_quadrature


class AnalyticHessianTests(unittest.TestCase):
    @staticmethod
    def numerical_hessian(fun, point):
        # Central differences: only use interior, nonzero lotteries.
        point = np.asarray(point, float)
        step = np.array([2e-4, 2e-4, 2e-4, 2e-4,
                         2e-3, 2e-3, 2e-4, 2e-3, 2e-3, 2e-4])
        H = np.zeros((10, 10))
        val0 = fun(point)
        for i in range(10):
            ei = np.zeros(10)
            ei[i] = step[i]
            H[i, i] = (fun(point + ei) - 2 * val0 +
                       fun(point - ei)) / step[i]**2
            for j in range(i):
                ej = np.zeros(10)
                ej[j] = step[j]
                mixed = (fun(point + ei + ej) - fun(point + ei - ej) -
                         fun(point - ei + ej) + fun(point - ei - ej))
                H[i, j] = H[j, i] = mixed / (4 * step[i] * step[j])
        return H

    def test_full_hessian_matches_finite_differences(self):
        point = np.array([.8, 2., .7, 2.2, 100., 60., .6, 80., 20., .4])
        h_eta, h_prob = oracle_hessians(point[None, :4], point[None, 4:])
        for channel, expected in (("logit", h_eta[0, 0]),
                                  ("probability", h_prob[0, 0])):
            def fun(w):
                eta, p = oracle_outputs(w[None, :4], w[None, 4:])
                return float(eta[0, 0] if channel == "logit" else p[0, 0])
            actual = self.numerical_hessian(fun, point)
            np.testing.assert_allclose(expected, actual, rtol=5e-3, atol=2e-4)
            np.testing.assert_allclose(expected, expected.T, atol=1e-12)

    def test_zero_amount_marked_undefined(self):
        theta = np.array([[.8, 2., .7, 2.2]])
        stim = np.array([[0., 30., .6, 80., 20., .4]])
        h_eta, h_prob = oracle_hessians(theta, stim)
        for arr in (h_eta, h_prob):
            self.assertTrue(np.isfinite(arr[0, 0, :4, :4]).all())
            self.assertTrue(np.isnan(arr[0, 0, 4, :]).all())
            self.assertTrue(np.isnan(arr[0, 0, :, 4]).all())

    def test_prior_quadrature_mass(self):
        pts, weights, errors = prior_quadrature(8)
        self.assertEqual(pts.shape, (8**4, 4))
        self.assertAlmostEqual(float(weights.sum()), 1., places=12)
        self.assertTrue(np.all(weights > 0))
        self.assertEqual(len(errors), 4)


class DataOnlyTests(unittest.TestCase):
    def test_polynomial_sieve_recovers_known_synthetic_hessian(self):
        rng = np.random.default_rng(21)
        T = rng.uniform([.3, .6, .2, .7], [.9, 3., .85, 3.], (240, 4))
        X = rng.uniform([5, 5, .1, 5, 5, .1],
                        [200, 200, .9, 200, 200, .9], (120, 6))
        Y = T[:, None, 0]**2 * X[None, :, 0] + \
            3 * T[:, None, 2] * X[None, :, 4]**2
        fit = TensorPolynomial(T, X, degree_t=2, degree_x=2,
                               ridge_t=1e-10, ridge_x=1e-10, seed=11)
        B = fit.fit(Y)
        actual = fit.expected_hessian(B, T, np.ones(len(T))/len(T))
        h = actual["strict_mean"]
        self.assertAlmostEqual(h[0, 0], float(2 * X[:, 0].mean()), delta=.05)
        self.assertAlmostEqual(h[0, 4], float(2 * T[:, 0].mean()), delta=.05)
        self.assertAlmostEqual(h[2, 8], float(6 * X[:, 4].mean()), delta=.1)
        self.assertLess(fit.holdout_metrics(Y, B)["rmse"], .1)


if __name__ == "__main__":
    unittest.main()
