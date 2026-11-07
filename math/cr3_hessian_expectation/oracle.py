"""CR3-4P analytic logit/probability and their full 10x10 Hessians.

Independent reproduction of np/reference/s0_cr2_design_recovery.py. This
module is ONLY used by the oracle branch and to construct the experimental
logit response. The data-only regression module does not import this file.

Coordinates: [alpha, lambda_loss, gamma, tau, gA, lA, pA, gB, lB, pB].
At an amount of exactly zero, alpha<1 gives non-finite amount derivatives.
Entries in the corresponding Hessian row/column are reported as NaN.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit

NAMES = ("alpha", "lambda_loss", "gamma", "tau",
         "gain_A", "loss_A", "prob_A", "gain_B", "loss_B", "prob_B")
EPS = 1e-12


def oracle_outputs(T: np.ndarray, X: np.ndarray):
    """Return true logit and probability, shape (n_subjects, n_trials)."""
    T, X = np.asarray(T, float), np.asarray(X, float)
    alpha, lam, gamma, tau = (T[:, None, i] for i in range(4))
    dv = 0.0
    for opt, sign in ((0, 1.0), (1, -1.0)):
        g, loss, prob = (X[None, :, opt * 3 + k] for k in range(3))
        w = expit(gamma * logit(np.clip(prob, EPS, 1-EPS)))
        value = w * (g / 100.0) ** alpha - lam * (1-w) * (loss / 100.0) ** alpha
        dv = dv + sign * value
    eta = tau * dv
    return eta, expit(eta)


def oracle_hessians(T: np.ndarray, X: np.ndarray):
    """Vectorized analytic Hessians of eta and sigmoid(eta).

    T: (n,4); X: (m,6), fixed stimulus bank.
    Returns (H_eta, H_probability), each (n,m,10,10).
    Handles clipped probability weighting exactly away from clipping knots.
    Invalid derivatives at zero monetary amounts are NaN, not zero.
    """
    T, X = np.asarray(T, float), np.asarray(X, float)
    n, m = len(T), len(X)
    alpha, lam, gamma, tau = (T[:, None, i] for i in range(4))
    h = np.zeros((n, m, 10, 10), dtype=np.float64)
    grad = np.zeros((n, m, 10), dtype=np.float64)
    eta = np.zeros((n, m), dtype=np.float64)

    def symmetric_add(i, j, value):
        h[..., i, j] += value
        if i != j:
            h[..., j, i] += value

    for opt, sign in ((0, 1.0), (1, -1.0)):
        k = 4 + 3 * opt
        g, loss, prob = (X[None, :, 3 * opt + j] for j in range(3))
        lg = np.where(g > 0, np.log(np.maximum(g / 100., 1e-300)), 0.)
        ll = np.where(loss > 0, np.log(np.maximum(loss / 100., 1e-300)), 0.)
        a = (g / 100.) ** alpha
        b = (loss / 100.) ** alpha
        aa, ba = a * lg, b * ll
        aaa, baa = aa * lg, ba * ll

        # Zero amounts: fractional-power amount derivative does not exist.
        sg = np.where(g > 0, g, 1.)
        sl = np.where(loss > 0, loss, 1.)
        ag = np.where(g > 0, alpha * a / sg, np.nan)
        bl = np.where(loss > 0, alpha * b / sl, np.nan)
        aag = np.where(g > 0, a / sg * (1. + alpha * lg), np.nan)
        bal = np.where(loss > 0, b / sl * (1. + alpha * ll), np.nan)
        agg = np.where(g > 0, alpha * (alpha - 1.) * a / sg**2, np.nan)
        bll = np.where(loss > 0, alpha * (alpha - 1.) * b / sl**2, np.nan)

        r = logit(np.clip(prob, EPS, 1. - EPS))
        w = expit(gamma * r)
        s = w * (1. - w)
        wg = s * r
        wgg = s * (1. - 2. * w) * r**2
        inside = (prob > EPS) & (prob < 1. - EPS)
        ps = np.where(inside, prob, .5)
        den = ps * (1. - ps)
        wp = np.where(inside, gamma * s / den, 0.)
        wgp = np.where(inside, s / den * (1. + gamma * r * (1. - 2. * w)), 0.)
        wpp = np.where(
            inside,
            s / den**2 * (gamma**2 * (1. - 2. * w) + gamma * (2. * ps - 1.)),
            0.
        )

        v = w * a - lam * (1. - w) * b
        v_alpha = w * aa - lam * (1. - w) * ba
        v_lambda = -(1. - w) * b
        v_gamma = wg * (a + lam * b)
        v_g = w * ag
        v_l = -lam * (1. - w) * bl
        v_p = wp * (a + lam * b)

        eta += sign * tau * v
        grad[..., 0] += sign * tau * v_alpha
        grad[..., 1] += sign * tau * v_lambda
        grad[..., 2] += sign * tau * v_gamma
        grad[..., 3] += sign * v
        grad[..., k] += sign * tau * v_g
        grad[..., k + 1] += sign * tau * v_l
        grad[..., k + 2] += sign * tau * v_p

        # theta-theta block
        symmetric_add(0, 0, sign * tau * (w * aaa - lam * (1. - w) * baa))
        symmetric_add(0, 1, sign * tau * (-(1. - w) * ba))
        symmetric_add(0, 2, sign * tau * wg * (aa + lam * ba))
        symmetric_add(0, 3, sign * v_alpha)
        symmetric_add(1, 2, sign * tau * wg * b)
        symmetric_add(1, 3, sign * v_lambda)
        symmetric_add(2, 2, sign * tau * wgg * (a + lam * b))
        symmetric_add(2, 3, sign * v_gamma)
        # lambda-lambda and tau-tau identically zero for logit.

        # theta-X block
        symmetric_add(0, k, sign * tau * w * aag)
        symmetric_add(0, k + 1, sign * (-tau * lam * (1. - w) * bal))
        symmetric_add(0, k + 2, sign * tau * wp * (aa + lam * ba))
        # d(lambda)/d(gain) is structurally zero
        symmetric_add(1, k + 1, sign * (-tau * (1. - w) * bl))
        symmetric_add(1, k + 2, sign * tau * wp * b)
        symmetric_add(2, k, sign * tau * wg * ag)
        symmetric_add(2, k + 1, sign * tau * lam * wg * bl)
        symmetric_add(2, k + 2, sign * tau * wgp * (a + lam * b))
        symmetric_add(3, k, sign * v_g)
        symmetric_add(3, k + 1, sign * v_l)
        symmetric_add(3, k + 2, sign * v_p)

        # X-X block; different lotteries do not couple on the logit scale.
        symmetric_add(k, k, sign * tau * w * agg)
        symmetric_add(k + 1, k + 1, sign * (-tau * lam * (1. - w) * bll))
        symmetric_add(k + 2, k + 2, sign * tau * wpp * (a + lam * b))
        symmetric_add(k, k + 2, sign * tau * wp * ag)
        symmetric_add(k + 1, k + 2, sign * tau * lam * wp * bl)

    prob_out = expit(eta)
    fac = prob_out * (1. - prob_out)
    h_prob = fac[..., None, None] * (
        h + (1. - 2. * prob_out)[..., None, None] *
        grad[..., :, None] * grad[..., None, :]
    )
    # Conservative domain mask: entire row/column of a zero amount is
    # undefined. Do not silently replace infinities by zero.
    for col in (0, 1, 3, 4):
        idx = np.flatnonzero(X[:, col] == 0.)
        if len(idx):
            for arr in (h, h_prob):
                arr[:, idx, 4 + col, :] = np.nan
                arr[:, idx, :, 4 + col] = np.nan
    return h, h_prob
