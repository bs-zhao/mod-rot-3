"""CR3-4P exact analytic logit/probability and full 10x10 Hessians.

Independently reproduces np/reference/s0_cr2_design_recovery.py.
Only used for the analytic oracle and the true-logit response construction;
data_only.py NEVER imports this module.

Coordinates: alpha, lambda, gamma, tau, gainA, lossA, pA, gainB, lossB, pB.
For alpha in (0,1), amount derivatives at EXACT zero do not exist.
Those Hessian rows/columns are marked NaN, without masking TT entries.

IMPORTANT: The source generator computes weighting(p) and weighting(1-p)
SEPARATELY (and clips their probabilities separately). Do NOT replace
weighting(1-p) by 1-weighting(p): finite-precision clipping at p=0/1
can produce meaningful mismatches, particularly for small gamma.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit

NAMES = ("alpha", "lambda_loss", "gamma", "tau",
         "gain_A", "loss_A", "prob_A", "gain_B", "loss_B", "prob_B")
EPS = 1e-12


def oracle_outputs(T: np.ndarray, X: np.ndarray):
    """Return true generator logit and probability, shape (subjects, trials)."""
    T, X = np.asarray(T, float), np.asarray(X, float)
    alpha, lam, gamma, tau = (T[:, None, i] for i in range(4))
    dv = 0.0
    for opt, sign in ((0, 1.0), (1, -1.0)):
        g, loss, prob = (X[None, :, opt * 3 + k] for k in range(3))
        wg = expit(gamma * logit(np.clip(prob, EPS, 1. - EPS)))
        wl = expit(gamma * logit(np.clip(1. - prob, EPS, 1. - EPS)))
        value = wg * (g / 100.0) ** alpha - lam * wl * (loss / 100.0) ** alpha
        dv = dv + sign * value
    eta = tau * dv
    return eta, expit(eta)


def oracle_hessians(T: np.ndarray, X: np.ndarray):
    """Analytic Hessians of the exact generated logit and probability.

    T: (n,4), X: (m,6) fixed stimulus bank.
    Returns (H_eta, H_prob), both (n,m,10,10).
    Derivatives in probability coordinates are those of the original clipped
    function away from clipping boundary knots (p=EPS or p=1-EPS).
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
        a, b = (g / 100.) ** alpha, (loss / 100.) ** alpha
        aa, ba = a * lg, b * ll
        aaa, baa = aa * lg, ba * ll

        # Fractional powers at zero monetary amounts are NOT differentiable.
        sg, sl = np.where(g > 0, g, 1.), np.where(loss > 0, loss, 1.)
        ag = np.where(g > 0, alpha * a / sg, np.nan)
        bl = np.where(loss > 0, alpha * b / sl, np.nan)
        aag = np.where(g > 0, a / sg * (1. + alpha * lg), np.nan)
        bal = np.where(loss > 0, b / sl * (1. + alpha * ll), np.nan)
        agg = np.where(g > 0, alpha * (alpha - 1.) * a / sg**2, np.nan)
        bll = np.where(loss > 0, alpha * (alpha - 1.) * b / sl**2, np.nan)

        # Exactly replicate the TWO independent probability-weight calls.
        rg = logit(np.clip(prob, EPS, 1. - EPS))
        rl = logit(np.clip(1. - prob, EPS, 1. - EPS))
        wg, wl = expit(gamma * rg), expit(gamma * rl)
        sgam, slam = wg * (1. - wg), wl * (1. - wl)
        wg_gamma, wl_gamma = sgam * rg, slam * rl
        wg_gg = sgam * (1. - 2. * wg) * rg**2
        wl_gg = slam * (1. - 2. * wl) * rl**2

        interior = (prob > EPS) & (prob < 1. - EPS)
        safe_p = np.where(interior, prob, .5)
        den = safe_p * (1. - safe_p)
        rg_p = np.where(interior, 1. / den, 0.)
        rl_p = np.where(interior, -1. / den, 0.)
        rg_pp = np.where(interior, (2. * safe_p - 1.) / den**2, 0.)
        rl_pp = -rg_pp
        wg_p = gamma * sgam * rg_p
        wl_p = gamma * slam * rl_p
        wg_gp = sgam * rg_p * (1. + gamma * rg * (1. - 2. * wg))
        wl_gp = slam * rl_p * (1. + gamma * rl * (1. - 2. * wl))
        wg_pp = sgam * (
            gamma * rg_pp + gamma**2 * (1. - 2. * wg) * rg_p**2)
        wl_pp = slam * (
            gamma * rl_pp + gamma**2 * (1. - 2. * wl) * rl_p**2)

        v = wg * a - lam * wl * b
        v_alpha = wg * aa - lam * wl * ba
        v_lambda = -wl * b
        v_gamma = wg_gamma * a - lam * wl_gamma * b
        v_g, v_l = wg * ag, -lam * wl * bl
        v_p = wg_p * a - lam * wl_p * b

        eta += sign * tau * v
        grad[..., 0] += sign * tau * v_alpha
        grad[..., 1] += sign * tau * v_lambda
        grad[..., 2] += sign * tau * v_gamma
        grad[..., 3] += sign * v
        grad[..., k] += sign * tau * v_g
        grad[..., k + 1] += sign * tau * v_l
        grad[..., k + 2] += sign * tau * v_p

        # TT block
        symmetric_add(0, 0, sign * tau * (wg * aaa - lam * wl * baa))
        symmetric_add(0, 1, sign * (-tau * wl * ba))
        symmetric_add(0, 2, sign * tau * (wg_gamma * aa - lam * wl_gamma * ba))
        symmetric_add(0, 3, sign * v_alpha)
        symmetric_add(1, 2, sign * (-tau * wl_gamma * b))
        symmetric_add(1, 3, sign * v_lambda)
        symmetric_add(2, 2, sign * tau * (wg_gg * a - lam * wl_gg * b))
        symmetric_add(2, 3, sign * v_gamma)
        # Lambda-lambda and tau-tau are identically zero on logit scale.

        # TX block
        symmetric_add(0, k, sign * tau * wg * aag)
        symmetric_add(0, k + 1, sign * (-tau * lam * wl * bal))
        symmetric_add(0, k + 2, sign * tau * (wg_p * aa - lam * wl_p * ba))
        # Lambda / gain cross derivative is structurally zero.
        symmetric_add(1, k + 1, sign * (-tau * wl * bl))
        symmetric_add(1, k + 2, sign * (-tau * wl_p * b))
        symmetric_add(2, k, sign * tau * wg_gamma * ag)
        symmetric_add(2, k + 1, sign * (-tau * lam * wl_gamma * bl))
        symmetric_add(2, k + 2, sign * tau * (wg_gp * a - lam * wl_gp * b))
        symmetric_add(3, k, sign * v_g)
        symmetric_add(3, k + 1, sign * v_l)
        symmetric_add(3, k + 2, sign * v_p)

        # XX block: separate lotteries contribute additively to logit.
        symmetric_add(k, k, sign * tau * wg * agg)
        symmetric_add(k + 1, k + 1, sign * (-tau * lam * wl * bll))
        symmetric_add(k + 2, k + 2, sign * tau * (wg_pp * a - lam * wl_pp * b))
        symmetric_add(k, k + 2, sign * tau * wg_p * ag)
        symmetric_add(k + 1, k + 2, sign * (-tau * lam * wl_p * bl))

    probability = expit(eta)
    fac = probability * (1. - probability)
    h_probability = fac[..., None, None] * (
        h + (1. - 2. * probability)[..., None, None] *
        grad[..., :, None] * grad[..., None, :]
    )
    # Mask undefined X derivatives only; TT remains well-defined.
    for col in (0, 1, 3, 4):
        bad_trials = np.flatnonzero(X[:, col] == 0.)
        if len(bad_trials):
            for matrix in (h, h_probability):
                matrix[:, bad_trials, 4 + col, :] = np.nan
                matrix[:, bad_trials, :, 4 + col] = np.nan
    return h, h_probability
