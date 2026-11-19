"""Exact CR4 Hessians for logit, probability, NLL and Fisher (physical W).

W=(lambda,gamma,tau,gain_A,loss_A,p_gain_A,gain_B,loss_B,p_gain_B).
No NP, learned Z, bank or subject dataset is required.

Run from repository root:
  python math/cr4_analytic_hessian/cr4_hessian.py --mode point
  python math/cr4_analytic_hessian/cr4_hessian.py --mode mc --samples 5000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import beta as beta_dist, gamma as gamma_dist, truncnorm

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "np" / "reference" / "cr4"),
                str(ROOT / "np" / "reference"),
                str(ROOT / "np" / "src")]
from s0_s1_cr4 import sample_theta, sample_subject_x

NAMES = ("lambda", "gamma", "tau", "gain_A", "loss_A", "p_A",
         "gain_B", "loss_B", "p_B")
PROTOCOL = "CR4 three-parameter CPT with iid subject/trial X"


def _weights(p, gamma):
    """Return (w, dw/dp, dw/dgamma, d2w/dp2, d2w/dp dgamma, d2w/dgamma2)."""
    if not 0 < p < 1:
        raise ValueError("CR4 X requires 0 < p < 1")
    h = np.log(p) - np.log1p(-p)
    hp = 1.0 / (p * (1.0 - p))
    hpp = (2 * p - 1.0) / (p * p * (1.0 - p)**2)
    w = float(expit(gamma * h))
    a = w * (1.0 - w)
    b = 1.0 - 2.0 * w
    return (w, a * gamma * hp, a * h,
            a * (gamma * hpp + b * gamma**2 * hp**2),
            a * hp * (1.0 + b * gamma * h),
            a * b * h**2)


def analytic_logit(theta, x):
    """Exact eta, d eta / d W, d2 eta / d W2 (9x9)."""
    theta, x = np.asarray(theta, float), np.asarray(x, float)
    if theta.shape != (3,) or x.shape != (6,) or not (
        np.isfinite(theta).all() and np.isfinite(x).all()):
        raise ValueError("theta shape=(3,), X shape=(6,), all finite")
    lam, gamma, tau = theta
    if min(theta) <= 0 or np.any(x[[0, 1, 3, 4]] < 0):
        raise ValueError("CR4 theta positive, payoff magnitudes nonnegative")
    eta = 0.0
    grad = np.zeros(9)
    H = np.zeros((9, 9))
    for s, i in ((1.0, 3), (-1.0, 6)):
        gain, loss, p = x[i-3:i]
        m = gain + lam * loss
        w, wp, wg, wpp, wpg, wgg = _weights(p, gamma)
        c, k = w - 1.0, s / 100.0
        v = w * m - lam * loss
        eta += tau * k * v
        grad[0] += tau * k * loss * c
        grad[1] += tau * k * m * wg
        grad[2] += k * v
        grad[i:i+3] = tau * k * np.array([w, lam*c, m*wp])

        # TT block; entries lambda-lambda and tau-tau are exactly zero.
        H[0, 1] += tau * k * loss * wg
        H[0, 2] += k * loss * c
        H[1, 1] += tau * k * m * wgg
        H[1, 2] += k * m * wg

        # TX: rows lambda, gamma, tau; columns gain_i, loss_i, p_i.
        H[0, i:i+3] = k * np.array([0.0, tau*c, tau*loss*wp])
        H[1, i:i+3] = k * np.array([tau*wg, tau*lam*wg, tau*m*wpg])
        H[2, i:i+3] = k * np.array([w, lam*c, m*wp])

        # XX within an option; A-B XX cross-block is structurally zero.
        H[i:i+3, i:i+3] = tau * k * np.array([
            [0.0, 0.0, wp], [0.0, 0.0, lam*wp],
            [wp, lam*wp, m*wpp],
        ])
    H[1, 0], H[2, 0], H[2, 1] = H[0, 1], H[0, 2], H[1, 2]
    H[3:, :3] = H[:3, 3:].T
    return float(eta), grad, H


def analytic_probability(theta, x):
    """Exact P(choice=A), gradient and Hessian under the sigmoid link."""
    eta, grad, H = analytic_logit(theta, x)
    q = float(expit(eta))
    a = q * (1.0 - q)
    return q, a * grad, a * (H + (1.0-2.0*q) * np.outer(grad, grad))


def analytic_nll_hessian(theta, x, choice):
    """Hessian of observed Bernoulli NLL; choice must be 0 or 1."""
    if choice not in (0, 1):
        raise ValueError("choice must be 0 or 1")
    eta, grad, H = analytic_logit(theta, x)
    q = float(expit(eta))
    return q*(1-q)*np.outer(grad, grad) + (q-choice)*H


def fisher_information(theta, x):
    """Choice-averaged NLL Hessian, 9x9; not a Hessian of P."""
    eta, grad, _ = analytic_logit(theta, x)
    q = float(expit(eta))
    return q*(1-q)*np.outer(grad, grad)


def prior_coordinate_scales():
    """Analytic/distributional SDs for theta=(lambda,gamma,tau)."""
    base = gamma_dist(a=9, scale=.22)
    mean = base.expect(lambda a: a, lb=.3, ub=5., conditional=True)
    second = base.expect(lambda a: a*a, lb=.3, ub=5., conditional=True)
    lstd = np.sqrt(max(0., second-mean*mean))
    gamstd = beta_dist(a=9, b=4).std()
    tstd = truncnorm((.5-2.2)/.55, (4.-2.2)/.55,
                     loc=2.2, scale=.55).std()
    return np.array([lstd, gamstd, tstd], float)


def coordinate_scales(amount_range=(5.,200.), prob_range=(.05,.95)):
    """SDs for the nine independent physical-coordinate distributions."""
    amount_sd = (amount_range[1]-amount_range[0])/np.sqrt(12.)
    prob_sd = (prob_range[1]-prob_range[0])/np.sqrt(12.)
    return np.r_[prior_coordinate_scales(),
                 [amount_sd, amount_sd, prob_sd,
                  amount_sd, amount_sd, prob_sd]]


def standardized_hessian(H, scales=None):
    """Exact Hessian under affine W = mean + diag(scales) @ W_std."""
    sd = coordinate_scales() if scales is None else np.asarray(scales, float)
    if sd.shape != (9,) or np.any(sd <= 0):
        raise ValueError("scales must be a positive vector of size 9")
    return np.asarray(H) * sd[:, None] * sd[None, :]


def monte_carlo(samples=5000, seed=20261118, standardized=True):
    """Compute E[H], E[abs(H)], sqrt(E[H**2]) over exact P(T) P(X).

    Samples are independent virtual subject/trial points, NOT a reused bank.
    The CR4 generator supplies exact theta and X marginal sampling policies.
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    theta = sample_theta(samples, np.random.default_rng(seed+4))
    x = sample_subject_x(samples, seed=seed+2004, trials=1).reshape(samples,6)
    result = {}
    for kind in ("logit", "probability"):
        result[kind] = dict(mean=np.zeros((9,9)),
                            mean_absolute=np.zeros((9,9)),
                            mean_squared=np.zeros((9,9)))
    for i in range(samples):
        for kind, fun in (("logit", analytic_logit),
                          ("probability", analytic_probability)):
            H = fun(theta[i], x[i])[2]
            if standardized:
                H = standardized_hessian(H)
            acc = result[kind]
            acc["mean"] += H
            acc["mean_absolute"] += np.abs(H)
            acc["mean_squared"] += H*H
    for kind in result:
        acc = result[kind]
        acc["mean"] = (acc["mean"]/samples).tolist()
        acc["mean_absolute"] = (acc["mean_absolute"]/samples).tolist()
        acc["rms"] = np.sqrt(acc.pop("mean_squared")/samples).tolist()
    return {"model": "CR4", "formula": PROTOCOL, "names": list(NAMES),
            "samples": samples, "seed": seed,
            "coordinate_system": "standardized" if standardized else "physical",
            "scales": coordinate_scales().tolist() if standardized else [1.]*9,
            "hessians": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("point","mc"), default="point")
    parser.add_argument("--theta", type=float, nargs=3, default=(1.98,.70,2.20))
    parser.add_argument("--x", type=float, nargs=6,
                        default=(100.,40.,.75,80.,60.,.35))
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--physical", action="store_true",
                        help="In MC mode do not standardize derivatives")
    parser.add_argument("--output", type=Path,
                        help="Optional JSON file path; preserves existing files")
    args = parser.parse_args()
    if args.mode == "point":
        eta, g, Heta = analytic_logit(args.theta, args.x)
        q, gq, Hq = analytic_probability(args.theta, args.x)
        payload = {"model":"CR4", "names":NAMES, "theta":args.theta,
                   "x":args.x, "eta":eta, "probability":q,
                   "grad_logit":g.tolist(), "hessian_logit":Heta.tolist(),
                   "grad_probability":gq.tolist(),
                   "hessian_probability":Hq.tolist(),
                   "fisher_information":fisher_information(args.theta,args.x).tolist()}
    else:
        payload = monte_carlo(args.samples,args.seed,
                              standardized=not args.physical)
    result = json.dumps(payload, indent=2)
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"Existing output preserved: {args.output}")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
        print(f"Saved {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
