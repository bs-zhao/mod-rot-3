"""CR4 S2 Oracle MLE/Fisher with per-subject IID X; CR5 legacy 2D API kept.

CR4 uses X_i from its S1 dataset without assuming a shared bank. The fitted
parameters use only X_i and choices; true parameters evaluate recovery only.
The 2D-X evaluate/fisher functions remain compatible with CR5's fixed bank.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit as logit_probability

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from s0_s1_cr4 import DATASET, SAMPLING_POLICY, OUT, sample_theta, logits, hash_file
from s0_cr2_design_recovery import BOUNDS
from modlearn_rot.generators.dataset import load_dataset

BOUNDS3 = np.asarray(BOUNDS[4][1:], dtype=float)
NAMES = ["lambda_loss", "gamma", "tau"]


def gradient(x, theta):
    """Exact 3P logit Jacobian wrt (lambda,gamma,tau), including the
    original CR2/CR3 probability-clipping rule at p=0 and p=1."""
    x = np.asarray(x, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    lam, gamma, tau = theta
    pa, pb = x[:, 2], x[:, 5]
    if (not np.isfinite(x).all() or
        np.any((pa < 0) | (pa > 1)) or np.any((pb < 0) | (pb > 1))):
        raise ValueError("Oracle requires finite X and probabilities within [0, 1]")
    # EXACTLY match s0_cr2_design_recovery.weighting(), including its
    # numerical clipping at p=0/1. Direct logit(0/1) produces +/-inf and
    # makes w*(1-w)*logit(p) an indeterminate 0*inf in the Fisher Jacobian.
    # The implemented simulator uses clip(p, 1e-12, 1-1e-12): derivatives
    # with respect to gamma must match that implemented (clipped) function,
    # not the separate ideal mathematical endpoint limit.
    def weight_and_dgamma(p):
        log_odds = logit_probability(np.clip(p, 1e-12, 1 - 1e-12))
        w = expit(gamma * log_odds)
        return w, (w * (1 - w)) * log_odds

    wa, da = weight_and_dgamma(pa)
    wb, db = weight_and_dgamma(pb)
    # Compute loss weighting separately: in the original simulator it is
    # weighting(1-p,gamma), not an independently imposed 1-weighting(p).
    wla, dla = weight_and_dgamma(1 - pa)
    wlb, dlb = weight_and_dgamma(1 - pb)
    gains = wa * (x[:, 0] / 100) - wb * (x[:, 3] / 100)
    losses = wla * (x[:, 1] / 100) - wlb * (x[:, 4] / 100)
    gain_deriv = da * (x[:, 0] / 100) - db * (x[:, 3] / 100)
    loss_deriv = dla * (x[:, 1] / 100) - dlb * (x[:, 4] / 100)
    return np.column_stack([
        -tau * losses,
        tau * (gain_deriv - lam * loss_deriv),
        gains - lam * losses,
    ])


def fisher_diagnostics(x, reference_thetas):
    """2D shared X (CR5), or 3D X_i per reference subject (CR4)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim not in (2, 3) or x.shape[-1] != 6:
        raise ValueError('Fisher X must be (trials,6) or (subjects,trials,6)')
    if x.ndim == 3 and len(x) != len(reference_thetas):
        raise ValueError('Fisher needs one X_i for each reference theta')
    prior_sd = np.std(reference_thetas, axis=0, ddof=1)
    scaled = []
    eigen = []
    for idx, theta in enumerate(reference_thetas):
        subject_x = x[idx] if x.ndim == 3 else x
        p = expit(logits(subject_x, theta))
        grads = gradient(subject_x, theta) * prior_sd
        f = grads.T @ (grads * (p * (1 - p))[:, None])
        scaled.append(np.linalg.cond(f) if np.linalg.eigvalsh(f).min() > 1e-12
                      else float("inf"))
        eigen.append(np.linalg.eigvalsh(f))
    eig = np.asarray(eigen)
    return dict(parameter_names=NAMES, reference_subjects=len(reference_thetas),
                prior_sd=prior_sd.tolist(),
                min_eigenvalue_p01=float(np.percentile(eig[:, 0], 1)),
                min_eigenvalue_median=float(np.median(eig[:, 0])),
                full_rank_fraction=float(np.mean(eig[:, 0] > 1e-8)),
                median_condition=float(np.median(scaled)),
                p95_condition=float(np.percentile(scaled, 95)))


def fit_subject(x, choice, starts, rng):
    def objective(param):
        eta = logits(x, param)
        return float(np.logaddexp(0, eta).sum() - np.dot(choice, eta))
    init = sample_theta(max(1, starts), rng)
    seeds = np.concatenate((np.median(init, axis=0, keepdims=True),
                            init[:max(0, starts-1)]), axis=0)
    best = None
    for seed in seeds:
        result = minimize(objective, seed, method="L-BFGS-B",
                          bounds=BOUNDS3.tolist(),
                          options={"maxiter":450,"ftol":1e-11})
        if best is None or result.fun < best.fun:
            best = result
    return best.x


def evaluate(x, subjects, starts, seed):
    """Simulate and recover 3P theta on variable X_i or a shared CR5 bank."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim not in (2, 3) or x.shape[-1] != 6:
        raise ValueError('Oracle X must be (trials,6) or (subjects,trials,6)')
    if x.ndim == 3 and x.shape[0] != subjects:
        raise ValueError('IID-X oracle requires a bank for each subject')
    rng_theta = np.random.default_rng(seed)
    rng_choice = np.random.default_rng(seed + 101)
    rng_fit = np.random.default_rng(seed + 202)
    truth = sample_theta(subjects, rng_theta)
    p = expit(logits(x[None] if x.ndim == 2 else x,
                     truth[:, None, :]))
    y = (rng_choice.random(p.shape) < p).astype(np.int64)
    fitted = np.zeros_like(truth)
    for i in range(subjects):
        subject_x = x if x.ndim == 2 else x[i]
        fitted[i] = fit_subject(subject_x, y[i], starts, rng_fit)
        if (i + 1) % 25 == 0:
            print(f"CR4 Oracle MLE {i+1}/{subjects}", flush=True)
    error = fitted - truth
    pearson = []
    r2 = []
    for j in range(3):
        t, h = truth[:, j], fitted[:, j]
        pearson.append(float(np.corrcoef(t, h)[0, 1]))
        r2.append(float(1 - np.square(h - t).sum() /
                        max(np.square(t - t.mean()).sum(), 1e-15)))
    result = dict(model="CR4", parameters=NAMES,
                  subjects=subjects, trials_per_subject=x.shape[-2],
                  starts=starts,
                  pearson_r=pearson, r2=r2,
                  rmse=np.sqrt(np.mean(error**2, axis=0)).tolist(),
                  bias=np.mean(error, axis=0).tolist(),
                  error_correlation=np.corrcoef(error, rowvar=False).tolist(),
                  lambda_tau_error_correlation=float(np.corrcoef(error[:, 0], error[:, 2])[0, 1]),
                  bound_hit_fraction=np.mean(
                      (fitted <= BOUNDS3[:, 0] + 1e-4) |
                      (fitted >= BOUNDS3[:, 1] - 1e-4), axis=0).tolist(),
                  oracle_r_ge_0p8=bool(np.all(np.asarray(pearson) >= .8)),
                  note="Independent oracle theta and binary choices; fitted only to X_i,Y_i.")
    return result, truth, fitted, y


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--subjects", type=int, default=100)
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--fisher-subjects", type=int, default=128)
    args = parser.parse_args()
    if args.subjects < 3 or args.starts < 1 or args.fisher_subjects < 3:
        parser.error("Need >=3 MLE and Fisher subjects, >=1 start")
    if not DATASET.is_file():
        raise FileNotFoundError(f"Generate CR4 IID-X S1 dataset first: {DATASET}")
    manifest_path = DATASET.with_name("s1_generate__manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("x_sampling_policy") != SAMPLING_POLICY:
        raise RuntimeError("Legacy CR4 shared-X dataset cannot be used for IID-X oracle")
    if manifest.get("dataset_sha256") != hash_file(DATASET):
        raise RuntimeError("CR4 dataset changed since S1")
    data = load_dataset(DATASET)
    total_subjects = len(data.parameters)
    needed = max(args.subjects, args.fisher_subjects)
    if total_subjects < needed:
        raise ValueError(f"Need {needed} generated CR4 subjects; found {total_subjects}")
    x_all = np.asarray(data.x, dtype=np.float64).reshape(total_subjects, 500, 6)
    truth_refs = sample_theta(args.fisher_subjects,
                              np.random.default_rng(args.seed + 810))
    fisher = fisher_diagnostics(x_all[:args.fisher_subjects], truth_refs)
    result, truth, fitted, choices = evaluate(
        x_all[:args.subjects], args.subjects, args.starts, args.seed + 719)
    result["fisher"] = fisher
    result["dataset_sha256"] = hash_file(DATASET)
    result["x_sampling_policy"] = SAMPLING_POLICY
    folder = OUT / "s2_precheck"
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"s2_precheck__oracle_recovery.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(folder/"s2_precheck__oracle_subjects.npz",
                        theta_true=truth, theta_mle=fitted, choices=choices)
    print("CR4 Oracle parameter r:", np.round(result["pearson_r"], 4))
    print("CR4 Fisher min eigenvalue p01:", fisher["min_eigenvalue_p01"])
    print("CR4 Oracle saved:", folder)


if __name__ == "__main__":
    main()
