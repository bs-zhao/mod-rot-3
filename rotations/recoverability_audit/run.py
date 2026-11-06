"""Supervised diagnostic ONLY: how much is left for an unsupervised rotation?

Consumes a previously completed mlp_teacher_rotation run, never a checkpoint.
No transform produced here is an unsupervised rotation candidate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment, minimize


def correlations(q, theta):
    q = q - q.mean(axis=0)
    theta = theta - theta.mean(axis=0)
    denom = np.sqrt((q*q).sum(axis=0)[:, None] * (theta*theta).sum(axis=0))
    if np.any(denom <= 0):
        raise ValueError("Constant latent axis or parameter")
    return q.T @ theta / denom


def score(q, theta):
    c = correlations(q, theta)
    rows, cols = linear_sum_assignment(-c*c)
    return float(np.mean(c[rows, cols]**2))


def linear_directions(q, theta):
    qc = q - q.mean(axis=0)
    tc = theta - theta.mean(axis=0)
    b = np.linalg.lstsq(qc, tc, rcond=None)[0]
    return b, q.mean(axis=0), theta.mean(axis=0)


def orthogonal_reference(q, theta, seed=31, starts=16):
    """Numerical supervised reference; NOT a certified orthogonal optimum."""
    k = q.shape[1]
    if theta.shape[1] != k:
        raise ValueError("Requires equal numbers of latent axes and parameters")
    q = q - q.mean(axis=0)
    t = theta - theta.mean(axis=0)
    t = t / np.sqrt((t*t).sum(axis=0))
    s, c = q.T @ q, q.T @ t
    pairs = np.triu_indices(k, 1)

    def rotation(v):
        a = np.zeros((k, k))
        a[pairs] = v
        return expm(a-a.T)

    def loss(v):
        r = rotation(v)
        numerator = np.diag(r @ c)**2
        denominator = np.einsum('ij,jk,ik->i', r, s, r)
        return -float(np.mean(numerator / np.maximum(denominator, 1e-30)))

    rng = np.random.default_rng(seed)
    best = None
    for index in range(starts):
        initial = np.zeros(len(pairs[0])) if index == 0 else rng.normal(0, 1, len(pairs[0]))
        fit = minimize(loss, initial, method="BFGS", options={"maxiter": 300, "gtol": 1e-7})
        if best is None or fit.fun < best.fun:
            best = fit
    return rotation(best.x), float(-best.fun)


def audit(path, out, seeds, starts):
    path, out = Path(path), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(path / "rotation__test_truth.npz", allow_pickle=False) as truth:
        ids, theta = truth["test_ids"], truth["theta_test"]
    with np.load(path / "candidate__identity" / "rotation__U.npz", allow_pickle=False) as latent:
        if not np.array_equal(ids, latent["test_ids"]):
            raise ValueError("TEST ID order differs between truth and saved U")
        q = latent["U"][ids]
    if len(np.unique(ids)) != len(ids) or not np.isfinite(q).all() or not np.isfinite(theta).all():
        raise ValueError("Invalid IDs or nonfinite data")
    meta = json.loads((path / "candidate__identity" / "rotation__test_evaluation.json").read_text())
    names = meta["parameter_names"]
    b, _, _ = linear_directions(q, theta)
    # In-sample OLS prediction yields the exact empirical multiple-correlation
    # bound for ANY linear axis, even when orthogonality is not required.
    empirical_bound = np.diag(correlations(q @ b, theta))**2
    bn = b / np.linalg.norm(b, axis=0)
    r, full_score = orthogonal_reference(q, theta, starts=starts)
    folds = []
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(len(q))
        for flip in (0, 1):
            parts = np.array_split(order, 2)
            train, test = parts[flip], parts[1-flip]
            bf, _, _ = linear_directions(q[train], theta[train])
            rf, fit_score = orthogonal_reference(q[train], theta[train], seed, starts)
            diagonal = np.diag(correlations(q[test] @ rf.T, theta[test]))**2
            linear_test = np.diag(correlations(q[test] @ bf, theta[test]))**2
            folds.append({"seed": seed, "half": flip,
                          "identity": score(q[test], theta[test]),
                          "supervised_orthogonal_fit": fit_score,
                          "supervised_orthogonal_heldout": float(diagonal.mean()),
                          "supervised_linear_heldout": float(linear_test.mean())})
    candidates = []
    for file in sorted(path.glob("candidate__*/rotation__test_evaluation.json")):
        record = json.loads(file.read_text())
        candidates.append({"candidate": file.parent.name.removeprefix("candidate__"),
                           "matched_mean_r2": record["matched_mean_r2"]})
    result = dict(
        source_run=path.name, sample_count=len(q), parameter_names=names,
        identity=score(q, theta),
        empirical_linear_bound_per_parameter=dict(zip(names, empirical_bound.tolist())),
        empirical_linear_bound_mean=float(empirical_bound.mean()),
        supervised_orthogonal_full_sample=full_score,
        normalized_linear_direction_gram=(bn.T @ bn).tolist(),
        half_split_diagnostics=folds, existing_unsupervised_candidates=candidates,
        interpretation="Supervised exploratory audit on previously inspected NP TEST subjects. "
        "Half splits assess supervised fitting generalization within this fixed dataset; they "
        "are not independent NP replications. The numerical orthogonal reference is not an "
        "upper bound. The full-sample OLS bound is empirical, not a population guarantee.")
    (out / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out / "half_split_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(folds[0]))
        writer.writeheader()
        writer.writerows(folds)
    np.savez_compressed(out / "SUPERVISED_DIAGNOSTIC_ONLY.npz", R=r, linear_directions=b)
    lines = ["# Recoverability audit", "", result["interpretation"], "",
             "| Quantity | Mean r² |", "|---|---:|",
             f"| Identity | {result['identity']:.6f} |",
             f"| Empirical unrestricted linear upper bound | {empirical_bound.mean():.6f} |",
             f"| Supervised orthogonal numerical reference, same sample | {full_score:.6f} |",
             f"| Supervised orthogonal, half-split held-out mean | {np.mean([f['supervised_orthogonal_heldout'] for f in folds]):.6f} |",
             "", "No unsupervised objective was fitted or selected using this audit.", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("source_run", "identity", "empirical_linear_bound_mean", "supervised_orthogonal_full_sample")}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 47, 73])
    parser.add_argument("--starts", type=int, default=16)
    args = parser.parse_args()
    if args.starts < 1 or not args.seeds:
        parser.error("Require positive starts and at least one seed")
    audit(args.run, args.out, args.seeds, args.starts)
