"""CR3 full Hessian expectation experiment: analytic vs T,X,Y-only.

Run at repo root:
  python math/cr3_hessian_expectation/run.py
  python math/cr3_hessian_expectation/run.py --quad-nodes 12 --degree-t 4 --degree-x 3

Does not train or access NP/rotation models. Reads the immutable S1 NPZ.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import beta, gamma as gamma_distribution, truncnorm

from oracle import NAMES, oracle_hessians, oracle_outputs
from data_only import TensorPolynomial

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "outputs/choice/CR3/s1_generate/s1_generate__dataset.npz"
DEFAULT_OUT = ROOT / "outputs/math/cr3_hessian_expectation"
CHANNELS = ("binary", "logit", "probability")


def to_builtin(value):
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_builtin(v) for v in value]
    return value


def save_json(path, obj):
    Path(path).write_text(json.dumps(to_builtin(obj), indent=2,
                                     ensure_ascii=False, allow_nan=False),
                          encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cr3(path):
    with np.load(path, allow_pickle=False) as d:
        required = ("parameters", "trial_counts", "subject_index",
                    "trial_index", "x", "y", "predictive_true")
        missing = [key for key in required if key not in d.files]
        if missing:
            raise ValueError(f"Dataset missing arrays: {missing}")
        T = np.asarray(d["parameters"], dtype=np.float64)
        counts = np.asarray(d["trial_counts"], dtype=int)
        if T.ndim != 2 or T.shape[1] != 4 or len(counts) != len(T):
            raise ValueError("Expected CR3-4P with T shape (subjects,4)")
        if not np.all(counts == 500):
            raise ValueError("CR3 experiment requires 500 trials per subject")
        n, m = len(T), 500
        xall = np.asarray(d["x"], dtype=np.float64).reshape(n, m, 6)
        X = xall[0].copy()
        if not np.all(xall == X[None, :, :]):
            raise ValueError("The 500 X stimuli must be identical for all subjects")
        if not np.array_equal(d["subject_index"],
                              np.repeat(np.arange(n), m)):
            raise ValueError("Subject order differs from CR3 generator")
        if not np.array_equal(d["trial_index"],
                              np.tile(np.arange(m), n)):
            raise ValueError("Trial order differs from CR3 generator")
        binary = np.asarray(d["y"], float).reshape(n, m)
        probability = np.asarray(d["predictive_true"], float).reshape(n, m)
        if not np.all(np.isin(binary, [0., 1.])):
            raise ValueError("binary choice must be 0/1")
        if not np.isfinite(probability).all():
            raise ValueError("Non-finite probability in saved dataset")
    return T, X, binary, probability


def prior_quadrature(nodes_per_dim):
    """4D independent marginal Gauss-Legendre integration of ACTUAL CR3 prior.

    alpha ~ Beta(12,3); lambda ~ Gamma(9,scale=.22) truncated [.3,5];
    gamma ~ Beta(9,4); tau ~ Normal(2.2,.55) truncated [.5,4].
    The fixed 500 X points are never re-sampled or continuously integrated.
    """
    legx, legw = leggauss(nodes_per_dim)
    norm_lambda = gamma_distribution.cdf(5., 9., scale=.22) - \
                  gamma_distribution.cdf(.3, 9., scale=.22)
    norm_tau = truncnorm((.5 - 2.2)/.55, (4. - 2.2)/.55,
                         loc=2.2, scale=.55)

    specifications = [
        (0., 1., lambda x: beta.pdf(x, 12, 3)),
        (.3, 5., lambda x: gamma_distribution.pdf(x, 9, scale=.22) / norm_lambda),
        (0., 1., lambda x: beta.pdf(x, 9, 4)),
        (.5, 4., lambda x: norm_tau.pdf(x)),
    ]
    nodes, weights, errors = [], [], []
    for lo, hi, density in specifications:
        xi = (hi - lo) * .5 * legx + (hi + lo) * .5
        wi = (hi - lo) * .5 * legw * density(xi)
        raw_mass = float(wi.sum())
        errors.append(raw_mass - 1.)
        nodes.append(xi)
        weights.append(wi / raw_mass)
    T = np.stack(np.meshgrid(*nodes, indexing="ij"), axis=-1).reshape(-1, 4)
    W = np.prod(np.stack(np.meshgrid(*weights, indexing="ij"), axis=-1),
                axis=-1).ravel()
    W /= W.sum()
    return T, W, errors


class HessianMoments:
    """Weighted moments, preserving the domain of non-existent derivatives."""
    def __init__(self):
        self.mass = np.zeros((10, 10))
        self.total = np.zeros((10, 10))
        self.total_squared = np.zeros((10, 10))

    def add(self, h, row_weights):
        # Each theta point's mass is uniformly split among the 500 trials.
        w = np.asarray(row_weights)[:, None, None, None] / h.shape[1]
        good = np.isfinite(h)
        clean = np.where(good, h, 0.)
        self.mass += np.sum(good * w, axis=(0, 1))
        self.total += np.sum(clean * w, axis=(0, 1))
        self.total_squared += np.sum((clean**2) * w, axis=(0, 1))

    def result(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = self.total / self.mass
            sq = self.total_squared / self.mass
        mu[self.mass == 0] = np.nan
        sq[self.mass == 0] = np.nan
        std = np.sqrt(np.maximum(sq - mu**2, 0.))
        return dict(
            strict_mean=np.where(self.mass >= 1 - 1e-10, mu, np.nan),
            conditional_mean=mu,
            conditional_rms=np.sqrt(sq),
            conditional_std=std,
            fraction_defined=self.mass,
        )


def oracle_moments(T_eval, X, weights, batch_size=32):
    acc_eta, acc_p = HessianMoments(), HessianMoments()
    n = len(T_eval)
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        H_eta, H_p = oracle_hessians(T_eval[lo:hi], X)
        acc_eta.add(H_eta, weights[lo:hi])
        acc_p.add(H_p, weights[lo:hi])
        if lo == 0 or hi == n or (lo // batch_size) % 50 == 0:
            print(f"  oracle Hessian batch {hi}/{n}", flush=True)
    p_res = acc_p.result()
    return dict(binary=p_res, logit=acc_eta.result(), probability=p_res)


def comparison(oracle, fitted):
    result = {}
    for block, rows, cols in (
        ("theta_theta", slice(0, 4), slice(0, 4)),
        ("theta_x", slice(0, 4), slice(4, 10)),
        ("x_x", slice(4, 10), slice(4, 10)),
        ("all", slice(0, 10), slice(0, 10)),
    ):
        a = oracle["conditional_mean"][rows, cols]
        b = fitted["conditional_mean"][rows, cols]
        valid = (oracle["fraction_defined"][rows, cols] > 0) & \
                np.isfinite(a) & np.isfinite(b)
        diff = (a - b)[valid]
        result[block] = dict(
            comparable_entries=int(valid.sum()),
            mae=float(np.mean(np.abs(diff))) if len(diff) else None,
            rmse=float(np.sqrt(np.mean(diff**2))) if len(diff) else None,
            max_abs=float(np.max(np.abs(diff))) if len(diff) else None,
        )
    return result


def write_matrix(path, matrix):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + list(NAMES))
        for name, row in zip(NAMES, matrix):
            writer.writerow([name] + ["" if not np.isfinite(v) else
                                      f"{v:.12g}" for v in row])


def heatmaps(root, expectation, results, statistic="strict_mean", block="full"):
    """Two methods x three response types, shared color limits per panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if block == "theta_theta":
        sel = np.s_[:4, :4]
        labels = NAMES[:4]
    elif block == "theta_x":
        sel = np.s_[:4, 4:]
        xlabels = NAMES[4:]
        labels = NAMES[:4]
    else:
        sel = np.s_[:, :]
        labels = NAMES
    arrays = [results[method][channel][statistic][sel]
              for method in ("analytic", "data_only")
              for channel in CHANNELS]
    abs_max = max((float(np.nanmax(np.abs(x))) for x in arrays
                   if np.isfinite(x).any()), default=1.0)
    abs_max = max(abs_max, 1e-12)
    fig, ax = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for ii, method in enumerate(("analytic", "data_only")):
        for jj, channel in enumerate(CHANNELS):
            mat = results[method][channel][statistic][sel]
            panel = ax[ii, jj]
            img = panel.imshow(np.ma.masked_invalid(mat), cmap="coolwarm",
                               vmin=-abs_max, vmax=abs_max)
            panel.set_title(f"{method} / {channel}")
            panel.set_yticks(range(len(labels)), labels, fontsize=7)
            if block == "theta_x":
                panel.set_xticks(range(len(xlabels)), xlabels,
                                 rotation=65, ha="right", fontsize=7)
            else:
                panel.set_xticks(range(len(labels)), labels,
                                 rotation=65, ha="right", fontsize=7)
            if block != "full":
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        if np.isfinite(mat[i, j]):
                            panel.text(j, i, f"{mat[i,j]:.2g}",
                                       ha="center", va="center", fontsize=7)
    fig.colorbar(img, ax=ax, shrink=.65)
    fig.suptitle(f"CR3: {expectation}, {statistic}, {block} (white=undefined)")
    target = root / f"{expectation}__{statistic}__{block}.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quad-nodes", type=int, default=10)
    parser.add_argument("--degree-t", type=int, default=4)
    parser.add_argument("--degree-x", type=int, default=3)
    parser.add_argument("--ridge-t", type=float, default=.01)
    parser.add_argument("--ridge-x", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.quad_nodes < 2 or args.batch_size < 1 or \
            args.degree_t < 2 or args.degree_x < 2 or \
            args.ridge_t <= 0 or args.ridge_x <= 0:
        parser.error("Quadrature, degrees, ridge penalties, or batch invalid")
    if not args.data.exists():
        parser.error(f"Missing CR3 dataset: {args.data}; run "
                     "python np/s1_generate.py --model CR3 --subjects 5000")
    args.output.mkdir(parents=True, exist_ok=True)
    T, X, y_binary, saved_p = load_cr3(args.data)
    print(f"Dataset: {len(T)} subjects x {len(X)} fixed trials", flush=True)

    # CR3 S1 does not store logits: compute ONCE with the true generator
    # strictly to construct the observable logit response channel.
    eta, regenerated_p = oracle_outputs(T, X)
    abs_errors = np.abs(regenerated_p - saved_p)
    p_error = float(np.max(abs_errors))
    # Both calculations must reproduce the separately clipped CR3 generator.
    # This is a precision check, not permission to hide systematic bias.
    if not np.allclose(regenerated_p, saved_p, atol=1e-12, rtol=1e-12):
        row, col = np.unravel_index(np.argmax(abs_errors), abs_errors.shape)
        raise ValueError(
            f"Saved P disagrees with exact CR3 formula: max_abs={p_error:.12g}; "
            f"subject={row}, trial={col}, saved={saved_p[row,col]:.17g}, "
            f"recomputed={regenerated_p[row,col]:.17g}, "
            f"X={X[col].tolist()}, T={T[row].tolist()}. "
            "Check source formula and independently clipped p versus (1-p)."
        )
    print(f"Exact CR3 saved-probability check PASS: max_abs={p_error:.3g}", flush=True)
    Y = {"binary": y_binary, "logit": eta, "probability": saved_p}

    qt, qw, mass_error = prior_quadrature(args.quad_nodes)
    evaluation = (
        ("sample", T, np.full(len(T), 1. / len(T))),
        ("prior", qt, qw),
    )
    fit = TensorPolynomial(T, X, args.degree_t, args.degree_x,
                           args.ridge_t, args.ridge_x, args.seed)
    coef, diagnostics = {}, {}
    for target in CHANNELS:
        print(f"Fitting T,X,Y-only sieve for: {target}", flush=True)
        coef[target] = fit.fit(Y[target])
        diagnostics[target] = fit.holdout_metrics(Y[target], coef[target])
        if target == "binary":
            diagnostics[target]["rmse_to_true_p"] = float(np.sqrt(np.mean(
                (fit.phi_t[fit.test_idx] @ coef[target] @ fit.phi_x.T -
                 saved_p[fit.test_idx])**2)))
    final_results = {}
    report = dict(
        dataset=str(args.data),
        dataset_sha256=file_sha256(args.data),
        shape={"T": list(T.shape), "X": list(X.shape),
               "Y": list(y_binary.shape)},
        parameters=list(NAMES[:4]),
        variables=list(NAMES),
        channels={
            "binary": "Bernoulli Y; analytic target is Hessian of E[Y|T,X]=p",
            "logit": "true deterministic sigmoid pre-activation eta",
            "probability": "stored predictive_true sigmoid(eta)"
        },
        expectation_definition={
            "sample": "1/N sum_subject [1/500 sum_fixed_trial H(T_i,X_t)]",
            "prior": "integral H(theta,X_t) p_CR3(theta) dtheta / 500, summed over fixed t",
            "undefined": "zero gains/losses and alpha<1: X derivatives undefined; "
                         "strict_mean=null, conditional_mean only over defined trials"
        },
        quadrature={"nodes_per_dimension": args.quad_nodes,
                    "total_nodes": len(qt),
                    "marginal_mass_error_before_normalization": mass_error},
        regression={"method": "tensor-product polynomial sieve with "
                              "two-sided Tikhonov regularized least squares",
                    "degree_t": args.degree_t, "degree_x": args.degree_x,
                    "ridge_t": args.ridge_t, "ridge_x": args.ridge_x,
                    "train_fraction": .8, "seed": args.seed,
                    "warning": "X-derivatives require untestable smooth extension "
                               "between the 500 discrete X stimuli"},
        saved_probability_max_abs_error=p_error,
        holdout=diagnostics, expectations={},
    )
    for name, values, weights in evaluation:
        print(f"Evaluating {name}: {len(values)} theta points", flush=True)
        oracle = oracle_moments(values, X, weights, args.batch_size)
        fitted = {
            channel: fit.expected_hessian(coef[channel], values, weights)
            for channel in CHANNELS
        }
        final_results[name] = dict(analytic=oracle, data_only=fitted)
        report["expectations"][name] = {}
        for channel in CHANNELS:
            report["expectations"][name][channel] = dict(
                comparison=comparison(oracle[channel], fitted[channel]))
            for method, collection in (("analytic", oracle),
                                       ("data_only", fitted)):
                for stat, matrix in collection[channel].items():
                    if matrix.shape != (10, 10):
                        continue
                    prefix = f"{name}__{method}__{channel}__{stat}"
                    write_matrix(args.output / f"{prefix}.csv", matrix)
        if not args.no_plots:
            for stat, block in (
                ("strict_mean", "full"),
                ("strict_mean", "theta_theta"),
                ("conditional_mean", "theta_x"),
                ("conditional_rms", "theta_theta"),
            ):
                heatmaps(args.output, name, final_results[name], stat, block)
    save_json(args.output / "summary.json", report)
    # Full numeric results also available in a single machine-readable NPZ.
    np.savez_compressed(args.output / "all_matrices.npz", **{
        f"{scope}__{method}__{target}__{stat}": mat
        for scope, methods in final_results.items()
        for method, targets in methods.items()
        for target, stats in targets.items()
        for stat, mat in stats.items()
    })
    print("Complete. Results saved to:", args.output, flush=True)
    for scope in ("sample", "prior"):
        print(scope, report["expectations"][scope]["probability"]["comparison"],
              flush=True)


if __name__ == "__main__":
    main()
