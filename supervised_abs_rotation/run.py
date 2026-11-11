"""Step 1: full-sample supervised orthogonal rotation + axis-wise scales.

Primary objective (literal absolute values), with every Z axis z-scored first:
    Zstd=(Z-mean_all(Z))/std_all(Z)
    min_{R.T R=I, D diagonal} mean((abs(Zstd @ R.T @ D)-abs(T_std))**2)
The optimal |D| for any R has a closed form. Axis signs are not identifiable
from magnitudes; we select them post hoc by the signed Z/T covariance.
This is a SUPERVISED, IN-SAMPLE ORACLE, never an unsupervised result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
DIMS = {"C1": 4, "C2": 4, "C3": 6, "CR3": 4}
PARAMETERS = {
    "C1": ["b1", "b2", "b3", "b4"],
    "C2": ["b1", "b2", "b3", "b4"],
    "C3": ["b1", "b2", "b3", "b4", "b5", "b6"],
    "CR3": ["alpha", "lambda", "gamma", "tau"],
}


def tag(x):
    return f"{float(x):g}".replace("-", "m").replace(".", "p").replace("+", "")


def condition(beta, sparse, cmin, cmax, dim):
    return f"beta{tag(beta)}_lambda{tag(sparse)}_ctx{tag(cmin)}to{tag(cmax)}_L{dim}"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_existing_np(model, beta, dim, sparse, cmin, cmax, repeat):
    """Read S5 posterior MEANS; verify S3/S1 hashes and all-subject coverage."""
    folder = ROOT / "outputs" / "choice" / model
    cond = condition(beta, sparse, cmin, cmax, dim)
    stem = folder / "s4_evaluate" / cond / f"repeat{repeat:02d}"
    source = stem / "s4_evaluate__latent.npz"
    checkpoint = folder / "s3_train" / cond / f"repeat{repeat:02d}" / "s3_train__checkpoint.pt"
    dataset = folder / "s1_generate" / "s1_generate__dataset.npz"
    for path, hint in ((source, "Run np/s5_extract_z.py with matching settings"),
                       (checkpoint, "Check trained S3 model"),
                       (dataset, "Check S1-generated data")):
        if not path.is_file():
            raise FileNotFoundError(f"{hint}: {path}")

    with np.load(source, allow_pickle=False) as saved:
        required = ("z", "theta_true", "checkpoint_sha256", "dataset_sha256",
                    "context_mode", "train_ids", "validation_ids", "test_ids")
        missing = [key for key in required if key not in saved]
        if missing:
            raise ValueError(f"S5 output is missing keys: {missing}")
        if str(saved["context_mode"]) != "all_observed_trials":
            raise ValueError("Require S5 posterior means from ALL observed trials")
        if str(saved["checkpoint_sha256"]) != sha256(checkpoint):
            raise ValueError("S5 latent does not match the frozen S3 checkpoint")
        if str(saved["dataset_sha256"]) != sha256(dataset):
            raise ValueError("S5 latent does not match S1 dataset")
        z = np.array(saved["z"], dtype=np.float64)
        theta = np.array(saved["theta_true"], dtype=np.float64)
        ids = [np.asarray(saved[f"{s}_ids"], dtype=np.int64)
               for s in ("train", "validation", "test")]

    if z.ndim != 2 or z.shape != theta.shape or z.shape[1] != dim:
        raise ValueError(f"{model} requires Z and T both shaped (N,{dim})")
    if not np.array_equal(np.sort(np.concatenate(ids)), np.arange(len(z))):
        raise ValueError("S5 split indices must be exhaustive and disjoint")
    if not np.isfinite(z).all() or not np.isfinite(theta).all():
        raise ValueError("Z and T must be finite")
    return z, theta, {"latent_path": str(source),
                      "latent_sha256": sha256(source),
                      "checkpoint_sha256": sha256(checkpoint),
                      "dataset_sha256": sha256(dataset),
                      "original_s5_split_sizes": dict(zip(
                          ("train", "validation", "test"), map(len, ids)))}


def corr(x, y):
    """Pearson matrix, rows=x columns, columns=y columns; zero-variance -> 0."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc, yc = x - x.mean(0), y - y.mean(0)
    den = np.linalg.norm(xc, axis=0)[:, None] * np.linalg.norm(yc, axis=0)[None, :]
    return np.divide(xc.T @ yc, den, out=np.zeros((x.shape[1], y.shape[1])),
                     where=den > 1e-12)


def prepare_all(z, t):
    z = np.asarray(z, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if z.ndim != 2 or t.shape != z.shape or len(z) < z.shape[1] + 2:
        raise ValueError("Require equal (N,d) Z and T, with N > d+1")
    if not np.isfinite(z).all() or not np.isfinite(t).all():
        raise ValueError("Nonfinite Z/T")
    zm, tm = z.mean(axis=0), t.mean(axis=0)
    zs, ts = z.std(axis=0, ddof=0), t.std(axis=0, ddof=0)
    if np.any(ts <= 1e-12) or np.any(zs <= 1e-12):
        raise ValueError("Constant Z or T axis: rotation/standardization unavailable")
    return (z - zm) / zs, (t - tm) / ts, zm, zs, tm, ts


def profile(zstd, target, r, objective="abs_mse"):
    """Profile out optimal diagonal D analytically for fixed orthogonal R."""
    v = zstd @ r.T
    if objective == "abs_mse":
        x, y = np.abs(v), np.abs(target)
    elif objective == "signed_mse":
        x, y = v, target
    else:
        raise ValueError(f"Unknown objective: {objective}")
    denom = np.sum(x * x, axis=0)
    if np.any(denom <= 1e-14):
        raise ValueError("A rotated axis has (near) zero variance")
    scale = np.sum(x * y, axis=0) / denom
    loss = float(np.mean((x * scale - y) ** 2))
    return loss, scale


def skew_matrix(params, dim):
    i, j = np.triu_indices(dim, 1)
    k = np.zeros((dim, dim), dtype=float)
    k[i, j], k[j, i] = params, -params
    return k


def fit(zstd, tstd, objective="abs_mse", *, seed=31, restarts=6, maxiter=350):
    """Multistart local search on O(d), with exact optimal D per candidate R."""
    zstd = np.asarray(zstd, float)
    tstd = np.asarray(tstd, float)
    n, d = zstd.shape
    if tstd.shape != (n, d) or restarts < 0 or maxiter < 1:
        raise ValueError("Invalid fitting settings")

    # Row-vector equivalent: Zc @ Q approximates Tstd, R = Q.T.
    u, _, vh = np.linalg.svd(zstd.T @ tstd, full_matrices=False)
    procrustes = (u @ vh).T
    ols = np.linalg.lstsq(zstd, tstd, rcond=None)[0]
    u2, _, vh2 = np.linalg.svd(ols, full_matrices=False)
    ols_polar = (u2 @ vh2).T
    rng = np.random.default_rng(seed)
    starts = [("procrustes", procrustes), ("ols_polar", ols_polar),
              ("identity", np.eye(d))]
    for j in range(restarts):
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        starts.append((f"random_{j + 1:02d}", q))

    n_angles = d * (d - 1) // 2
    attempts = []
    best = None
    for label, r0 in starts:
        def candidate(x):
            return r0 @ expm(skew_matrix(x, d))

        def objective_fn(x):
            return profile(zstd, tstd, candidate(x), objective)[0]

        start_loss = objective_fn(np.zeros(n_angles))
        opt = minimize(objective_fn, np.zeros(n_angles), method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12})
        r = candidate(opt.x)
        loss, scale = profile(zstd, tstd, r, objective)
        attempts.append({"start": label, "initial_loss": start_loss,
                         "final_loss": loss, "iterations": int(opt.nit),
                         "success": bool(opt.success), "message": str(opt.message)})
        if best is None or loss < best["loss"]:
            best = {"R": r, "scale_profile": scale,
                    "loss": loss, "best_start": label}

    # |DRZ| does not identify signs: choose a sign using signed covariance.
    v = zstd @ best["R"].T
    if objective == "abs_mse":
        signed_cov = np.sum(v * tstd, axis=0)
        signs = np.where(signed_cov < 0, -1.0, 1.0)
        scale = best["scale_profile"] * signs
    else:
        scale = best["scale_profile"]
        signs = np.where(scale < 0, -1.0, 1.0)
    r = best["R"]
    D = np.diag(scale)
    if not np.allclose(r @ r.T, np.eye(d), atol=1e-8):
        raise AssertionError("R is not orthogonal")
    prediction = v @ D
    actual_loss = (np.mean((np.abs(prediction) - np.abs(tstd)) ** 2)
                   if objective == "abs_mse" else np.mean((prediction - tstd) ** 2))
    if not np.isclose(actual_loss, best["loss"], atol=1e-10, rtol=1e-9):
        raise AssertionError("Profiled objective does not match predictions")
    best.update(D=D, scales=scale, chosen_signs=signs, U=v,
                prediction=prediction, DR=D @ r,
                attempts=attempts, determinant_R=float(np.linalg.det(r)))
    return best


def zca_whiten_all(u, relative_cutoff=1e-10):
    """ZCA-whiten final unscaled U using ALL subjects (no true T involved).

    For a rank-deficient covariance use truncated pseudoinverse whitening:
    the whitened covariance is a projection, not identity.
    Row-vector convention: U_white=(U-mean(U))@W.
    """
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2 or len(u) < u.shape[1] + 2 or not np.isfinite(u).all():
        raise ValueError("Expected finite U with shape (N,d), N>d+1")
    mean = u.mean(axis=0)
    centered = u - mean
    covariance = centered.T @ centered / len(u)  # ddof=0, all subjects
    eigvals, eigvecs = np.linalg.eigh(covariance)
    maxeig = float(eigvals[-1])
    if maxeig <= 1e-14:
        raise ValueError("Cannot whiten zero-variance U")
    cutoff = max(1e-14, relative_cutoff * maxeig)
    keep = eigvals > cutoff
    inverse_root = np.zeros_like(eigvals)
    inverse_root[keep] = 1.0 / np.sqrt(eigvals[keep])
    W = (eigvecs * inverse_root) @ eigvecs.T
    u_white = centered @ W
    covariance_white = u_white.T @ u_white / len(u_white)
    return dict(U_whitened=u_white, whitening_matrix=W, U_mean_all=mean,
                U_cov_all=covariance, U_whitened_cov_all=covariance_white,
                eigenvalues=eigvals, effective_rank=int(keep.sum()),
                eigenvalue_cutoff=cutoff)


def save_matrix(path, matrix, row_names, col_names):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", *col_names])
        for name, row in zip(row_names, matrix):
            w.writerow([name, *[f"{float(x):.14g}" for x in row]])


def save_rows(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_png(path, figure, dpi=170):
    """Render PNG in memory before filesystem I/O (avoids Pillow opening the path).

    Retry using a shorter deterministic name if the original Windows pathname
    triggers OSError. Returns (actual_file, fallback_reason_or_None).
    """
    from io import BytesIO
    path = Path(path)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi)
    payload = buffer.getvalue()
    try:
        path.write_bytes(payload)
        return path, None
    except OSError as original_error:
        short = path.with_name("fig_" + hashlib.sha256(
            path.name.encode("utf-8")).hexdigest()[:12] + ".png")
        try:
            short.write_bytes(payload)
        except OSError as second_error:
            raise OSError(
                f"Cannot save plot as {path} or shorter alternative {short}. "
                f"First error: {original_error}; fallback error: {second_error}"
            ) from second_error
        return short, str(original_error)


def plot_heatmap(path, matrix, row_names, col_names, title, *, correlation=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(5.5, len(col_names) * .95 + 2),
                                    max(4.5, len(row_names) * .72 + 2)))
    bound = 1.0 if correlation else max(float(np.abs(matrix).max()), 1e-8)
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-bound, vmax=bound)
    ax.set_xticks(range(len(col_names)), col_names, rotation=40, ha="right")
    ax.set_yticks(range(len(row_names)), row_names)
    ax.set_title(title)
    for i in range(len(row_names)):
        for j in range(len(col_names)):
            x = matrix[i, j]
            ax.text(j, i, f"{x:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(x) > .65 * bound else "black")
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout()
    try:
        return write_png(path, fig, dpi=170)
    finally:
        plt.close(fig)


def analyze(z, theta, *, model, output, objective="abs_mse", seed=31,
            restarts=6, maxiter=350, plots=True, provenance=None, overwrite=False):
    """ALL subjects standardize each Z axis before orthogonal R, D and whitening."""
    if model not in DIMS or z.shape[1] != DIMS[model]:
        raise ValueError("Wrong model or latent dimension")
    zstd, tstd, zm, zs, tm, ts = prepare_all(z, theta)
    result = fit(zstd, tstd, objective, seed=seed, restarts=restarts, maxiter=maxiter)
    u = result["U"]
    whitened = zca_whiten_all(u)  # post-processing; R and D remain fixed
    uw = whitened["U_whitened"]
    pred = result["prediction"]
    n, d = zstd.shape
    labels = PARAMETERS[model]
    axes = [f"U{i+1}" for i in range(d)]
    znames = [f"z{i+1}" for i in range(d)]
    zstd_names = [f"zstd{i+1}" for i in range(d)]
    loss_abs = np.mean((np.abs(pred) - np.abs(tstd)) ** 2, axis=0)
    loss_signed = np.mean((pred - tstd) ** 2, axis=0)
    abs_target = np.abs(tstd)
    denom_abs = np.sum((abs_target - abs_target.mean(0)) ** 2, axis=0)
    abs_r2 = 1 - np.sum((np.abs(pred) - abs_target) ** 2, axis=0) / denom_abs
    signed_r2 = 1 - np.mean((pred - tstd) ** 2, axis=0)
    cross_signed = corr(pred, tstd)
    cross_abs = corr(np.abs(pred), abs_target)
    corr_matrices = {
        "Z_Z": (corr(z, z), znames, znames),
        "Tstd_Tstd": (corr(tstd, tstd), labels, labels),
        "Z_Tstd": (corr(zstd, tstd), znames, labels),
        "Zstd_Tstd": (corr(zstd, tstd), zstd_names, labels),
        "U_U": (corr(u, u), axes, axes),
        "U_Tstd": (corr(u, tstd), axes, labels),
        "U_whitened_Tstd": (corr(uw, tstd), axes, labels),
        "prediction_Tstd_signed": (cross_signed, axes, labels),
        "prediction_Tstd_magnitude": (cross_abs, axes, labels),
        "prediction_prediction_signed": (corr(pred, pred), axes, axes),
        "prediction_prediction_magnitude": (corr(np.abs(pred), np.abs(pred)), axes, axes),
    }
    out = Path(output)
    if (out / "summary.json").exists() and not overwrite:
        raise FileExistsError(f"Results exist; use --overwrite to replace: {out}")
    out.mkdir(parents=True, exist_ok=True)
    save_matrix(out / "U_zca_whitening_matrix.csv",
                whitened["whitening_matrix"], axes, axes)
    save_matrix(out / "U_whitened_covariance.csv",
                whitened["U_whitened_cov_all"], axes, axes)
    save_matrix(out / "corr__U_whitened_U_whitened.csv",
                corr(uw, uw), axes, axes)
    for key, (matrix, rows, columns) in corr_matrices.items():
        save_matrix(out / f"corr__{key}.csv", matrix, rows, columns)
    save_rows(out / "Z_zscore_stats.csv", [
        {"dimension": name, "mean_all": float(mean), "std_all_ddof0": float(sd)}
        for name, mean, sd in zip(znames, zm, zs)
    ])
    save_matrix(out / "R.csv", result["R"], axes, zstd_names)
    save_matrix(out / "D.csv", result["D"], axes, axes)
    save_matrix(out / "DR.csv", result["DR"], axes, zstd_names)
    # DR acts on standardized Z; convert back to original units for raw-Z maps.
    A_raw = result["DR"] / zs[None, :]
    save_matrix(out / "A_rawZ_to_Tstd.csv", A_raw, axes, znames)
    intercept = -(A_raw @ zm)
    save_rows(out / "b_rawZ_to_Tstd.csv", [
        {"parameter": label, "intercept": float(value)}
        for label, value in zip(labels, intercept)
    ])
    rows = []
    for j, name in enumerate(labels):
        wrong = np.delete(cross_signed[j], j)
        rows.append({
            "parameter": name, "scale_D": float(result["scales"][j]),
            "signed_r": float(cross_signed[j, j]),
            "abs_signed_r": float(abs(cross_signed[j, j])),
            "magnitude_r": float(cross_abs[j, j]),
            "magnitude_R2": float(abs_r2[j]),
            "signed_R2": float(signed_r2[j]),
            "magnitude_MSE": float(loss_abs[j]),
            "signed_MSE": float(loss_signed[j]),
            "max_wrong_abs_signed_r": float(np.max(np.abs(wrong))),
        })
    save_rows(out / "per_parameter.csv", rows)
    np.savez_compressed(out / "oracle_full_sample.npz",
                        Z_centered=np.asarray(z) - zm,
                        Z_std=zstd,
                        theta_true=np.asarray(theta),
                        T_std=tstd, U_unscaled=u,
                        U_whitened=uw,
                        U_zca_whitening_matrix=whitened["whitening_matrix"],
                        U_mean_all=whitened["U_mean_all"],
                        U_cov_all=whitened["U_cov_all"],
                        U_whitened_cov_all=whitened["U_whitened_cov_all"],
                        U_cov_eigenvalues=whitened["eigenvalues"],
                        prediction_Tstd=pred, abs_prediction=np.abs(pred),
                        abs_Tstd=np.abs(tstd),
                        R=result["R"], D=result["D"], DR=result["DR"],
                        A_rawZ_to_Tstd=A_raw, b_rawZ_to_Tstd=intercept,
                        z_mean_all=zm, z_std_all=zs, theta_mean_all=tm,
                        theta_std_all=ts, scales=result["scales"],
                        full_sample_indices=np.arange(n))
    summary = {
        "model": model, "objective": objective, "sample_policy": "ALL subjects supervised; no holdout",
        "n_subjects": n, "latent_dim": d, "parameter_names": labels,
        "definition": "Tstd=(T-mean_all(T))/std_all(T); "
                      "Zstd=(Z-mean_all(Z))/std_all(Z) [ddof=0]; "
                      "U=Zstd @ R.T; prediction=U @ D; D diagonal; "
                      "DR maps Zstd to Tstd, A_raw=DR @ diag(1/std_all(Z))",
        "objective_value": result["loss"], "det_R": result["determinant_R"],
        "orthogonality_error_fro": float(np.linalg.norm(result["R"].T @ result["R"] - np.eye(d))),
        "best_start": result["best_start"], "attempts": result["attempts"],
        "R": result["R"].tolist(), "D": result["D"].tolist(), "DR": result["DR"].tolist(),
        "A_rawZ_to_Tstd": A_raw.tolist(),
        "b_rawZ_to_Tstd": intercept.tolist(),
        "post_rotation_whitening": {
            "method": "full-sample ZCA symmetric inverse covariance square root",
            "input": "U=Zstd @ R.T (before D); Zstd is all-subject z-score",
            "uses_true_T": False,
            "purpose": "diagnostic; whitening never participates in fitting R or D",
            "whitening_matrix": whitened["whitening_matrix"].tolist(),
            "U_mean_all": whitened["U_mean_all"].tolist(),
            "U_cov_all": whitened["U_cov_all"].tolist(),
            "U_whitened_cov_all": whitened["U_whitened_cov_all"].tolist(),
            "U_cov_eigenvalues": whitened["eigenvalues"].tolist(),
            "effective_rank": whitened["effective_rank"],
            "eigenvalue_cutoff": whitened["eigenvalue_cutoff"],
            "rank_deficient_note": (
                "Truncated pseudoinverse whitening: covariance not identity"
                if whitened["effective_rank"] < d else
                "Full-rank whitening: covariance approximately identity"),
            "U_whitened_Tstd_corr": corr_matrices["U_whitened_Tstd"][0].tolist(),
        },
        "z_mean_all": zm.tolist(), "z_std_all": zs.tolist(),
        "z_scaling": "each raw Z column standardized with all-subject ddof=0 moments BEFORE R",
        "theta_mean_all": tm.tolist(),
        "theta_std_all": ts.tolist(), "axis_scales": result["scales"].tolist(),
        "magnitude_MSE_mean": float(loss_abs.mean()),
        "signed_MSE_mean": float(loss_signed.mean()),
        "magnitude_R2_mean": float(abs_r2.mean()),
        "signed_R2_mean": float(signed_r2.mean()),
        "magnitude_correlation_diag": np.diag(cross_abs).tolist(),
        "signed_correlation_diag": np.diag(cross_signed).tolist(),
        "mean_abs_signed_diag_r": float(np.abs(np.diag(cross_signed)).mean()),
        "max_offdiag_abs_signed_r": float(np.max(np.abs(cross_signed[~np.eye(d, dtype=bool)]))),
        "correlations": {k: values[0].tolist() for k, values in corr_matrices.items()},
        "provenance": provenance or {},
        "interpretation": (
            "This is an in-sample T-supervised upper bound, NOT an unsupervised axis discovery "
            "nor held-out performance. For abs_mse, signs of the rows are unidentifiable "
            "and selected post hoc from signed covariances. "
            "High magnitude correlation alone does not prove signed parameter recovery."
        ),
    }
    # Numerical outputs are complete before attempting any PNG writes.
    # A single Windows/Pillow filesystem error must not discard model results.
    def write_summary():
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8")

    summary["plot_files"] = {}
    summary["plot_errors"] = {}
    write_summary()
    if plots:
        work = [(f"corr__{key}.png", matrix, rows, columns,
                 f"{model} FULL SAMPLE | {key}", True)
                for key, (matrix, rows, columns) in corr_matrices.items()]
        work.extend([
            ("R.png", result["R"], axes, zstd_names,
             f"{model} supervised orthogonal R on Z scores", False),
            ("D.png", result["D"], axes, axes,
             f"{model} fitted diagonal D", False),
            ("DR.png", result["DR"], axes, zstd_names,
             f"{model} complete DR map on Z scores", False),
        ])
        for filename, matrix, rows, cols, title, correlation in work:
            try:
                actual_path, fallback_reason = plot_heatmap(
                    out / filename, matrix, rows, cols, title,
                    correlation=correlation)
                summary["plot_files"][filename] = actual_path.name
                if fallback_reason:
                    print(f"[PLOT FALLBACK] {filename} -> {actual_path.name}: "
                          f"{fallback_reason}", flush=True)
            except OSError as exc:
                summary["plot_errors"][filename] = str(exc)
                print(f"[PLOT WARNING] {filename}: {exc}", flush=True)
        write_summary()
        if summary["plot_errors"]:
            print("Numerical output saved; some PNG plots could not be written. "
                  "See summary.json -> plot_errors.", flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=DIMS)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--lambda-sparse", type=float, default=0.)
    p.add_argument("--context-min", type=float, default=.25)
    p.add_argument("--context-max", type=float, default=.8)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--objective", choices=("abs_mse", "signed_mse"), default="abs_mse")
    p.add_argument("--restarts", type=int, default=6)
    p.add_argument("--maxiter", type=int, default=350)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--output", type=Path)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    dim = DIMS[args.model]
    if args.latent_dim is None:
        args.latent_dim = dim
    if args.beta is None:
        args.beta = .00005 if args.model == "CR3" else .0005
    if args.latent_dim != dim or args.beta < 0 or args.lambda_sparse < 0 or args.repeat < 1:
        p.error("Invalid NP configuration")
    if not (0 < args.context_min <= args.context_max <= 1):
        p.error("Invalid context fractions")
    if args.restarts < 0 or args.maxiter < 1:
        p.error("Invalid optimizer settings")
    z, t, provenance = load_existing_np(args.model, args.beta, dim, args.lambda_sparse,
                                        args.context_min, args.context_max, args.repeat)
    output = args.output or (ROOT / "outputs" / "supervised_abs_rotation" / args.model /
                            condition(args.beta, args.lambda_sparse,
                                      args.context_min, args.context_max, dim) /
                            f"repeat{args.repeat:02d}" / args.objective)
    summary = analyze(z, t, model=args.model, output=output, objective=args.objective,
                      seed=args.seed, restarts=args.restarts, maxiter=args.maxiter,
                      plots=not args.no_plots, overwrite=args.overwrite,
                      provenance=provenance)
    print(f"[{args.model}] ALL N={len(z)} | {args.objective}={summary['objective_value']:.6f}")
    print(f"  magnitude mean r2 = {summary['magnitude_R2_mean']:.4f}"
          f" | signed mean r2 = {summary['signed_R2_mean']:.4f}")
    print(f"  mean |signed diag r| = {summary['mean_abs_signed_diag_r']:.4f}"
          f" | max signed offdiag |r| = {summary['max_offdiag_abs_signed_r']:.4f}")
    print(f"  output: {output}")


if __name__ == "__main__":
    main()
