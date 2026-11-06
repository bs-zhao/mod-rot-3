"""Package-local experiment I/O, FIT ZXY preparation and post-fit evaluation.

This file is intentionally bundled with the individual rotation package. Other
rotation packages are never imported, registered, scheduled, or executed here.
The *only* external project inputs are NP's frozen S1 dataset and S5 Z.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # repo root: /rotations/<this_package>/experiment.py
PACKAGE = HERE.name


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_from_file() -> dict:
    path = HERE / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Each rotation package needs config.json: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config.json must be a JSON object")
    return config


def cli(description: str, argv=None) -> tuple[argparse.Namespace, dict]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", required=True, choices=("C1", "C2", "C3", "CR3"))
    parser.add_argument("--condition", required=True, help="NP S3/S5 condition, e.g. beta0_lambda0_ctx0p25to0p8_L8")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--representation", choices=("pca", "full"), default="pca")
    parser.add_argument("--rotation-dim", type=int, default=None)
    parser.add_argument("--fit-max-subjects", type=int, default=2500)
    parser.add_argument("--whiten-floor", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--set", action="append", default=[], metavar="PARAM=JSON_VALUE",
                        help="Override this package's config.json, e.g. --set maxiter=300")
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.fit_max_subjects < 1 or args.whiten_floor <= 0:
        parser.error("repeat, fit-max-subjects, whiten-floor must be positive")
    config = config_from_file()
    if "seed" in config:
        config["seed"] = args.seed
    for entry in args.set:
        key, sep, raw = entry.partition("=")
        if not sep or key not in config:
            parser.error(f"Unknown package parameter {key!r}; allowed: {sorted(config)}")
        try:
            config[key] = json.loads(raw)
        except json.JSONDecodeError:
            # On Windows CMD, quotes in JSON strings are often stripped.
            # Allow plain words ONLY for existing string-valued settings;
            # keep strict JSON for numbers, lists, booleans and objects.
            if isinstance(config[key],str) and raw.strip():
                config[key] = raw.strip()
            else:
                parser.error(f"Expected valid JSON for {key}: {raw}")
    return args, config


def aligned_xy(dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subject x trial shape, with false mask excluding variable-length padding."""
    counts = np.asarray(dataset.trial_counts, dtype=int)
    if counts.ndim != 1 or np.any(counts < 1):
        raise ValueError("Expected at least one observed trial per subject")
    n, t, d = len(counts), int(counts.max()), int(dataset.x.shape[1])
    indices = np.repeat(np.arange(n), counts)
    if not np.array_equal(np.asarray(dataset.subject_index), indices):
        raise ValueError("Dataset subject_index is not in original contiguous order")
    if len(dataset.x) != len(indices) or len(dataset.y) != len(indices):
        raise ValueError("Trial arrays do not match subject trial counts")
    x = np.zeros((n, t, d), dtype=np.float32)
    y = np.zeros((n, t), dtype=np.int64)
    mask = np.zeros((n, t), dtype=bool)
    offsets = np.r_[0, np.cumsum(counts)]
    for i, count in enumerate(counts):
        start, end = int(offsets[i]), int(offsets[i + 1])
        x[i, :count] = dataset.x[start:end]
        y[i, :count] = np.asarray(dataset.y[start:end]).reshape(-1)
        mask[i, :count] = True
    if not np.all(np.isin(y[mask], [0, 1])):
        raise ValueError("Rotation accepts only binary choice Y")
    return x, y, mask


def load_frozen(args) -> dict:
    """Load only NP's finished Z and S1 observations; theta never passed to fit."""
    folder = ROOT / "outputs" / "choice" / args.model
    latent = folder / "s4_evaluate" / args.condition / f"repeat{args.repeat:02d}" / "s4_evaluate__latent.npz"
    observed = folder / "s1_generate" / "s1_generate__dataset.npz"
    if not latent.is_file():
        raise FileNotFoundError(f"Run np/s5_extract_z.py first: {latent}")
    if not observed.is_file():
        raise FileNotFoundError(f"Run np/s1_generate.py first: {observed}")
    dataset_hash = sha256(observed)
    with np.load(latent, allow_pickle=False) as file:
        if str(file["context_mode"]) != "all_observed_trials":
            raise RuntimeError("Expected S5 Z extracted from ALL observed trials")
        if str(file["dataset_sha256"]) != dataset_hash:
            raise RuntimeError("S1 dataset changed after S5 Z extraction")
        z = np.asarray(file["z"], dtype=float)
        train = np.asarray(file["train_ids"], dtype=int)
        validation = np.asarray(file["validation_ids"], dtype=int)
        test = np.asarray(file["test_ids"], dtype=int)
        checkpoint_hash = str(file["checkpoint_sha256"])
    if (z.ndim != 2 or not np.isfinite(z).all() or
            not np.array_equal(np.sort(np.r_[train, validation, test]), np.arange(len(z)))):
        raise ValueError("Invalid S5 latent shape or subject splits")
    k_native = z.shape[1]
    k = args.rotation_dim or (k_native if args.representation == "full"
                               else {"C1": 4, "C2": 4, "C3": 6, "CR3": 4}[args.model])
    if not 2 <= k <= k_native or (args.representation == "full" and k != k_native):
        raise ValueError("rotation-dim must be 2..latent_dim; full requires all native dimensions")
    pca = PCA(n_components=k, svd_solver="full").fit(z[train])
    W = pca.components_ / np.sqrt(np.maximum(pca.explained_variance_, args.whiten_floor))[:, None]
    q = (z - pca.mean_) @ W.T
    selected = np.sort(np.random.default_rng(args.seed).choice(
        train, size=min(len(train), args.fit_max_subjects), replace=False))
    # Dataset is loaded to obtain X/Y; its true parameters are never passed to fit().
    from modlearn_rot.generators.dataset import load_dataset
    dataset = load_dataset(observed)
    X, Y, mask = aligned_xy(dataset)
    if len(X) != len(z) or np.any(mask.sum(axis=1) != np.asarray(dataset.trial_counts)):
        raise RuntimeError("S1/S5 participant alignment mismatch")
    q_fit, x_fit, y_fit, mask_fit = (np.array(arr[selected], copy=True)
                                    for arr in (q, X, Y, mask))
    for array in (q_fit, x_fit, y_fit, mask_fit):
        array.setflags(write=False)
    return dict(z=z, q_fit=q_fit, x_fit=x_fit, y_fit=y_fit, mask_fit=mask_fit,
                W=W, pca=pca, chosen=selected, tr=train, val=validation, te=test,
                dataset=dataset, latent_sha=sha256(latent), dataset_sha=dataset_hash,
                checkpoint_sha=checkpoint_hash, model=args.model, condition=args.condition,
                native_k=k_native, rotation_dim=k, output_root=folder)


def check_R(R, k: int) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    if R.shape != (k, k) or not np.isfinite(R).all():
        raise ValueError(f"Method returned invalid {k}x{k} rotation matrix")
    if not np.allclose(R @ R.T, np.eye(k), atol=1e-7, rtol=0):
        raise ValueError("R is not orthogonal")
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-7):
        raise ValueError("R must be in SO(K) (det=+1)")
    return R


def corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)
    a -= a.mean(axis=0)
    b -= b.mean(axis=0)
    d = np.sqrt((a * a).sum(axis=0)[:, None] * (b * b).sum(axis=0)[None, :])
    return np.divide(a.T @ b, d, out=np.zeros((a.shape[1], b.shape[1])), where=d > 1e-12)


def evaluate_after_fit(theta, U, test, folder: Path, model: str, package: str) -> dict:
    """True theta accessed ONLY after R,U, fit metadata have been written."""
    if model == "CR3":
        names = ["alpha", "lambda_loss", "gamma", "tau"]
    else:
        from modlearn_rot.generators.registry import get_model
        names = list(get_model(model).parameter_names)
    c = corr(np.asarray(theta)[test], np.asarray(U)[test])
    ri, ci = linear_sum_assignment(-(c ** 2))
    match = dict(zip((int(v) for v in ri), (int(v) for v in ci)))
    chosen = np.array([c[i, match[i]] ** 2 for i in range(len(names)) if i in match])
    off = np.ones(c.shape, dtype=bool)
    for i, j in match.items():
        off[i, j] = False
    report = dict(package=package, method=package, parameter_names=names,
                  matched_mean_r2=float(chosen.mean()) if len(chosen) else None,
                  matched_min_r2=float(chosen.min()) if len(chosen) else None,
                  parameter_r2={n: (float(c[i, match[i]] ** 2) if i in match else None)
                                for i, n in enumerate(names)},
                  max_offtarget_abs_r=float(np.abs(c[off]).max()) if off.any() else 0.0,
                  mean_offtarget_r2=float(np.mean(c[off] ** 2)) if off.any() else 0.0,
                  matching=match, theta_axis_corr=c.tolist())
    (folder / "rotation__test_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(max(6, c.shape[1] * .8), max(4, c.shape[0] * .8)))
    im = ax.imshow(c ** 2, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(c.shape[1]), [f"u{j+1}" for j in range(c.shape[1])])
    ax.set_yticks(range(c.shape[0]), names)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            val = c[i, j] ** 2
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if val > .55 else "black")
    from matplotlib.patches import Rectangle
    for i, j in match.items():
        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                               fill=False, edgecolor="red", lw=2))
    fig.colorbar(im, ax=ax, label="TEST Pearson r²")
    fig.tight_layout()
    fig.savefig(folder / "rotation__test_corr2.png", dpi=170)
    plt.close(fig)
    return report


def package_fingerprint(source: dict, config: dict, args) -> str:
    h = hashlib.sha256()
    for path in sorted(HERE.rglob("*.py")):
        h.update(str(path.relative_to(HERE)).encode())
        h.update(path.read_bytes())
    payload = dict(package=PACKAGE, model=args.model, condition=args.condition,
                   repeat=args.repeat, representation=args.representation,
                   rotation_dim=source["rotation_dim"],
                   fit_max_subjects=args.fit_max_subjects, seed=args.seed,
                   whiten_floor=args.whiten_floor,
                   dataset_sha=source["dataset_sha"], latent_sha=source["latent_sha"],
                   checkpoint_sha=source["checkpoint_sha"], config=config,
                   fit_ids_sha=hashlib.sha256(source["chosen"].tobytes()).hexdigest(),
                   no_evaluate=args.no_evaluate)
    h.update(json.dumps(payload, sort_keys=True).encode())
    return h.hexdigest()[:16]


def execute(args, config: dict, fit: Callable, source: dict | None = None) -> Path:
    """Run just this rotation package; fit() receives q_fit,X_fit,Y_fit,mask_fit."""
    source = load_frozen(args) if source is None else source
    fingerprint = package_fingerprint(source, config, args)
    folder = (source["output_root"] / "rotations" / PACKAGE / args.condition /
              f"repeat{args.repeat:02d}" / f"run_{fingerprint}")
    if (folder / "rotation__complete.json").is_file() and not args.overwrite:
        print("[SKIP] Exact package/configuration already completed:", folder)
        return folder
    if folder.exists() and any(folder.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Incomplete previous run at {folder}. Inspect or use --overwrite")
    if args.dry_run:
        print("[PLAN]", PACKAGE, args.model, args.condition, config, folder)
        return folder
    # Nothing outside the local package's fit() performs rotation optimization.
    R, info, extra = fit(source["q_fit"], source["x_fit"], source["y_fit"],
                         source["mask_fit"], config)
    R = check_R(R, source["rotation_dim"])
    A = R @ source["W"]
    offset = -A @ source["pca"].mean_
    U = source["z"] @ A.T + offset
    if not np.allclose(U[source["chosen"]], source["q_fit"] @ R.T, atol=1e-6):
        raise RuntimeError("Z -> U affine map disagrees with the FIT rotation")
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "rotation__matrix.npz", R=R, A=A, offset=offset,
                        pca_components=source["pca"].components_,
                        pca_eigenvalues=source["pca"].explained_variance_,
                        pca_mean=source["pca"].mean_, fit_ids=source["chosen"],
                        source_z_sha256=np.array(source["latent_sha"]),
                        source_dataset_sha256=np.array(source["dataset_sha"]),
                        package=np.array(PACKAGE))
    np.savez_compressed(folder / "rotation__U.npz", U=U, train_ids=source["tr"],
                        validation_ids=source["val"], test_ids=source["te"])
    if extra:
        if any(np.asarray(value).dtype == object for value in extra.values()):
            raise ValueError("Method diagnostics must not contain object arrays")
        np.savez_compressed(folder / "rotation__method_arrays.npz", **extra)
    meta = dict(package=PACKAGE, model=args.model, condition=args.condition,
                representation=args.representation, rotation_dim=source["rotation_dim"],
                native_latent_dim=source["native_k"], fit_subject_count=len(source["chosen"]),
                config=config, objective_fit=info,
                source_z_sha256=source["latent_sha"],
                source_dataset_sha256=source["dataset_sha"],
                source_checkpoint_sha256=source["checkpoint_sha"], fingerprint=fingerprint)
    (folder / "rotation__fit.json").write_text(json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
    if not args.no_evaluate:
        # Snapshot ONLY held-out TEST truths, after FIT has persisted R/U.
        # This supports entirely local package ablations without reloading
        # simulator data or ever passing theta to FIT/MDL selection.
        true_theta=np.asarray(source["dataset"].parameters)
        np.savez_compressed(folder/"rotation__test_truth.npz",
                            theta_test=true_theta[source["te"]],test_ids=source["te"])
        report = evaluate_after_fit(true_theta, U, source["te"],
                                    folder, args.model, PACKAGE)
        print(f"[OK] {PACKAGE}: TEST matched mean r²={report['matched_mean_r2']:.4f}")
    (folder / "rotation__complete.json").write_text(
        json.dumps(dict(fingerprint=fingerprint, evaluated=not args.no_evaluate), indent=2),
        encoding="utf-8")
    print("Saved:", folder)
    return folder