"""Step 2: exhaustive SINGLE-PAIR Givens sweeps around frozen Step-1 best R.

Exactly one plane is perturbed per case, never Cartesian products of planes.
U_s(theta) = G_ij(theta) @ R_best @ Zstd_s (column convention)
W_trial     = [U_subject[subject_index], X_original_trial].
Y is identical for every case and stored exactly once in the Step-2 root.

This script requires completed Step-1 /stage1_data and never trains or
runs a neural network. Use --dry-run to inspect expected disk usage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STEP1_CODE = HERE.parent
ROOT = STEP1_CODE.parent
sys.path.insert(0, str(STEP1_CODE))
from run import DIMS, condition

ANGLES = (15, 30, 45, 60, 75, 90)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json_atomic(path, data):
    path = Path(path)
    tmp = path.with_name("." + path.name + ".pending")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


@dataclass(frozen=True)
class Case:
    name: str
    pair: tuple[int, int] | None
    angle: int


def all_cases(dim):
    """1 baseline plus 6 angles times all distinct coordinate pairs."""
    if dim < 2:
        raise ValueError("At least two latent dimensions are needed")
    cases = [Case("baseline", None, 0)]
    for i, j in combinations(range(dim), 2):
        for angle in ANGLES:
            cases.append(Case(f"u{i+1}_u{j+1}_{angle:03d}", (i, j), angle))
    return cases


def givens(dim, pair, angle_deg):
    """Active 2D rotation on column vectors: [cos,-sin; sin,cos]."""
    g = np.eye(dim, dtype=np.float64)
    if pair is None:
        if angle_deg != 0:
            raise ValueError("Baseline must have zero angle")
        return g
    i, j = pair
    if not (0 <= i < j < dim) or angle_deg not in ANGLES:
        raise ValueError("Illegal pair or angle")
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    g[i, i], g[i, j] = c, -s
    g[j, i], g[j, j] = s, c
    return g


def orientation(best_r, case):
    """Baseline is best R; a case rotates ONE of its output U planes."""
    return givens(best_r.shape[0], case.pair, case.angle) @ best_r


def source_files(stage1):
    names = ("Zstd.npy", "R.npy", "W.npy", "Y.npz",
             "subject_index.npy", "trial_index.npy", "trial_counts.npy",
             "manifest.json")
    paths = {name: stage1 / name for name in names}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Step 1 export missing files; rerun Step 1:\n"
                                + "\n".join(missing))
    return paths


def inspect_stage1(stage1, model):
    """Strictly check row order, zscores, dimensions and Y alignment."""
    stage1 = Path(stage1)
    paths = source_files(stage1)
    meta = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    if meta["model"] != model:
        raise ValueError(f"Step 1 data are for {meta['model']}, not {model}")
    z = np.load(paths["Zstd.npy"], mmap_mode="r", allow_pickle=False)
    best_r = np.load(paths["R.npy"], mmap_mode="r", allow_pickle=False)
    base_w = np.load(paths["W.npy"], mmap_mode="r", allow_pickle=False)
    sid = np.load(paths["subject_index.npy"], mmap_mode="r", allow_pickle=False)
    tid = np.load(paths["trial_index.npy"], mmap_mode="r", allow_pickle=False)
    counts = np.load(paths["trial_counts.npy"], mmap_mode="r", allow_pickle=False)

    n, dz = z.shape
    m = base_w.shape[0]
    if dz != DIMS[model] or best_r.shape != (dz, dz):
        raise ValueError("Step 1 Z and R do not match the chosen model")
    if counts.shape != (n,) or sid.shape != (m,) or tid.shape != (m,):
        raise ValueError("Trial and subject indices do not agree with W rows")
    if base_w.ndim != 2 or base_w.shape[1] <= dz:
        raise ValueError("Step 1 W must consist of [Zstd, X_original]")
    if m != int(np.sum(counts)) or np.any(counts <= 0):
        raise ValueError("Original total trial count does not match W")
    if tuple(meta["w_shape"]) != tuple(base_w.shape):
        raise ValueError("Step 1 W shape differs from its manifest")
    if not np.allclose(best_r @ best_r.T, np.eye(dz), atol=1e-8):
        raise ValueError("Step 1 best R is not orthogonal")
    if not np.isfinite(z).all() or not np.isfinite(best_r).all():
        raise ValueError("Nonfinite Zstd or best R")
    if not np.allclose(z.mean(axis=0), 0.0, atol=1e-8):
        raise ValueError("Zstd must have all-subject zero mean")
    if not np.allclose(z.std(axis=0, ddof=0), 1.0, atol=1e-6):
        raise ValueError("Zstd must have all-subject unit standard deviation")

    expected_sid = np.repeat(np.arange(n), counts)
    if not np.array_equal(sid, expected_sid):
        raise ValueError("Step 1 W trial subject mapping is not in S1 order")
    offsets = np.r_[0, np.cumsum(counts)[:-1]]
    expected_tid = np.arange(m) - np.repeat(offsets, counts)
    if not np.array_equal(tid, expected_tid):
        raise ValueError("Step 1 trial indices are not in the S1 order")

    # Validate Zstd replication throughout the original W in bounded chunks.
    for start in range(0, m, 200_000):
        end = min(start + 200_000, m)
        actual = base_w[start:end, :dz]
        expected = z[sid[start:end]].astype(np.float32)
        if not np.allclose(actual, expected, rtol=2e-6, atol=1e-6):
            raise ValueError(f"Step 1 W and Zstd disagree at rows {start}:{end}")
    with np.load(paths["Y.npz"], allow_pickle=False) as y:
        required = ("choice_observed", "logit_nn", "p_nn",
                    "choice_nn", "logits_nn_2class")
        for key in required:
            if key not in y:
                raise ValueError(f"Step 1 Y.npz lacks {key}")
            if y[key].shape[0] != m:
                raise ValueError(f"Step 1 Y[{key}] does not match {m} trial rows")
        if y["logits_nn_2class"].shape != (m, 2):
            raise ValueError("Expected two frozen decoder logits per trial")

    fingerprint_files = ("Zstd.npy", "R.npy", "W.npy", "Y.npz",
                         "subject_index.npy", "trial_index.npy",
                         "trial_counts.npy", "manifest.json")
    fingerprints = {name: sha256(paths[name]) for name in fingerprint_files}
    run_fingerprint = hashlib.sha256(
        json.dumps(fingerprints, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "paths": paths, "metadata": meta, "Zstd": z, "best_R": best_r,
        "W": base_w, "subject_index": sid, "trial_index": tid,
        "counts": counts, "n": n, "m": m, "dz": dz,
        "dx": base_w.shape[1] - dz,
        "fingerprints": fingerprints,
        "fingerprint": run_fingerprint,
    }


def make_case(stage2, src, case, *, chunk_rows, fingerprint, overwrite):
    """Generate a complete W file in bounded memory, then publish atomically."""
    case_dir = stage2 / "cases" / case.name
    best_r = np.asarray(src["best_R"], dtype=np.float64)
    r_case = orientation(best_r, case)
    u_subject = np.asarray(src["Zstd"], dtype=np.float64) @ r_case.T
    dz, dx, m = src["dz"], src["dx"], src["m"]

    if case_dir.exists() and not overwrite:
        case_meta_path = case_dir / "case.json"
        if not case_meta_path.is_file():
            raise RuntimeError(f"Partial case directory without metadata: {case_dir}")
        recorded = json.loads(case_meta_path.read_text(encoding="utf-8"))
        w_path = case_dir / "W.npy"
        r_path = case_dir / "R.npy"
        if (recorded.get("source_fingerprint") != fingerprint
                or recorded.get("angle_deg") != case.angle
                or recorded.get("pair_0based") !=
                    (list(case.pair) if case.pair is not None else None)
                or not w_path.is_file() or not r_path.is_file()):
            raise RuntimeError(f"Existing case is inconsistent; rerun with --overwrite: {case_dir}")
        old_w = np.load(w_path, mmap_mode="r", allow_pickle=False)
        old_r = np.load(r_path, allow_pickle=False)
        if old_w.shape != (m, dz + dx) or not np.allclose(old_r, r_case, atol=1e-11):
            raise RuntimeError(f"Existing case has wrong geometry; --overwrite: {case_dir}")
        return "reused"

    (stage2 / "cases").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".step2_case_", dir=stage2 / "cases") as temp:
        pending = Path(temp) / case.name
        pending.mkdir()
        np.save(pending / "R.npy", r_case)
        np.save(pending / "U_subject.npy", u_subject)
        w = np.lib.format.open_memmap(
            pending / "W.npy", mode="w+", dtype=np.float32, shape=(m, dz + dx))
        base_w = src["W"]
        sid = src["subject_index"]
        for start in range(0, m, chunk_rows):
            end = min(start + chunk_rows, m)
            w[start:end, :dz] = u_subject[sid[start:end]].astype(np.float32)
            w[start:end, dz:] = base_w[start:end, dz:]
        w.flush()
        del w
        manifest = {
            "case": case.name,
            "source_fingerprint": fingerprint,
            "angle_deg": case.angle,
            "pair_0based": list(case.pair) if case.pair is not None else None,
            "pair_1based": [k + 1 for k in case.pair] if case.pair is not None else None,
            "definition": "R_case=G_ij(theta)@R_best; U=Zstd@R_case.T",
            "W_columns": [f"u{k+1}" for k in range(dz)]
                         + [f"x{k+1}" for k in range(dx)],
            "W_shape": [m, dz + dx],
            "W_dtype": "float32",
            "Y_path": "../../Y.npz",
            "original_X_unchanged": True,
            "NN_Y_never_recomputed": True,
        }
        (pending / "case.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        if case_dir.exists():
            backup = (stage2 / "cases" / (".backup_" + case.name))
            if backup.exists():
                raise FileExistsError(f"Stale case backup: {backup}")
            os.replace(case_dir, backup)
            try:
                os.replace(pending, case_dir)
            except BaseException:
                os.replace(backup, case_dir)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(pending, case_dir)
    return "written"


def run_sweep(stage1_dir, stage2_dir, *, model, chunk_rows=100_000,
              overwrite=False, dry_run=False, only_case=None):
    """No model fitting, no decoder calls; default builds ALL pair-angle cases."""
    if chunk_rows < 1:
        raise ValueError("--chunk-rows must be positive")
    src = inspect_stage1(Path(stage1_dir), model)
    cases = all_cases(src["dz"])
    names = {c.name for c in cases}
    if only_case is not None and only_case not in names:
        raise ValueError(f"Unknown --only-case {only_case!r}; choose from {sorted(names)}")
    selected = [c for c in cases if only_case is None or c.name == only_case]
    stage2 = Path(stage2_dir)

    n_cases = len(cases)
    per_w_bytes = (src["m"] * (src["dz"] + src["dx"]) * 4 + 128)
    info = {
        "model": model, "n_subjects": src["n"], "n_trials": src["m"],
        "Z_dim": src["dz"], "X_dim": src["dx"],
        "n_pairs": src["dz"] * (src["dz"] - 1) // 2,
        "angles_deg": list(ANGLES), "n_cases_all": n_cases,
        "n_cases_selected": len(selected),
        "size_each_W_gib": round(per_w_bytes / 1024**3, 4),
        "size_all_W_gib": round(n_cases * per_w_bytes / 1024**3, 3),
        "source_fingerprint": src["fingerprint"],
    }
    print(f"[Step 2] {model}: {info['n_pairs']} planes x {len(ANGLES)} angles "
          f"+ baseline = {n_cases} cases; {src['m']:,} trial rows per W.",
          flush=True)
    print(f"[Step 2] W storage ~{info['size_each_W_gib']:.3f} GiB/case, "
          f"~{info['size_all_W_gib']:.2f} GiB for all W; Y once.", flush=True)
    if dry_run:
        return info

    stage2.mkdir(parents=True, exist_ok=True)
    index_path = stage2 / "index.json"
    if index_path.is_file():
        prior = json.loads(index_path.read_text(encoding="utf-8"))
        if prior.get("source_fingerprint") != src["fingerprint"] and not overwrite:
            raise RuntimeError(
                "Step 1 arrays changed after this Step 2 export; "
                "pass --overwrite after inspecting previous output")

    for name in ("Y.npz", "Zstd.npy", "R_best.npy",
                 "subject_index.npy", "trial_index.npy", "trial_counts.npy"):
        source_name = "R.npy" if name == "R_best.npy" else name
        origin = src["paths"][source_name]
        target = stage2 / name
        if target.is_file() and not overwrite:
            if sha256(target) != src["fingerprints"][source_name]:
                raise RuntimeError(f"Shared {target} does not match Step 1; --overwrite")
            continue
        tmp = stage2 / ("." + name + ".pending")
        shutil.copyfile(origin, tmp)
        os.replace(tmp, target)

    index = {
        **info,
        "source_stage1": str(Path(stage1_dir).resolve()),
        "output_stage2": str(stage2.resolve()),
        "definition": "For case pair (i,j), angle a: R_case=G_ij(a)@R_best; U=Zstd@R_case.T; W=[U[subject_index],X_original]",
        "zero_angle": "baseline=R_best (not raw identity); do not apply multiple planes simultaneously",
        "angles_orientation": "G_ii=cos, G_ij=-sin, G_ji=sin, G_jj=cos (active column rotation)",
        "Y_shared": "Y.npz in stage2 root; unchanged from stage1",
        "original_trial_order": "identical to Step 1",
        "source_files_sha256": src["fingerprints"],
        "cases": [{
            "id": c.name,
            "pair_1based": [i+1 for i in c.pair] if c.pair else None,
            "angle_deg": c.angle,
            "relative_dir": f"cases/{c.name}",
        } for c in cases],
        "complete": False,
    }
    write_json_atomic(index_path, index)
    table_path = stage2 / "cases.csv"
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=(
            "case", "dimension_1", "dimension_2", "angle_deg", "R_path",
            "W_path", "Y_shared_path"))
        writer.writeheader()
        for case in cases:
            writer.writerow({
                "case": case.name,
                "dimension_1": "" if case.pair is None else case.pair[0] + 1,
                "dimension_2": "" if case.pair is None else case.pair[1] + 1,
                "angle_deg": case.angle,
                "R_path": f"cases/{case.name}/R.npy",
                "W_path": f"cases/{case.name}/W.npy",
                "Y_shared_path": "Y.npz",
            })

    for idx, case in enumerate(selected, start=1):
        status = make_case(stage2, src, case, chunk_rows=chunk_rows,
                           fingerprint=src["fingerprint"], overwrite=overwrite)
        print(f"  [{idx:3}/{len(selected)}] {case.name}: {status}", flush=True)

    # Report completion only if all cases have been generated, even if only_case
    # was used to split expensive work across multiple invocations.
    index["complete"] = all(
        (stage2 / "cases" / case.name / "case.json").is_file()
        and (stage2 / "cases" / case.name / "W.npy").is_file()
        for case in cases)
    write_json_atomic(index_path, index)
    print(f"[Step 2] {model}: {'complete' if index['complete'] else 'partial'} "
          f"at {stage2}", flush=True)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=DIMS, required=True)
    parser.add_argument("--objective", choices=("abs_mse", "signed_mse"),
                        default="abs_mse")
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--lambda-sparse", type=float, default=0.0)
    parser.add_argument("--context-min", type=float, default=.25)
    parser.add_argument("--context-max", type=float, default=.8)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--stage1-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate case W files (otherwise valid cases are reused)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and estimate storage without writing")
    parser.add_argument("--only-case", type=str,
                        help="Optional one case, e.g. baseline or u1_u2_045")
    args = parser.parse_args(argv)
    dim = DIMS[args.model]
    if args.latent_dim is None:
        args.latent_dim = dim
    if args.beta is None:
        args.beta = .00005 if args.model == "CR3" else .0005
    if (args.latent_dim != dim or args.beta < 0 or args.lambda_sparse < 0
            or args.repeat < 1 or args.chunk_rows < 1 or
            not 0 < args.context_min <= args.context_max <= 1):
        parser.error("Invalid configuration")
    step1 = (ROOT / "outputs" / "supervised_abs_rotation" / args.model /
             condition(args.beta, args.lambda_sparse, args.context_min,
                       args.context_max, dim) /
             f"repeat{args.repeat:02d}" / args.objective)
    stage1_dir = args.stage1_dir or (step1 / "stage1_data")
    output_dir = args.output or (step1 / "step2_pairwise_sweep")
    run_sweep(stage1_dir, output_dir, model=args.model,
              chunk_rows=args.chunk_rows, overwrite=args.overwrite,
              dry_run=args.dry_run, only_case=args.only_case)


if __name__ == "__main__":
    main()
