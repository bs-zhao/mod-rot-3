"""Wave H: prioritize effective support over dominant-entry concentration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

import run as wave
import run_finite
import tail_support as ts


def source_moments(g, h, x, pooling):
    covariance = ts.moment_covariance(h, pooling)
    scale = np.std(x, axis=0); scale[scale < 1e-10] = 1.
    neighbors = cKDTree(x/scale).query(x/scale, k=min(4, len(x)))[1]
    edges = sorted({tuple(sorted((i, int(j)))) for i in range(len(x)) for j in neighbors[i, 1:] if i != j})
    a, b = np.asarray(edges).T
    delta = g[:, a]-g[:, b]
    energy = np.sum(delta**2, axis=-1)
    if pooling == "context":
        delta = delta/np.maximum(np.sqrt(energy.mean(0))[None, :, None], 1e-100)
    elif pooling == "point":
        delta = delta/np.maximum(np.sqrt(energy)[..., None], 1e-100)
    ux = np.einsum("nei,nej->eij", delta, delta, optimize=True)/len(delta)
    return covariance, ux


def evaluate_moments(covariance, ux, q):
    eu = ts.uu_energy(covariance, q)
    ex = np.maximum(np.einsum("ai,eab,bi->ie", q, ux, q, optimize=True), 0.)
    d = len(q); ii, jj = np.triu_indices(d, 1)
    groups = dict(UU_all=np.r_[np.diag(eu), 2*eu[ii, jj]], UU_edges=eu[ii, jj],
                  U_curvature=np.diag(eu), UX_entries=ex)
    out = {}
    for group, energy in groups.items():
        for metric, loss in ts.scores(energy).items():
            out[group+"_"+metric] = loss
    columns = [ts.scores(ex[:, k]) for k in range(ex.shape[1]) if ex[:, k].sum() > 1e-100]
    if columns:
        for metric in columns[0]:
            out["UX_columns_"+metric] = float(np.mean([v[metric] for v in columns]))
    out["control_UU_fro2"] = float(eu.sum())
    out["control_UX_contrast_fro2"] = float(ex.sum())
    return out


def analyze(source, fields, output, seed=387):
    source, fields, output = Path(source), Path(fields), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Retain old runs; choose a new output: {output}")
    old = json.loads((fields / "protocol.json").read_text())
    if old["status"] != "COMPLETE" or hashlib.sha256((fields / "blind_scores.csv").read_bytes()).hexdigest() != old["scores_sha256"]:
        raise ValueError("Source experiment incomplete or frozen scores changed")
    data = wave.io.MatrixCases(source, tuple(old["channels"]), old["seed"])
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=data.index["model"], Z_dim=data.d, n_subjects=data.n,
        source=str(source.resolve()), seed=seed, source_run_name=fields.name,
        source_code_sha256=old["code_sha256"], estimators=old["estimators"], channels=list(data.channels),
        pooling=["global", "context", "point"], relative_L2_errors=[.01, .03, .1],
        effective_support_p=[.5, 1.], all_subjects_used=True,
        score_sign="lower for all datasets", prioritizes_support_over_single_entry_dominance=True,
        cr3_gate="same channel/metric must pass both polynomial and RBF families",
        status="RUNNING", code_sha256=hashlib.sha256(b"".join((wave.HERE / f).read_bytes() for f in
            ("run_tail_support.py", "tail_support.py", "matrix_io.py", "structure.py"))).hexdigest())
    wave.json_file(output / "protocol.json", config)
    rng = np.random.default_rng(seed)
    paths = [data.paths[i] for i in rng.permutation(len(data.paths))]
    maps, reveal = [], {}
    for k, path in enumerate(paths):
        q, _ = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); reveal[f"case_{k:03d}"] = path.parent.name
    with np.load(fields / "response_bank.npz", allow_pickle=False) as saved:
        x = saved["x"]
    with (fields / "fit_quality.csv").open(newline="") as stream:
        quality = {(r["estimator"], r["channel"]): r for r in csv.DictReader(stream)}
    blind, moments = [], {}
    for name in old["estimators"]:
        with np.load(fields / f"model_{name}.npz", allow_pickle=False) as saved:
            stim = saved["score_stimuli"]
        with np.load(fields / f"field_{name}.npz", allow_pickle=False) as saved:
            g, h = saved["gradient"], saved["hessian"]
        for c, channel in enumerate(data.channels):
            for pooling in ("global", "context", "point"):
                covariance, ux = source_moments(g[:, c], h[:, c], x[stim], pooling)
                moments[f"{name}_{channel}_{pooling}_UU"] = covariance
                moments[f"{name}_{channel}_{pooling}_UX"] = ux
                for k, q in enumerate(maps):
                    for metric, loss in evaluate_moments(covariance, ux, q).items():
                        blind.append(dict(anonymous_case=f"case_{k:03d}", estimator=name,
                            channel=channel, metric=pooling+"_"+metric, loss=loss))
        wave.csv_file(output / "blind_scores.csv", blind)
        print(f"[WaveH] {name}: all anonymous support scores saved", flush=True)
    np.savez_compressed(output / "sufficient_moments.npz", **moments)
    digest = hashlib.sha256((output / "blind_scores.csv").read_bytes()).hexdigest()
    wave.json_file(output / "reveal.json", dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row["estimator"], row["channel"], row["metric"]][reveal[row["anonymous_case"]]] = row["loss"]
    summaries = []
    for (name, channel, metric), values in grouped.items():
        control = "_control_" in metric
        if control and np.ptp(list(values.values())) > 1e-7*max(1., max(values.values())):
            raise ArithmeticError(f"Invariant covariance control changed: {name}/{channel}/{metric}")
        qr = quality[name, channel]
        summaries.append(dict(estimator=name, channel=channel, metric=metric,
            **wave.st.evaluate_curve(values, data.index["cases"]),
            adequate=qr["adequate"] == "True", validation_nmse=qr["validation_nmse"], is_control=control))
    wave.csv_file(output / "metric_summary.csv", summaries)
    wave.json_file(output / "read_audit.json", dict(files=sorted(set(data.read_files)),
        source_scores_sha256=old["scores_sha256"], forbidden_files_read=[],
        all_subjects=data.n, all_rows=data.m))
    config.update(status="COMPLETE", scores_sha256=digest)
    wave.json_file(output / "protocol.json", config)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=wave.io.DIMS, default="CR3")
    p.add_argument("--run-name", default="waveH_support_seed387")
    p.add_argument("--source-run", default="waveC_finite_seed383")
    p.add_argument("--seed", type=int, default=387)
    p.add_argument("--batch", action="store_true")
    args = p.parse_args()
    paths = {}
    with threadpool_limits(limits=4):
        for model in (("CR3", "C1", "C2", "C3") if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model)
            fields = source.parent / "step5_matrix_metrics" / args.source_run
            if not (fields / "protocol.json").exists():
                run_finite.analyze(source, fields)
            output = source.parent / "step5_matrix_metrics" / args.run_name
            paths[model] = output
            rows = analyze(source, fields, output, args.seed)
            passing = defaultdict(set)
            for r in rows:
                if r["adequate"] and r["all_planes_pass"] and not r["is_control"]:
                    passing[r["channel"], r["metric"]].add("poly" if r["estimator"].startswith("poly") else "rbf")
            robust = [key for key, families in passing.items() if families >= {"poly", "rbf"}]
            print(f"[WaveH] {model}: {len(robust)} cross-family support candidates", flush=True)
            if args.batch and model == "CR3" and not robust:
                print("[WaveH] CR3 screen failed; later models not launched", flush=True)
                return
        if args.batch:
            wave.compare(paths, wave.ROOT / "outputs" / "supervised_abs_rotation" / f"step5_{args.run_name}_four_models.csv")


if __name__ == "__main__":
    main()
