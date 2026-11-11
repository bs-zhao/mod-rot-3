"""Step 5, wave A: all-row numerical surrogates and UU/UX structural metrics.

No theta, formula, decoder, saved R, or case angle enters any score.
No claim of success can be made before real four-model results are present.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def dependency(name):
    spec = importlib.util.spec_from_file_location("step5_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


io = dependency("matrix_io")
sur = dependency("surrogates")
st = dependency("structure")


def json_file(path, value):
    path = Path(path)
    pending = path.with_name("." + path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(pending, path)


def csv_file(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def analyze(folder, output, *, channels=("logit_nn", "p_nn"), seed=381,
            centers=128, per_subject=8, chunk=8192, models=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a NEW --output to preserve earlier trials: {output}")
    data = io.MatrixCases(folder, channels, seed)
    if data.n < 20:
        raise ValueError("At least 20 subjects required for this experiment")
    output.mkdir(parents=True, exist_ok=True)
    requested = models or ("poly2", "poly3", "rbf0.5", "rbf1", "rbf2")
    if len(set(requested)) != len(requested):
        raise ValueError("Repeated estimators are not independent evidence")
    config = dict(model=data.index.get("model"), n_subjects=data.n, Z_dim=data.d,
                  seed=seed, centers=centers, per_subject=per_subject,
                  chunk=chunk, estimators=list(requested), channels=list(channels),
                  ridge_grid=[1e-6, 1e-4, 1e-2], heldout_fraction=.25,
                  adequacy_validation_nmse=.10, source=str(Path(folder).resolve()),
                  score_sign="lower is simpler for ALL models", all_subjects_used=True,
                  derivative_accuracy_verified=False,
                  code_sha256=hashlib.sha256(b"".join(
                      (HERE / file).read_bytes() for file in
                      ("matrix_io.py", "surrogates.py", "structure.py", "run.py"))).hexdigest(),
                  status="RUNNING", read_policy="W + common Y + row-layout metadata only")
    json_file(output / "protocol.json", config)
    rng = np.random.default_rng(seed)
    train_subjects = np.sort(rng.permutation(data.n)[:int(.75*data.n)])
    # Centers are selected from training subjects only. This avoids using
    # validation coordinates to adapt the kernel's feature representation.
    eligible_rows = np.flatnonzero(np.isin(data.sid, train_subjects))
    center_ids = np.sort(rng.choice(eligible_rows, min(centers, len(eligible_rows)), replace=False))
    scaler = sur.Scaler(data.w, data.d, row_mask=np.isin(data.sid, train_subjects))
    rows = data.evaluation_rows(per_subject)
    evaluation = np.asarray(data.w[rows], dtype=float)
    # In unequal trial-count designs each subject still gets equal weight.
    counts = np.bincount(data.sid[rows], minlength=data.n)
    weights = 1. / counts[data.sid[rows]]
    maps = []
    for i, path in enumerate(data.paths):
        candidate_u = data.coordinates(path)
        q, residual = io.registration(data.u, candidate_u)
        maps.append((f"case_{i:03d}", q, residual))
    quality = []; blind = []
    for name in requested:
        if name in ("poly2", "poly3"):
            model = sur.Polynomial(scaler, int(name[-1]))
        elif name.startswith("rbf"):
            model = sur.RBF(scaler, data.w[center_ids], float(name[3:]))
        else:
            raise ValueError(f"Unknown estimator: {name}")
        print(f"[Step5] {name}: fitting ALL {data.m:,} rows / {data.n:,} subjects", flush=True)
        fit = sur.fit_all_rows(model, data.w, data.y, data.sid, train_subjects, chunk)
        for c, channel in enumerate(channels):
            quality.append(dict(estimator=name, channel=channel,
                                validation_nmse=fit["validation_nmse"][c],
                                adequate=fit["validation_nmse"][c] <= .10,
                                ridge=fit["ridge"], feature_count=fit["feature_count"],
                                final_fit_subjects=data.n, final_fit_rows=data.m,
                                derivative_accuracy_verified=False))
        json_file(output / f"fit_{name}.json", fit)
        g = np.empty((len(rows), len(channels), data.d + data.dx))
        h = np.empty((*g.shape, data.d + data.dx))
        for a in range(0, len(rows), chunk):
            b = min(a + chunk, len(rows))
            gg, hh = model.derivatives(evaluation[a:b])
            # Standardized coordinate units, common U scale for every case.
            g[a:b] = gg * scaler.scale
            h[a:b] = hh * scaler.scale[None, None, :, None] * scaler.scale[None, None, None, :]
        np.savez_compressed(output / f"field_{name}.npz", gradient=g, hessian=h,
                            evaluation_rows=rows, subject_ids=data.sid[rows], weights=weights)
        for anonymous, q, residual in maps:
            for c, channel in enumerate(channels):
                gg, hh = st.rotate_fields(g[:, c], h[:, c], q, data.d)
                scores = st.structural_scores(gg, hh, data.d, weights)
                for metric, loss in scores.items():
                    blind.append(dict(anonymous_case=anonymous, estimator=name,
                                      channel=channel, metric=metric, loss=loss))
        csv_file(output / "blind_scores.csv", blind)
        csv_file(output / "fit_quality.csv", quality)
        del g, h
        print(f"[Step5] {name}: anonymous scores saved", flush=True)

    # Freeze all numerical scores before exposing case names to the evaluator.
    scores_sha256 = hashlib.sha256((output / "blind_scores.csv").read_bytes()).hexdigest()
    reveal = {anonymous: path.parent.name for (anonymous, _, _), path in zip(maps, data.paths)}
    json_file(output / "reveal.json", dict(scores_sha256=scores_sha256, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[(row["estimator"], row["channel"], row["metric"])][reveal[row["anonymous_case"]]] = row["loss"]
    qmap = {(r["estimator"], r["channel"]): r for r in quality}
    summaries = []
    for (estimator, channel, metric), losses in grouped.items():
        stats = st.evaluate_curve(losses, data.index["cases"])
        summaries.append(dict(estimator=estimator, channel=channel, metric=metric,
                              **stats, adequate=qmap[(estimator, channel)]["adequate"],
                              validation_nmse=qmap[(estimator, channel)]["validation_nmse"],
                              is_control=metric.startswith("control_")))
    csv_file(output / "metric_summary.csv", summaries)
    for (estimator, channel, metric), losses in grouped.items():
        if metric.startswith("control_"):
            scale = max(1., max(abs(v) for v in losses.values()))
            if np.ptp(list(losses.values())) > 1e-7*scale:
                raise ArithmeticError(f"Rotation-invariant control drifted: {estimator}/{channel}/{metric}")
    json_file(output / "read_audit.json", dict(files=sorted(set(data.read_files)),
              forbidden_files_read=[], all_subjects=data.n, all_trial_rows=data.m,
              evaluation_subjects=len(np.unique(data.sid[rows])),
              evaluation_rows=len(rows), registration_max_residual=max(v[2] for v in maps)))
    config["status"] = "COMPLETE"
    config["scores_sha256"] = scores_sha256
    json_file(output / "protocol.json", config)
    return summaries


def compare(paths, destination):
    """Exact intersection: same metric, channel AND estimator in all models."""
    if set(paths) != set(io.DIMS):
        raise ValueError("C1, C2, C3 AND CR3 are all required")
    all_rows = {}
    for model, folder in paths.items():
        folder = Path(folder)
        protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
        if protocol["status"] != "COMPLETE":
            raise ValueError(f"Incomplete run: {model}")
        if (protocol["model"] != model or protocol["Z_dim"] != io.DIMS[model] or
                protocol["n_subjects"] != 5000):
            raise ValueError(f"{model} requires its own completed 5000-subject dataset")
        digest = hashlib.sha256((folder / "blind_scores.csv").read_bytes()).hexdigest()
        if digest != protocol["scores_sha256"]:
            raise ValueError(f"Frozen scores changed: {model}")
        # Source paths differ, but every scoring and fit rule must match.
        comparable = {k: v for k, v in protocol.items()
                      if k not in ("source", "scores_sha256", "status", "model", "n_subjects", "Z_dim")}
        if all_rows and comparable != common_protocol:
            raise ValueError("Model configurations differ; no universal-pass declaration")
        common_protocol = comparable
        with (folder / "metric_summary.csv").open(newline="", encoding="utf-8") as stream:
            all_rows[model] = {(r["estimator"], r["channel"], r["metric"]): r
                               for r in csv.DictReader(stream)}
    keys = set.intersection(*(set(rows) for rows in all_rows.values()))
    result = []
    for key in sorted(keys):
        rows = [all_rows[m][key] for m in io.DIMS]
        passed = all(r["all_planes_pass"] == "True" and r["adequate"] == "True" and
                     r["is_control"] == "False" for r in rows)
        row = dict(estimator=key[0], channel=key[1], metric=key[2],
                   all_four_pass=passed, derivative_accuracy_verified=False)
        for m, r in zip(io.DIMS, rows):
            row[m + "_rank"] = r["baseline_rank"]
            row[m + "_planes"] = r["success_planes"] + "/" + r["n_planes"]
        result.append(row)
    families = defaultdict(set)
    for row in result:
        if row["all_four_pass"]:
            family = "poly" if row["estimator"].startswith("poly") else "rbf"
            families[(row["channel"], row["metric"])].add(family)
    for row in result:
        row["cross_family_pass"] = bool(row["all_four_pass"] and
            families[(row["channel"], row["metric"])] >= {"poly", "rbf"})
    csv_file(destination, result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=io.DIMS)
    p.add_argument("--step2-dir", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--channels", default="logit_nn,p_nn")
    p.add_argument("--estimators", default="poly2,poly3,rbf0.5,rbf1,rbf2")
    p.add_argument("--centers", type=int, default=128)
    p.add_argument("--per-subject", type=int, default=8)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--seed", type=int, default=381)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--run-name", default="waveA_seed381")
    args = p.parse_args(argv)
    if args.centers < 2 or args.per_subject < 1 or args.chunk < 1:
        p.error("Invalid resource configuration")
    if args.compare:
        paths = {m: io.default_step2(ROOT, m).parent / "step5_matrix_metrics" / args.run_name
                 for m in io.DIMS}
        dest = args.output or ROOT / "outputs" / "supervised_abs_rotation" / f"step5_{args.run_name}_four_models.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        rows = compare(paths, dest)
        stable = {(r["channel"], r["metric"]) for r in rows if r["cross_family_pass"]}
        print(f"Four-model candidates stable across both estimator families: {len(stable)}; {dest}")
        return
    if not args.model:
        p.error("--model is required unless --compare is used")
    source = args.step2_dir or io.default_step2(ROOT, args.model)
    output = args.output or source.parent / "step5_matrix_metrics" / args.run_name
    # --step2-dir may override a path, but must still match the selected model.
    source_index = json.loads((source / "index.json").read_text(encoding="utf-8"))
    if source_index.get("model") != args.model or int(source_index["Z_dim"]) != io.DIMS[args.model]:
        p.error("--step2-dir does not contain the selected model's matrices")
    analyze(source, output, channels=tuple(args.channels.split(",")), seed=args.seed,
            centers=args.centers, per_subject=args.per_subject, chunk=args.chunk,
            models=tuple(args.estimators.split(",")))


if __name__ == "__main__":
    main()
