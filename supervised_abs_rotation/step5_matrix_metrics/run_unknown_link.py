"""Wave D: learn response links numerically before testing additive compression."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run as wave
import run_finite
import unknown_link as ul


def analyze(source, output, seed=384, families=("poly3", "poly5", "rbf6", "rbf10"), iterations=12, patience=None, cache_source=None):
    if iterations < 0 or (patience is not None and patience < 1):
        raise ValueError("Nonnegative iterations and positive patience required")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output to preserve prior attempts: {output}")
    data = wave.io.MatrixCases(source, ("logit_nn", "p_nn"), seed)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=data.index["model"], Z_dim=data.d, n_subjects=data.n,
        source=str(Path(source).resolve()), seed=seed, estimators=list(families),
        iterations=iterations, patience=patience, identity_mix=.1, ridge=.001, subject_split=[.6, .2, .2],
        channels=list(data.channels), all_subjects_used=True,
        score_sign="lower for all datasets", adequacy_validation_nmse=.1,
        transform_training="training subjects only; iteration selected on validation raw-Y NMSE",
        status="RUNNING", code_sha256=hashlib.sha256(b"".join((wave.HERE / f).read_bytes() for f in
            ("run_unknown_link.py", "unknown_link.py", "additive_profiles.py", "run_finite.py", "matrix_io.py", "surrogates.py", "structure.py"))).hexdigest())
    wave.json_file(output / "protocol.json", config)
    rng = np.random.default_rng(seed)
    ids = rng.permutation(data.n)
    split = (np.sort(ids[:int(.6*data.n)]), np.sort(ids[int(.6*data.n):int(.8*data.n)]), np.sort(ids[int(.8*data.n):]))
    mean = data.u[split[0]].mean(0)
    scale = np.sqrt(np.var(data.u[split[0]], axis=0).mean())
    u = (data.u-mean)/scale
    maps, reveal = [], {}
    for k, path in enumerate(data.paths):
        q, _ = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); reveal[f"case_{k:03d}"] = path.parent.name
    bank, x, profile_quality = run_finite.response_bank(data, seed)
    wave.json_file(output / "response_quality.json", profile_quality)
    np.savez_compressed(output / "subject_split.npz", train=split[0], valid=split[1], test=split[2])
    blind, fits = [], []
    cached_fits, cached_scores = {}, {}
    reuse = dict(reused_fits=0, rejected_fits=0)
    if cache_source is not None:
        cache_source = Path(cache_source)
        cached_protocol = json.loads((cache_source / "protocol.json").read_text())
        previous_patience = cached_protocol.get("patience")
        if previous_patience is not None and (patience is None or previous_patience < patience):
            raise ValueError("Cached histories stopped before the requested patience rule")
        for key in ("model", "Z_dim", "n_subjects", "source", "seed", "estimators", "iterations", "identity_mix", "ridge", "subject_split", "channels"):
            if cached_protocol[key] != config[key]:
                raise ValueError(f"Cached experiment differs in {key}")
        reuse.update(source=str(cache_source.resolve()),
            original_code_sha256=cached_protocol["code_sha256"],
            scores_sha256=hashlib.sha256((cache_source / "blind_scores.csv").read_bytes()).hexdigest(),
            diagnostics_sha256=hashlib.sha256((cache_source / "fit_diagnostics.json").read_bytes()).hexdigest())
        for row in json.loads((cache_source / "fit_diagnostics.json").read_text()):
            cached_fits[row["anonymous_case"], row["estimator"], row["channel"]] = row
        with (cache_source / "blind_scores.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                cached_scores.setdefault((row["anonymous_case"], row["estimator"], row["channel"]), {})[row["metric"]] = float(row["loss"])
    for family in families:
        for k, q in enumerate(maps):
            for c, channel in enumerate(data.channels):
                key = (f"case_{k:03d}", family, channel)
                previous = cached_fits.get(key)
                usable = False
                if previous is not None:
                    stopped, selected = ul.validation_checkpoint(previous["history"], patience)
                    usable = selected == previous["best_iteration"] and key in cached_scores
                if usable:
                    fit = {kk: vv for kk, vv in previous.items() if kk not in ("anonymous_case", "estimator", "channel")}
                    fit["history"] = previous["history"][:stopped+1]
                    scores = cached_scores[key]
                    reuse["reused_fits"] += 1
                else:
                    if previous is not None:
                        reuse["rejected_fits"] += 1
                    scores, fit = ul.fit_case(u @ q, bank[:, c], split, family, iterations=iterations, patience=patience)
                fits.append(dict(anonymous_case=f"case_{k:03d}", estimator=family, channel=channel, **fit))
                for metric, value in scores.items():
                    blind.append(dict(anonymous_case=f"case_{k:03d}", estimator=family,
                                      channel=channel, metric=metric, loss=value))
            if (k+1) % 4 == 0 or k == len(maps)-1:
                wave.csv_file(output / "blind_scores.csv", blind)
                wave.json_file(output / "fit_diagnostics.json", fits)
                print(f"[WaveD] {family}: {k+1}/{len(maps)} anonymous cases", flush=True)
        wave.csv_file(output / "blind_scores.csv", blind)
        wave.json_file(output / "fit_diagnostics.json", fits)
        wave.json_file(output / "runtime_reuse_audit.json", reuse)
    digest = hashlib.sha256((output / "blind_scores.csv").read_bytes()).hexdigest()
    wave.json_file(output / "reveal.json", dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row["estimator"], row["channel"], row["metric"]][reveal[row["anonymous_case"]]] = row["loss"]
    summaries = []
    for (family, channel, metric), values in grouped.items():
        prediction_error = max(grouped[family, channel, "unknown_link_test_nmse"].values())
        profile_error = profile_quality["validation_nmse"][data.channels.index(channel)]
        summaries.append(dict(estimator=family, channel=channel, metric=metric,
            **wave.st.evaluate_curve(values, data.index["cases"]),
            adequate=bool(prediction_error <= .1 and profile_error <= .1),
            validation_nmse=max(prediction_error, profile_error),
            worst_raw_Y_test_nmse=prediction_error, profile_nmse=profile_error, is_control=False))
    wave.csv_file(output / "metric_summary.csv", summaries)
    wave.json_file(output / "read_audit.json", dict(files=sorted(set(data.read_files)),
        forbidden_files_read=[], all_subjects=data.n, all_rows=data.m))
    config.update(status="COMPLETE", scores_sha256=digest)
    wave.json_file(output / "protocol.json", config)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=wave.io.DIMS, default="CR3")
    p.add_argument("--families", default="poly3,poly5,rbf6,rbf10")
    p.add_argument("--run-name", default="waveD_unknown_link_seed384")
    p.add_argument("--seed", type=int, default=384)
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--patience", type=int)
    p.add_argument("--reuse-run-name")
    p.add_argument("--batch", action="store_true")
    args = p.parse_args()
    paths = {}
    with threadpool_limits(limits=4):
        for model in (("CR3", "C1", "C2", "C3") if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model)
            output = source.parent / "step5_matrix_metrics" / args.run_name
            paths[model] = output
            cache_source = source.parent / "step5_matrix_metrics" / args.reuse_run_name if args.reuse_run_name else None
            if cache_source is not None and not cache_source.exists():
                cache_source = None
            rows = analyze(source, output, args.seed, tuple(args.families.split(",")), args.iterations, args.patience, cache_source)
            passed = [r for r in rows if r["adequate"] and r["all_planes_pass"]]
            print(f"[WaveD] {model}: {len(passed)} adequate full-plane candidates", flush=True)
            if args.batch and model == "CR3" and not passed:
                print("[WaveD] CR3 failed; later models not launched", flush=True)
                return
        if args.batch:
            wave.compare(paths, wave.ROOT / "outputs" / "supervised_abs_rotation" / f"step5_{args.run_name}_four_models.csv")


if __name__ == "__main__":
    main()
