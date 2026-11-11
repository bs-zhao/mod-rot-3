"""Wave B: derivative-free additive/pairwise compression; screen CR3 first."""
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
import additive_profiles as ap


def analyze(source, output, families=("poly3", "poly4", "rbf3", "rbf4"), seed=382):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserve previous attempts; choose a new output: {output}")
    channels = ("logit_nn", "p_nn")
    data = wave.io.MatrixCases(source, channels, seed)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=data.index["model"], Z_dim=data.d, n_subjects=data.n,
                  seed=seed, channels=list(channels), estimators=list(families),
                  x_centers=64, x_lengthscale=2., x_polynomial_degree=2,
                  ridge_grid=[1e-6, 1e-3, .1],
                  subject_split=[.6, .2, .2], profile_trial_holdout="trial_index modulo 5 == 0",
                  adequacy_validation_nmse=.1, all_subjects_used=True,
                  source=str(Path(source).resolve()), score_sign="lower for all datasets",
                  sobol_independence_assumed=False, status="RUNNING",
                  code_sha256=hashlib.sha256(b"".join((wave.HERE / f).read_bytes() for f in
                      ("additive_profiles.py", "run_additive.py", "matrix_io.py", "surrogates.py", "structure.py"))).hexdigest())
    wave.json_file(output / "protocol.json", config)
    rng = np.random.default_rng(seed)
    ids = rng.permutation(data.n)
    split = (np.sort(ids[:int(.6*data.n)]), np.sort(ids[int(.6*data.n):int(.8*data.n)]), np.sort(ids[int(.8*data.n):]))
    profiles, quality = ap.x_profiles(data, seed=seed)
    wave.json_file(output / "profile_quality.json", quality)
    np.savez_compressed(output / "response_profiles.npz", profiles=profiles,
                        train=split[0], valid=split[1], test=split[2])
    blind, diagnostics = [], []
    reveals = {}
    mu = data.u[split[0]].mean(0)
    scale = np.sqrt(np.mean(np.var(data.u[split[0]], axis=0)))
    # All mappings come from actual W. Validate every case's row alignment.
    coordinates = []
    for k, path in enumerate(data.paths):
        u = data.coordinates(path)
        q, error = wave.io.registration(data.u, u)
        coordinates.append(((u-mu @ q)/scale, error))
        reveals[f"case_{k:03d}"] = path.parent.name
    for family in families:
        for k, (u, error) in enumerate(coordinates):
            for c, channel in enumerate(channels):
                scores, diagnostic = ap.projection_scores(u, profiles[:, c], split, family)
                for metric, loss in scores.items():
                    blind.append(dict(anonymous_case=f"case_{k:03d}", estimator=family,
                                      channel=channel, metric=metric, loss=loss))
                diagnostics.append(dict(anonymous_case=f"case_{k:03d}", estimator=family,
                                        channel=channel, registration_residual=error, **diagnostic))
            if (k+1) % 8 == 0 or k == len(coordinates)-1:
                print(f"[WaveB] {family}: {k+1}/{len(coordinates)} anonymous cases", flush=True)
        wave.csv_file(output / "blind_scores.csv", blind)
        wave.json_file(output / "fit_diagnostics.json", diagnostics)
    digest = hashlib.sha256((output / "blind_scores.csv").read_bytes()).hexdigest()
    wave.json_file(output / "reveal.json", dict(scores_sha256=digest, mapping=reveals))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row["estimator"], row["channel"], row["metric"]][reveals[row["anonymous_case"]]] = row["loss"]
    summaries = []
    for (family, channel, metric), values in grouped.items():
        pair_error = max(grouped[family, channel, "pairwise_test_nmse"].values())
        profile_error = quality["validation_nmse"][channels.index(channel)]
        summaries.append(dict(estimator=family, channel=channel, metric=metric,
                              **wave.st.evaluate_curve(values, data.index["cases"]),
                              adequate=bool(profile_error <= .1 and pair_error <= .1),
                              profile_nmse=profile_error, worst_pairwise_nmse=pair_error,
                              validation_nmse=max(pair_error, profile_error), is_control=False))
    wave.csv_file(output / "metric_summary.csv", summaries)
    wave.json_file(output / "read_audit.json", dict(files=sorted(set(data.read_files)),
                   forbidden_files_read=[], all_subjects=data.n, all_rows=data.m))
    config.update(status="COMPLETE", scores_sha256=digest)
    wave.json_file(output / "protocol.json", config)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=wave.io.DIMS, default="CR3")
    parser.add_argument("--families", default="poly3,poly4,rbf3,rbf4")
    parser.add_argument("--run-name", default="waveB_additive_seed382")
    parser.add_argument("--seed", type=int, default=382)
    parser.add_argument("--batch", action="store_true", help="CR3 gate, then C1/C2/C3 only for surviving metrics")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    sequence = ("CR3", "C1", "C2", "C3") if args.batch else (args.model,)
    paths = {}
    with threadpool_limits(limits=args.threads):
        for label in sequence:
            source = wave.io.default_step2(wave.ROOT, label)
            dest = source.parent / "step5_matrix_metrics" / args.run_name
            paths[label] = dest
            rows = analyze(source, dest, tuple(args.families.split(",")), args.seed)
            passing = [r for r in rows if r["adequate"] and r["all_planes_pass"]]
            print(f"[WaveB] {label}: {len(passing)} adequate full-plane candidates", flush=True)
            if args.batch and label == "CR3" and not passing:
                print("[WaveB] CR3 screen failed; other models not launched. Failure retained.", flush=True)
                return
        if args.batch:
            dest = wave.ROOT / "outputs" / "supervised_abs_rotation" / f"step5_{args.run_name}_four_models.csv"
            wave.compare(paths, dest)


if __name__ == "__main__":
    main()
