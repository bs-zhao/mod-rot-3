"""Wave E: local output-link-profiled interaction sparsity, CR3 first."""
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
import profiled_links as pl


def analyze(source, fields, output, seed=385):
    source, fields, output = Path(source), Path(fields), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output to retain earlier trials: {output}")
    old = json.loads((fields / "protocol.json").read_text())
    digest = hashlib.sha256((fields / "blind_scores.csv").read_bytes()).hexdigest()
    if old["status"] != "COMPLETE" or digest != old["scores_sha256"]:
        raise ValueError("Source field experiment is incomplete or frozen scores changed")
    data = wave.io.MatrixCases(source, tuple(old["channels"]), old["seed"])
    if old["model"] != data.index["model"] or old["n_subjects"] != data.n:
        raise ValueError("Source fields and W matrices do not match")
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=data.index["model"], Z_dim=data.d, n_subjects=data.n,
        source=str(source.resolve()), seed=seed, source_run_name=fields.name,
        channels=list(data.channels), estimators=old["estimators"],
        profiles=["l2", "l1"], adequacy_validation_nmse=.1,
        all_subjects_used=True, local_link_integrability_verified=False,
        score_sign="lower for all datasets", status="RUNNING",
        source_code_sha256=old["code_sha256"],
        code_sha256=hashlib.sha256(b"".join((wave.HERE / f).read_bytes() for f in
            ("run_profiled.py", "profiled_links.py", "matrix_io.py", "structure.py"))).hexdigest())
    wave.json_file(output / "protocol.json", config)
    rng = np.random.default_rng(seed)
    paths = [data.paths[i] for i in rng.permutation(len(data.paths))]
    maps, reveal = [], {}
    for k, path in enumerate(paths):
        q, _ = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); reveal[f"case_{k:03d}"] = path.parent.name
    with (fields / "fit_quality.csv").open(newline="") as stream:
        quality = {(r["estimator"], r["channel"]): r for r in csv.DictReader(stream)}
    blind, audits, field_hashes = [], [], {}
    for name in old["estimators"]:
        path = fields / f"field_{name}.npz"
        field_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as saved:
            g = saved["gradient"].copy(); h = saved["hessian"].copy()
        for k, q in enumerate(maps):
            gg = g @ q
            hh = np.einsum("ij,ncsjk,kl->ncsil", q.T, h, q, optimize=True)
            for c, channel in enumerate(data.channels):
                for method in ("l2", "l1"):
                    scores, audit = pl.scores(gg[:, c].reshape(-1, data.d),
                        hh[:, c].reshape(-1, data.d, data.d), method)
                    audits.append(dict(anonymous_case=f"case_{k:03d}", estimator=name,
                                       channel=channel, method=method, **audit))
                    for metric, loss in scores.items():
                        blind.append(dict(anonymous_case=f"case_{k:03d}", estimator=name,
                            channel=channel, metric=method+"_"+metric, loss=loss))
            if (k+1) % 8 == 0 or k == len(maps)-1:
                print(f"[WaveE] {name}: {k+1}/{len(maps)} anonymous cases", flush=True)
        wave.csv_file(output / "blind_scores.csv", blind)
        wave.json_file(output / "profiling_audit.json", audits)
    digest = hashlib.sha256((output / "blind_scores.csv").read_bytes()).hexdigest()
    wave.json_file(output / "reveal.json", dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row["estimator"], row["channel"], row["metric"]][reveal[row["anonymous_case"]]] = row["loss"]
    summaries = []
    for (name, channel, metric), values in grouped.items():
        qr = quality[name, channel]
        summaries.append(dict(estimator=name, channel=channel, metric=metric,
            **wave.st.evaluate_curve(values, data.index["cases"]),
            adequate=qr["adequate"] == "True", validation_nmse=qr["validation_nmse"], is_control=False))
    wave.csv_file(output / "metric_summary.csv", summaries)
    wave.json_file(output / "read_audit.json", dict(files=sorted(set(data.read_files)),
        field_source=str(fields.resolve()), field_sha256=field_hashes,
        forbidden_files_read=[], all_subjects=data.n, all_rows=data.m,
        source_scores_sha256=old["scores_sha256"]))
    config.update(status="COMPLETE", scores_sha256=digest)
    wave.json_file(output / "protocol.json", config)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=wave.io.DIMS, default="CR3")
    p.add_argument("--run-name", default="waveE_profiled_seed385")
    p.add_argument("--source-run", default="waveC_finite_seed383")
    p.add_argument("--seed", type=int, default=385)
    p.add_argument("--batch", action="store_true")
    args = p.parse_args()
    paths = {}
    with threadpool_limits(limits=4):
        for model in (("CR3", "C1", "C2", "C3") if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model)
            fields = source.parent / "step5_matrix_metrics" / args.source_run
            if not (fields / "protocol.json").exists():
                print(f"[WaveE] Creating identical generic source fits for {model}", flush=True)
                run_finite.analyze(source, fields)
            output = source.parent / "step5_matrix_metrics" / args.run_name
            paths[model] = output
            rows = analyze(source, fields, output, args.seed)
            passed = [r for r in rows if r["adequate"] and r["all_planes_pass"]]
            print(f"[WaveE] {model}: {len(passed)} adequate full-plane candidates", flush=True)
            if args.batch and model == "CR3" and not passed:
                print("[WaveE] CR3 failed; later models not launched", flush=True)
                return
        if args.batch:
            wave.compare(paths, wave.ROOT / "outputs" / "supervised_abs_rotation" / f"step5_{args.run_name}_four_models.csv")


if __name__ == "__main__":
    main()
