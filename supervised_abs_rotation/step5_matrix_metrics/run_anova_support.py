"""Wave I: effective-support scoring of frozen local ANOVA components."""
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
import run_anova
import tail_support as ts


def analyze(source_run, output, seed=388):
    source_run, output = Path(source_run), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output: {output}")
    old = json.loads((source_run / "protocol.json").read_text())
    if old["status"] != "COMPLETE" or hashlib.sha256((source_run / "blind_scores.csv").read_bytes()).hexdigest() != old["scores_sha256"]:
        raise ValueError("Source local ANOVA experiment incomplete or changed")
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=old["model"], Z_dim=old["Z_dim"], n_subjects=old["n_subjects"],
        source=old["source"], source_run_name=source_run.name, source_code_sha256=old["code_sha256"],
        seed=seed, estimators=old["estimators"], channels=old["channels"], all_subjects_used=True,
        relative_L2_errors=[.01, .03, .1], effective_support_p=[.5, 1.],
        score_sign="lower for all datasets", cr3_gate="same support metric passes polynomial and RBF",
        status="RUNNING", code_sha256=hashlib.sha256(b"".join((wave.HERE / f).read_bytes() for f in
            ("run_anova_support.py", "tail_support.py", "structure.py"))).hexdigest())
    wave.json_file(output / "protocol.json", config)
    audits = json.loads((source_run / "anova_audit.json").read_text())
    identities = sorted({a["anonymous_case"] for a in audits})
    rng = np.random.default_rng(seed)
    shuffled = [identities[i] for i in rng.permutation(len(identities))]
    anonymous = {identity: f"case_{k:03d}" for k, identity in enumerate(shuffled)}
    blind = []
    for audit in audits:
        e = np.asarray(audit["component_energy"])
        order = np.array([len(s) for s in audit["subsets"]])
        groups = dict(all_components=e, interaction_components=e[order >= 2], pair_components=e[order == 2])
        for label, values in groups.items():
            for metric, loss in ts.scores(values).items():
                blind.append(dict(anonymous_case=anonymous[audit["anonymous_case"]],
                    estimator=audit["estimator"], channel=audit["channel"],
                    metric=f"s{audit['sigma']:g}_{label}_{metric}", loss=loss))
    wave.csv_file(output / "blind_scores.csv", blind)
    digest = hashlib.sha256((output / "blind_scores.csv").read_bytes()).hexdigest()
    # Only reveal source identities after every support score is frozen.
    original = json.loads((source_run / "reveal.json").read_text())["mapping"]
    reveal = {anonymous[identity]: original[identity] for identity in identities}
    wave.json_file(output / "reveal.json", dict(scores_sha256=digest, mapping=reveal))
    with (source_run / "metric_summary.csv").open(newline="") as stream:
        quality = {(r["estimator"], r["channel"], r["metric"].split("_")[0]): r
                   for r in csv.DictReader(stream)}
    index = json.loads((Path(old["source"]) / "index.json").read_text())
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row["estimator"], row["channel"], row["metric"]][reveal[row["anonymous_case"]]] = row["loss"]
    summaries = []
    for (name, channel, metric), values in grouped.items():
        qr = quality[name, channel, metric.split("_")[0]]
        summaries.append(dict(estimator=name, channel=channel, metric=metric,
            **wave.st.evaluate_curve(values, index["cases"]), adequate=qr["adequate"] == "True",
            validation_nmse=qr["validation_nmse"], reference_coverage=qr["reference_coverage"], is_control=False))
    wave.csv_file(output / "metric_summary.csv", summaries)
    wave.json_file(output / "read_audit.json", dict(source_scores_sha256=old["scores_sha256"],
        source_audit_sha256=hashlib.sha256((source_run / "anova_audit.json").read_bytes()).hexdigest(),
        derived_files=[str(source_run / f) for f in ("protocol.json", "blind_scores.csv", "anova_audit.json", "reveal.json", "metric_summary.csv")],
        forbidden_files_read=[]))
    config.update(status="COMPLETE", scores_sha256=digest)
    wave.json_file(output / "protocol.json", config)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=wave.io.DIMS, default="CR3")
    p.add_argument("--run-name", default="waveI_anova_support_seed388")
    p.add_argument("--source-run", default="waveF_anova_seed386")
    p.add_argument("--seed", type=int, default=388)
    p.add_argument("--batch", action="store_true")
    args = p.parse_args()
    paths = {}
    with threadpool_limits(limits=4):
        for model in (("CR3", "C1", "C2", "C3") if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model)
            parent = source.parent / "step5_matrix_metrics"
            anova = parent / args.source_run
            if not (anova / "protocol.json").exists():
                fields = parent / "waveC_finite_seed383"
                if not (fields / "protocol.json").exists():
                    run_finite.analyze(source, fields)
                run_anova.analyze(source, fields, anova)
            output = parent / args.run_name
            paths[model] = output
            rows = analyze(anova, output, args.seed)
            passing = defaultdict(set)
            for r in rows:
                if r["adequate"] and r["all_planes_pass"]:
                    passing[r["channel"], r["metric"]].add("poly" if r["estimator"].startswith("poly") else "rbf")
            robust = [key for key, families in passing.items() if families >= {"poly", "rbf"}]
            print(f"[WaveI] {model}: {len(robust)} cross-family support candidates", flush=True)
            if args.batch and model == "CR3" and not robust:
                print("[WaveI] CR3 failed; later models not launched", flush=True)
                return
        if args.batch:
            wave.compare(paths, wave.ROOT / "outputs" / "supervised_abs_rotation" / f"step5_{args.run_name}_four_models.csv")


if __name__ == "__main__":
    main()
