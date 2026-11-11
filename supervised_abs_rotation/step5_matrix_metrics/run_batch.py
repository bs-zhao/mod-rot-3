"""CR3-first Wave A runner, retaining and verifying completed default runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from threadpoolctl import threadpool_limits
import run as wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-models", action="store_true", help="Complete original four-model batch even if CR3 fails")
    args = parser.parse_args()
    paths = {}
    digest = hashlib.sha256(b"".join((wave.HERE / f).read_bytes() for f in
        ("matrix_io.py", "surrogates.py", "structure.py", "run.py"))).hexdigest()
    with threadpool_limits(limits=4):
        for model in ("CR3", "C1", "C2", "C3"):
            source = wave.io.default_step2(wave.ROOT, model)
            output = source.parent / "step5_matrix_metrics" / "waveA_seed381"
            paths[model] = output
            if (output / "protocol.json").exists():
                protocol = json.loads((output / "protocol.json").read_text())
                if protocol["status"] != "COMPLETE" or protocol["code_sha256"] != digest:
                    raise ValueError(f"Existing run is incomplete or uses different scoring code: {output}")
                if (protocol["model"] != model or protocol["seed"] != 381 or
                    protocol["centers"] != 128 or protocol["per_subject"] != 8 or
                    protocol["channels"] != ["logit_nn", "p_nn"] or
                    protocol["estimators"] != ["poly2", "poly3", "rbf0.5", "rbf1", "rbf2"]):
                    raise ValueError(f"Existing run does not match defaults: {output}")
                scores_digest = hashlib.sha256((output / "blind_scores.csv").read_bytes()).hexdigest()
                if scores_digest != protocol["scores_sha256"]:
                    raise ValueError(f"Frozen scores changed: {output}")
                with (output / "metric_summary.csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                passed = any(r["all_planes_pass"] == "True" and r["adequate"] == "True" and r["is_control"] == "False" for r in rows)
                print(f"[Step5 batch] {model}: verified completed run; no refit", flush=True)
            else:
                rows = wave.analyze(source, output)
                passed = any(r["all_planes_pass"] and r["adequate"] and not r["is_control"] for r in rows)
            if model == "CR3" and not passed and not args.all_models:
                print("[Step5 batch] CR3 failed; no later models launched. Use --all-models for a full audit.", flush=True)
                return
        dest = wave.ROOT / "outputs" / "supervised_abs_rotation" / "step5_waveA_seed381_four_models.csv"
        rows = wave.compare(paths, dest)
        print(f"[Step5 batch] cross-family four-model candidates: {sum(r['cross_family_pass'] for r in rows)}; {dest}")


if __name__ == "__main__":
    main()
