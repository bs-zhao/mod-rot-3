"""Validate one fixed cubic-projection lead across models; no full-function claim."""
import argparse
import hashlib
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

import run as wave
import prepare_response_bank
import run_separation


RULE = dict(estimator="poly3", channel="logit_nn", metric="s0.2_axis_worst_effective_support_p0p5")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=wave.io.DIMS, default="CR3")
    p.add_argument("--batch", action="store_true")
    p.add_argument("--run-name", default="waveJ3_cubic_projection_seed389")
    p.add_argument("--bank-name", default="projection_cubic_bank_seed383")
    args = p.parse_args()
    paths, screen = {}, []
    destination = wave.ROOT / "outputs" / "supervised_abs_rotation" / f"step5_{args.run_name}_projection_screen.csv"
    with threadpool_limits(limits=4):
        for model in (("CR3", "C1", "C2", "C3") if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model)
            parent = source.parent / "step5_matrix_metrics"
            bank = parent / args.bank_name
            if not (bank / "protocol.json").exists():
                prepare_response_bank.prepare(source, bank, estimators=("poly3",))
            bank_protocol = json.loads((bank / "protocol.json").read_text())
            if bank_protocol["status"] != "COMPLETE" or bank_protocol["estimators"] != ["poly3"]:
                raise ValueError("Expected the completed frozen cubic-only function bank")
            output = parent / args.run_name
            paths[model] = output
            rows = run_separation.analyze(source, bank, output)
            protocol = json.loads((output / "protocol.json").read_text())
            protocol.update(criterion_scope="fixed isotropic cubic projection of W/Y",
                fixed_candidate=RULE, full_function_estimator_stability_passed=False,
                screen_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            wave.json_file(output / "protocol.json", protocol)
            selected = next(r for r in rows if all(r[k] == v for k, v in RULE.items()))
            passed = bool(selected["adequate"] and selected["all_planes_pass"])
            screen.append(dict(model=model, **RULE, baseline_rank=selected["baseline_rank"],
                success_planes=selected["success_planes"], n_planes=selected["n_planes"],
                adequate=selected["adequate"], projection_screen_pass=passed,
                full_function_stability_pass=False))
            wave.csv_file(destination, screen)
            print(f"[Cubic projection] {model}: rank {selected['baseline_rank']}; "
                  f"{selected['success_planes']}/{selected['n_planes']} planes; adequate={selected['adequate']}", flush=True)
            if args.batch and not passed:
                print("[Cubic projection] Fixed common rule failed; later models not launched. No universal claim.", flush=True)
                return
        if args.batch:
            compared = destination.with_name(destination.stem+"_all_metrics.csv")
            wave.compare(paths, compared)
            print("[Cubic projection] All four projection screens passed; stability/optimization still required.", flush=True)


if __name__ == "__main__":
    main()
