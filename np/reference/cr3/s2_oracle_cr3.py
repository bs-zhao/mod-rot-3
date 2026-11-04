"""CR3 Oracle stage: CR2 vs CR3, both EXACTLY 500 trials, NO NP/ICA.

Uses exactly the audited conditional-nullspace and automatic-X-condition
discovery algorithms from CR2's functional_role_identifiability folder.
Reuses identical independently drawn oracle parameter cohorts for D0/D1.
Only stimulus design changes. No augmented 628-trial branch is used.
Optional: --stage mle for bounded Oracle parameter fitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(RESEARCH),
                str(RESEARCH / "functional_role_identifiability")]
from s0_s1_cr3 import BANK, ORIGINAL, OUT, condition_counts, hash_file, dominance_audit
from s0_cr2_design_recovery import sample_prior, metrics, evaluate
from conditional_nullspace_oracle import (
    BLOCKS, audit, assignment_cost, alpha_gamma_rotation,
    coordinate_stationarity, optimize,
)
from discover_conditions_oracle import explore

RESULTS = OUT / "s2_oracle"
INPUT_NAMES = ("D0_CR2_original500", "D1_CR3_balanced500")


def save_report(filename, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    target = RESULTS / filename
    target.write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(f"Saved {target}", flush=True)


def read_banks(original):
    banks = {}
    for name, path in zip(INPUT_NAMES, (Path(original), BANK)):
        if not path.exists():
            raise FileNotFoundError(f"Required 500-trial bank: {path}")
        with np.load(path, allow_pickle=False) as f:
            x = np.asarray(f["X"], dtype=np.float64)
        if x.shape != (500, 6):
            raise ValueError(f"{path}: expected 500x6, got {x.shape}")
        if name == "D1_CR3_balanced500":
            audit_result = dominance_audit(x)
            if audit_result["total"]:
                raise ValueError(
                    f"CR3 fixed bank contains {audit_result['total']} FOSD-dominated trials; "
                    "this bank was generated before the no-dominance rule. "
                    "Archive previous CR3 outputs and regenerate the 500-X bank "
                    "using updated s0_s1_cr3.py.")
        banks[name] = (x, path)
    return banks


def reference_subjects(args):
    rng = np.random.default_rng(args.seed + 60001)
    n = args.oracle_subjects
    theta = sample_prior(n, 4, rng)
    # Same SD reference across both D0 and D1. SD scaling is diagonal,
    # not full covariance whitening, and cannot rotate true parameter axes.
    sd = sample_prior(2048, 4, np.random.default_rng(args.seed+61121)).std(0)
    return theta[:n//2], theta[n//2:], sd


def run_conditional(args, banks, train, val, sd):
    report = dict(study="CR3 stratified 500-X Oracle vs CR2 original 500-X",
                  model="CR2-4P simulator and priors unchanged",
                  no_NP=True, no_ICA=True,
                  n_oracle_train=len(train), n_oracle_validation=len(val),
                  parameter_names=["alpha","lambda_loss","gamma","tau"],
                  criterion="distinct conditional PSD operator 1D nullspaces",
                  min_condition_trials=args.min_condition_trials,
                  prior_standardization_sd=sd.tolist(),
                  comparison_is_same_n500_and_same_oracle_theta=True)
    for name, (x, path) in banks.items():
        fit, valid, diag = audit(
            x, train, val, sd, args.min_condition_trials, args.minimum_rank_gap)
        eligible = [k for k in BLOCKS if
                    diag["train"][k]["certified_eligible"] and
                    diag["validation"][k]["certified_eligible"]]
        diag["eligible_blocks"] = eligible
        diag["stimulus_counts"] = condition_counts(x)
        diag["source_bank_sha256"] = hash_file(path)
        diag["three_axis_rank_and_coverage_gate_pass"] = bool(len(eligible) == 3)
        if eligible:
            if args.skip_optimization:
                diag["optimization_skipped"] = True
            else:
                diag["tangent_at_oracle"] = coordinate_stationarity(fit, set(eligible))
                diag["optimized"] = optimize(
                    fit, valid, set(eligible), args.starts, args.maxiter,
                    args.seed + (9 if name.startswith("D1") else 5))
            diag["alpha_gamma_rotation_family"] = [
                dict(angle_degrees=float(np.rad2deg(angle)),
                     oracle_loss=float(assignment_cost(
                         alpha_gamma_rotation(angle), fit, set(eligible))))
                for angle in np.linspace(0, np.pi/2, 7)
            ]
        report[name] = diag
        print(f"{name}: count={diag['coverage']}, eligible={eligible}, "
              f"full_gate={len(eligible)==3}", flush=True)
    save_report("conditional_nullspace__report.json", report)
    return report


def run_discovery(args, banks, train, val, sd):
    report = dict(study="Automatic conditional information-nullspace discovery",
                  no_NP=True, no_ICA=True, parameters_only_for_posthoc_audit=True,
                  same_oracle_reference_persons=True, n_stimuli_each=500,
                  X_candidate_search="1D exact/quantile regions and cross-feature pairs")
    for name, (x,path) in banks.items():
        result = explore(x, train, val, sd, args.min_discovery_trials,
                         threshold=args.null_threshold, gap=args.discovery_gap)
        result["source_bank_sha256"] = hash_file(path)
        report[name] = result
        print(f"{name}: regions={result['n_candidates']} "
              f"valid={result['n_valid']} triad={result['selected_triad'] is not None}",
              flush=True)
    save_report("automatic_nullspace_discovery__report.json", report)
    return report


def run_fisher(args, banks):
    rng = np.random.default_rng(args.seed + 77555)
    refs = sample_prior(args.fisher_reference_subjects, 4, rng)
    result = dict(study="Oracle Fisher parameter-information diagnostics",
                  formula_and_prior_identical=True,
                  reference_subject_count=len(refs), n_trials=500,
                  fisher_balance_gate_threshold=4.0,
                  note="These are prior-reference Fisher diagnostics, not observed-choice MLE results.")
    for name, (x, path) in banks.items():
        m=metrics(x, refs)
        result[name] = dict(metrics=m,
                            fisher_balance_gate_pass=bool(m["fisher_balance_ratio"]<4),
                            bank_sha256=hash_file(path))
        print(f"{name}: Fisher balance max/min = {m['fisher_balance_ratio']:.3g}",
              flush=True)
    save_report("fisher_precheck__report.json", result)


def run_mle(args, banks):
    result=dict(study="Oracle MLE four-parameter recovery using same prior seeds",
                no_NP=True, n_stimuli=500,
                n_subjects=args.mle_subjects,
                n_starts=args.mle_starts,
                note="No train/validation NP; bounded individual MLE with CR2 unchanged formula.")
    for name, (x,path) in banks.items():
        summary,_,_,_ = evaluate(x, 4, args.mle_subjects, args.seed + 719,
                                 starts=args.mle_starts)
        summary["model"] = name
        result[name] = summary
        print(f"{name}: MLE Pearson r = {np.round(summary['pearson_r'],4)}",
              flush=True)
    save_report("oracle_mle_comparison__report.json", result)


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--stage", choices=("all","conditional","discover",
                                            "fisher","mle","selftest"),default="all")
    parser.add_argument("--original-bank", default=str(ORIGINAL))
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--oracle-subjects", type=int, default=96)
    parser.add_argument("--min-condition-trials", type=int, default=8)
    parser.add_argument("--minimum-rank-gap", type=float, default=1e-5)
    parser.add_argument("--min-discovery-trials", type=int, default=12)
    parser.add_argument("--null-threshold", type=float, default=1e-7)
    parser.add_argument("--discovery-gap", type=float, default=1e-3)
    parser.add_argument("--starts", type=int, default=10)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--skip-optimization", action="store_true")
    parser.add_argument("--fisher-reference-subjects", type=int, default=128)
    parser.add_argument("--mle-subjects", type=int, default=100)
    parser.add_argument("--mle-starts", type=int, default=4)
    args=parser.parse_args()
    if args.stage=="selftest":
        from conditional_nullspace_oracle import check_gradients
        from discover_conditions_oracle import selftest
        check_gradients()
        selftest()
        print("CR3 imported exact Oracle mathematics selftests PASS")
        return
    if args.oracle_subjects<16 or args.min_condition_trials<1 or \
       args.min_discovery_trials<4 or args.starts<1 or args.maxiter<1 or \
       args.fisher_reference_subjects<8 or args.mle_subjects<3:
        parser.error("Invalid oracle sample sizes or optimization budgets")
    banks=read_banks(args.original_bank)
    train,val,sd=reference_subjects(args)
    if args.stage in ("all","conditional"):
        run_conditional(args,banks,train,val,sd)
    if args.stage in ("all","discover"):
        run_discovery(args,banks,train,val,sd)
    if args.stage in ("all","fisher"):
        run_fisher(args,banks)
    if args.stage=="mle":
        run_mle(args,banks)


if __name__=="__main__":
    main()
