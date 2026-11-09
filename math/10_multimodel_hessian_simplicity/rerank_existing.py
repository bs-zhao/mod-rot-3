"""Rebuild coverage-aware cross-model Gap2/3 ranking from existing CSV only.

No data loading, no fitted regression, no neural model / GPU.
Run from repo root:
    python math/10_multimodel_hessian_simplicity/rerank_existing.py
Writes a new *_coverage_aware.csv without overwriting existing rankings.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from run_gaps import DEFAULT_OUT, cross_summary
from run import write_csv
from oracle import DIMENSIONS


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,
                   default=DEFAULT_OUT/"gaps23"/"all_models_landscape_comparisons.csv")
    p.add_argument("--output",type=Path,
                   default=DEFAULT_OUT/"gaps23"/"cross_model_threeway_ranking_coverage_aware.csv")
    p.add_argument("--models",nargs="+",default=["C1","C2","C3","CR3"])
    args=p.parse_args(argv)
    models=[s.upper() for s in args.models]
    if any(s not in DIMENSIONS for s in models):
        p.error("Only C1,C2,C3,CR3 are implemented")
    if not args.input.is_file():
        p.error(f"Missing comparisons CSV: {args.input}. Run Gap2+3 first")
    with args.input.open("r",encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    if not rows:
        p.error(f"Empty comparisons file: {args.input}")
    seen=set(r["model"] for r in rows)
    if not set(models).issubset(seen):
        print(f"WARNING: Requested models {models}, present {sorted(seen)}. "
              "Incomplete model coverage will be explicitly flagged.")
    summary=cross_summary(rows,models)
    write_csv(args.output,summary)
    print(f"Saved {len(summary)} coverage-aware metric/comparison records: {args.output}")
    print("First 5 fully covered Gap3 vs analytic candidates:")
    winners=[s for s in summary if s["comparison"]=="gap3_vs_analytic" and s["full_coverage"]]
    for s in winners[:5]:
        print(s["metric"],"worst_model_both_near=",f'{s["min_model_both_near_fraction"]:.3f}',
              "worst_model_curve_r=",f'{s["min_model_median_curve_r"]:.3f}')
    if not winners:
        print("NO fully covered four-model candidates; do not infer universal success.")


if __name__=="__main__":
    main()
