"""One command: Gap1 analytic oracle, THEN Gap2 + Gap3 simultaneously.

The paired numeric gap stages use identical W, numeric estimator, validation,
TEST evaluation anchors and metric shortlist. Existing Gap1 may be reused,
but this NEVER silently substitutes estimated Hessians for an analytic oracle.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from run import main as run_gap1, ROOT, DIMENSIONS
from run_gaps import main as run_gaps23, DEFAULT_OUT
from numeric import digest


def valid_gap1(directory,models):
    directory=Path(directory)
    if not (directory/"cross_model_ranking.csv").is_file():
        return False
    for model in models:
        metadata=directory/model/"metadata.json"
        ranking=directory/model/"ranking_by_model_channel.csv"
        if not metadata.is_file() or not ranking.is_file():
            return False
        meta=json.loads(metadata.read_text(encoding="utf-8"))
        if meta.get("source")!="exact_analytic_oracle":
            return False
        # Prevent a stale Gap1 being mistaken for the present S1 generator.
        saved=meta.get("dataset_sha256")
        data=ROOT/"outputs"/"choice"/model/"s1_generate"/"s1_generate__dataset.npz"
        if not data.is_file() or saved!=digest(data):
            return False
        if not len(ranking.read_bytes()):
            return False
    return True


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,
       epilog="Additional unknown options are forwarded to run_gaps.py. "
              "Use --only-gap1 to stop after analytic screening.")
    ap.add_argument("--models",nargs="+",default=["C1","C2","C3","CR3"])
    ap.add_argument("--gap1-dir",type=Path,default=DEFAULT_OUT)
    ap.add_argument("--reuse-gap1",action="store_true",
                    help="Fail unless completed Gap1 matches current S1 datasets")
    ap.add_argument("--rerun-gap1",action="store_true",
                    help="Force re-evaluation of Gap1 (default reuses valid Gap1)")
    ap.add_argument("--only-gap1",action="store_true")
    ap.add_argument("--only-gaps23",action="store_true",
                    help="Require valid completed Gap1, then run only Gap2/3")
    ap.add_argument("--angles",default="0:90:10")
    ap.add_argument("--gap1-subjects",type=int,default=36)
    ap.add_argument("--gap1-trials",type=int,default=12)
    ap.add_argument("--gap1-multiway",type=int,default=6)
    ap.add_argument("--gap1-plots",type=int,default=3)
    ap.add_argument("--seed",type=int,default=20261109)
    args,unknown=ap.parse_known_args(argv)
    models=[]
    for token in args.models:
        for m in token.upper().split(","):
            if m=="CR1":
                raise ValueError("Repository has no separate CR1. Use actual CR3.")
            if m not in DIMENSIONS:
                raise ValueError(f"Unknown {m}; supported: {list(DIMENSIONS)}")
            if m not in models:
                models.append(m)
    if args.only_gap1 and args.only_gaps23:
        ap.error("Cannot set both --only-gap1 and --only-gaps23")
    if args.rerun_gap1 and args.reuse_gap1:
        ap.error("Cannot set both --rerun-gap1 and --reuse-gap1")
    ready=valid_gap1(args.gap1_dir,models)
    if args.reuse_gap1 or args.only_gaps23:
        if not ready:
            raise FileNotFoundError("Requested reuse of Gap1, but original analytic Gap1 "
                                    "files or saved S1 SHA256 no longer match")
    if not args.only_gaps23:
        if args.rerun_gap1 or not ready:
            print("[Pipeline] Gap1: exact analytic oracle first",flush=True)
            run_gap1(["--models",*models,"--angles",args.angles,
                      "--subjects",str(args.gap1_subjects),
                      "--trials-per-subject",str(args.gap1_trials),
                      "--multiway",str(args.gap1_multiway),
                      "--plots",str(args.gap1_plots),
                      "--output",str(args.gap1_dir),
                      "--seed",str(args.seed)])
        else:
            print("[Pipeline] Existing Gap1 verified by source hashes; reusing its frozen ranking",flush=True)
    if args.only_gap1:
        print("[Pipeline] Gap1 only complete",flush=True)
        return
    print("[Pipeline] Gap2+Gap3: matched numeric WY, fit in parallel columns",flush=True)
    # Shared controls passed explicitly; any remaining numeric-gap CLI options
    # are forwarded. Reject unknowns at the target parser, never silently ignore.
    run_gaps23(["--models",*models,"--gap1-dir",str(args.gap1_dir),
                "--angles",args.angles,"--seed",str(args.seed),*unknown])
    print("[Pipeline] All stages finished; Gap2/3 share the same WY design",flush=True)


if __name__=="__main__":
    main()
