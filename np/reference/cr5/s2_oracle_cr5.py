"""CR5 Oracle MLE + Fisher; EXACT CR4 estimator, CR3 stimulus bank.

This is intentionally a narrow wrapper around the existing CR4 oracle
implementations, not another likelihood/MLE algorithm. Both benchmark
designs are evaluated using identical parameter priors, fitting starts,
Fisher scaling and random seed policy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/"cr4"),str(HERE)]
from s2_oracle_cr4 import evaluate,fisher_diagnostics,sample_theta
from s0_s1_cr5 import BANK,OUT,validate_immutable_bank,hash_file


def main(argv=None):
    ap=argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--subjects",type=int,default=100)
    ap.add_argument("--starts",type=int,default=4)
    ap.add_argument("--seed",type=int,default=20261023)
    ap.add_argument("--fisher-subjects",type=int,default=128)
    args=ap.parse_args(argv)
    if args.subjects<3 or args.starts<1 or args.fisher_subjects<3:
        ap.error("subjects/fisher-subjects >= 3, starts >= 1")
    x,codes,names,design=validate_immutable_bank()
    ref_t=sample_theta(args.fisher_subjects,np.random.default_rng(args.seed+810))
    fisher=fisher_diagnostics(x,ref_t)
    output,theta,mle,y=evaluate(x,args.subjects,args.starts,args.seed+719)
    output.update(model="CR5",fixed_alpha=1.0,
                  X_source="exact_CR3_500_including_special_conditions",
                  cr3_source_sha256=design["cr3_source_sha256"],
                  bank_sha256=hash_file(BANK),fisher=fisher,
                  type_counts=design["source_condition_counts"],
                  estimator="reused np/reference/cr4/s2_oracle_cr4.py (identical priors/bounds/seeds)")
    out=OUT/"s2_precheck"
    out.mkdir(parents=True,exist_ok=True)
    (out/"s2_precheck__oracle_recovery.json").write_text(
        json.dumps(output,indent=2),encoding="utf-8")
    np.savez_compressed(out/"s2_precheck__oracle_subjects.npz",
                        theta_true=theta,theta_mle=mle,choices=y)
    print("CR5 Oracle recovery [lambda,gamma,tau]:",np.round(output["pearson_r"],4),flush=True)
    print("CR5 Fisher min eigen p01:",fisher["min_eigenvalue_p01"],flush=True)


if __name__=="__main__":
    main()
