"""Compare FINISHED effect_fingerprint runs across beta, without fitting any R.

Match axes using TRAIN subjects only. Report the held-out TEST stability of the
matched axes. The optional saved TEST-theta r² is for descriptive auditing,
not for choosing a winning rotation or modifying hyperparameters.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]


def load_result(folder):
    folder=Path(folder)
    if not (folder/"rotation__complete.json").is_file():
        raise FileNotFoundError("Incomplete effect_fingerprint run: "+str(folder))
    with np.load(folder/"rotation__matrix.npz",allow_pickle=False) as f:
        R=np.array(f["R"],float)
        source=str(f["source_dataset_sha256"])
    with np.load(folder/"rotation__U.npz",allow_pickle=False) as f:
        U=np.asarray(f["U"],float)
        tr=np.asarray(f["train_ids"],int)
        te=np.asarray(f["test_ids"],int)
    if U.ndim!=2 or not np.isfinite(U).all() or U.shape[1]!=len(R):
        raise ValueError("Incompatible U/R in "+str(folder))
    eval_path=folder/"rotation__test_evaluation.json"
    matched=None
    if eval_path.is_file():
        matched=json.loads(eval_path.read_text())["matched_mean_r2"]
    return dict(folder=folder,source_sha=source,U=U,R=R,
                train=tr,test=te,matched_mean_r2=matched)


def corr(A,B):
    a=np.asarray(A,float).copy()
    b=np.asarray(B,float).copy()
    a-=a.mean(axis=0);b-=b.mean(axis=0)
    denom=np.sqrt(np.sum(a*a,axis=0)[:,None]*np.sum(b*b,axis=0)[None,:])
    return np.divide(a.T@b,denom,out=np.zeros((a.shape[1],b.shape[1])),
                     where=denom>1e-12)


def compare_pair(a,b):
    if a["source_sha"]!=b["source_sha"]:
        raise ValueError("Different S1 datasets: cannot align respondents")
    if a["U"].shape!=b["U"].shape or not np.array_equal(a["train"],b["train"]):
        raise ValueError("Subject order or NP TRAIN splits differ")
    if not np.array_equal(a["test"],b["test"]):
        raise ValueError("NP TEST sets differ")
    C=corr(a["U"][a["train"]],b["U"][b["train"]])
    rows,cols=linear_sum_assignment(-np.abs(C))
    signs=np.where(C[rows,cols]>=0,1.,-1.)
    testC=corr(a["U"][a["test"]],b["U"][b["test"]])
    train_match=np.abs(C[rows,cols])
    test_match=signs*testC[rows,cols]
    return dict(run_a=str(a["folder"]),run_b=str(b["folder"]),
                train_axis_mean_abs_corr=float(train_match.mean()),
                train_axis_min_abs_corr=float(train_match.min()),
                heldout_test_axis_mean_signed_corr=float(test_match.mean()),
                heldout_test_axis_min_signed_corr=float(test_match.min()),
                axis_assignment_b=list(map(int,cols)),
                axis_sign_b=list(map(int,signs)),
                train_axis_correlations=train_match.tolist(),
                test_axis_correlations=test_match.tolist(),
                matched_test_theta_r2_a=a["matched_mean_r2"],
                matched_test_theta_r2_b=b["matched_mean_r2"])


def locate(root,model,condition,repeat):
    base=(Path(root)/"outputs"/"choice"/model/"rotations"/
          "effect_fingerprint"/condition/f"repeat{repeat:02d}")
    candidates=sorted(p for p in base.glob("run_*")
                      if (p/"rotation__complete.json").is_file())
    if len(candidates)!=1:
        raise ValueError(f"{len(candidates)} complete runs under {base}. "
                         "Provide explicit --runs directories to disambiguate.")
    return candidates[0]


def main(argv=None):
    parser=argparse.ArgumentParser(description="Compare only completed effect_fingerprint runs")
    parser.add_argument("--model",choices=("C1","C2","C3","CR3"),required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--conditions",nargs="+")
    group.add_argument("--runs",nargs="+")
    parser.add_argument("--repeat",type=int,default=1)
    args=parser.parse_args(argv)
    if args.runs:
        dirs=[Path(p) for p in args.runs]
    else:
        dirs=[locate(ROOT,args.model,c,args.repeat) for c in args.conditions]
    if len(dirs)<2:parser.error("Supply at least two finished effect_fingerprint runs")
    runs=[load_result(p) for p in dirs]
    results=[compare_pair(a,b) for a,b in combinations(runs,2)]
    identity=json.dumps([str(p.resolve()) for p in dirs],sort_keys=True).encode()
    code=hashlib.sha256(identity).hexdigest()[:12]
    output=ROOT/"outputs"/"choice"/args.model/"rotations"/"effect_fingerprint"/"comparisons"
    output.mkdir(parents=True,exist_ok=True)
    path=output/f"comparison_{code}.json"
    path.write_text(json.dumps(results,indent=2,allow_nan=False),encoding="utf-8")
    flat=[{k:v for k,v in result.items() if not isinstance(v,list)}
          for result in results]
    with (output/f"comparison_{code}.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]))
        writer.writeheader();writer.writerows(flat)
    print("Saved:",path)
    for result in results:
        print(result["run_a"],"vs",result["run_b"],
              "TRAIN mean |r|=",round(result["train_axis_mean_abs_corr"],4),
              "TEST mean signed r=",round(result["heldout_test_axis_mean_signed_corr"],4))
    return results


if __name__=="__main__":
    main()
