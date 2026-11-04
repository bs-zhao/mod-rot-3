"""Discover local Fisher one-dimensional nullspaces using X-only regions.

No CPT parameter labels are used to create or select candidate stimulus
conditions. This is an ORACLE analysis: response gradients use theta.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.optimize import linear_sum_assignment
from conditional_nullspace_oracle import (
    ROOT, PARAMS, _digest, _json, response_and_gradient,
    operator_matrices, supplement_stimuli,
)


def x_regions(X, minimum=12):
    """Single-value bins and cross-feature intersections, built from X only."""
    conditions=[]
    singles=[]
    for j in range(X.shape[1]):
        vals=np.unique(np.round(X[:,j],10))
        if len(vals)>32:
            values=np.quantile(X[:,j],[0,.25,.5,.75,1.])
            groups=[(f"x{j}:[{lo:g},{hi:g}]",(X[:,j]>=lo)&(X[:,j]<=hi))
                    for lo,hi in zip(values[:-1],values[1:])]
        else:
            groups=[(f"x{j}={v:g}",np.isclose(X[:,j],v,atol=1e-9,rtol=0))
                    for v in vals]
        singles.extend((key,mask,j) for key,mask in groups
                       if minimum<=int(mask.sum())<len(X))
    seen=set()
    for key,mask,j in singles:
        signature=np.packbits(mask).tobytes()
        if signature not in seen:
            seen.add(signature);conditions.append((key,mask))
    for (ka,ma,ja),(kb,mb,jb) in itertools.combinations(singles,2):
        if ja==jb:continue
        mask=ma&mb
        if not minimum<=int(mask.sum())<len(X):continue
        signature=np.packbits(mask).tobytes()
        if signature in seen:continue
        seen.add(signature);conditions.append((ka+" & "+kb,mask))
    return conditions


def explore(X,train,val,sd,min_trials=12,threshold=1e-7,gap=1e-3):
    def trial_bank(subjects):
        f,G=response_and_gradient(X,subjects,sd)
        w=expit(f)*(1-expit(f))
        return np.einsum("nt,nti,ntj->tij",w,G,G)/len(subjects)
    train_bank=trial_bank(train)
    val_bank=trial_bank(val)
    candidates=x_regions(X,min_trials)
    passed=[]
    for key,mask in candidates:
        A=train_bank[mask].mean(0); B=val_bank[mask].mean(0)
        e,V=np.linalg.eigh(A);ev,Vv=np.linalg.eigh(B)
        r=e/max(e[-1],1e-30);rv=ev/max(ev[-1],1e-30)
        cosine=abs(float(V[:,0]@Vv[:,0]))
        if r[0]>threshold or rv[0]>threshold or min(r[1],rv[1])<gap or cosine<.98:continue
        score=float(np.sqrt(mask.mean())*min(r[1],rv[1])*cosine)
        passed.append(dict(condition=key,n_trials=int(mask.sum()),
                           eigengap=float(r[1]),val_eigengap=float(rv[1]),
                           null_vector=V[:,0].tolist(),score=score))
    passed.sort(key=lambda v:-v["score"])
    clusters=[]
    for item in passed:
        v=np.asarray(item["null_vector"])
        if any(abs(v@np.asarray(c["null_vector"]))>.985 for c in clusters):continue
        clusters.append(item)
    shape=operator_matrices(X,train,sd)["shape"]
    _,V=np.linalg.eigh(shape);tau=V[:,0]
    best=None
    eligible=[c for c in clusters if abs(np.asarray(c["null_vector"])@tau)<.2]
    for a,b in itertools.combinations(eligible,2):
        va=np.asarray(a["null_vector"]);vb=np.asarray(b["null_vector"])
        cross=abs(float(va@vb))
        if cross>=.2:continue
        score=min(a["score"],b["score"])*(1-cross)
        if best is None or score>best["score"]:
            best=dict(score=float(score),conditions=[a["condition"],b["condition"]],
                      axes=[a["null_vector"],b["null_vector"]])
    result=dict(n_candidates=len(candidates),n_valid=len(passed),
                candidate_conditions=passed[:25],clusters=clusters[:30],
                shape_null_vector=tau.tolist(),selected_triad=best)
    if best is not None:
        axes=np.abs(np.vstack([tau,*best["axes"]]))
        rr,cc=linear_sum_assignment(-axes)
        result["posthoc_oracle_alignment"]=dict(parameter_names=[PARAMS[j] for j in cc],
            mean_cosine=float(axes[rr,cc].mean()),absolute_loadings=axes.tolist())
    return result


def selftest():
    from s0_cr2_design_recovery import sample_prior
    rng=np.random.default_rng(7401)
    X=supplement_stimuli(80,7402)
    candidates=x_regions(X,12)
    conditions=[name for name,_ in candidates]
    if not any("x1=0" in name and "x4=0" in name for name in conditions):
        raise AssertionError("Unsupervised X scan failed to find gains-only support")
    if not any("x2=0.5" in name and "x5=0.5" in name for name in conditions):
        raise AssertionError("Unsupervised X scan failed to find gamma-neutral support")
    theta=sample_prior(96,4,rng)
    result=explore(X,theta[:48],theta[48:],theta.std(0),12)
    triad=result["selected_triad"]
    if triad is None:
        raise AssertionError("Failed to discover three orthogonal null directions")
    A=result["posthoc_oracle_alignment"]
    if A["mean_cosine"]<.98:
        raise AssertionError("Recovered directions not aligned to generating parameter axes")
    print("Automatic X-only region scan + heldout nullspace clustering PASS")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    p.add_argument("--dataset",default=None)
    p.add_argument("--seed",type=int,default=20261023)
    p.add_argument("--subjects",type=int,default=96)
    p.add_argument("--min-trials",type=int,default=12)
    p.add_argument("--supplement-count",type=int,default=64)
    args=p.parse_args()
    if args.selftest:
        selftest()
        return
    file=Path(args.dataset) if args.dataset else (
        ROOT/"outputs"/"choice"/"CR2"/"s1_s4"/"4P"/"s1_generate"/"s1_generate__dataset.npz")
    with np.load(file,allow_pickle=False) as f:
        X=np.asarray(f["x"][:500],float)
        theta=np.asarray(f["parameters"],float)
    if X.shape!=(500,6) or theta.shape[1]!=4:raise ValueError("Expect CR2-4P")
    rng=np.random.default_rng(args.seed)
    ids=rng.choice(len(theta),min(args.subjects,len(theta)),replace=False)
    mid=len(ids)//2
    sd=theta.std(0)
    report=dict(data_sha256=_digest(file),oracle=True,no_NP=True,no_ICA=True,
                parameter_names_used_only_for_posthoc_evaluation=True)
    for name,xx in (("observed_500",X),
                    ("augmented_feasibility_only",
                     np.vstack((X,supplement_stimuli(args.supplement_count,args.seed+17))))):
        report[name]=explore(xx,theta[ids[:mid]],theta[ids[mid:]],sd,args.min_trials)
        print(name,"valid conditions",report[name]["n_valid"],"triad",
              report[name]["selected_triad"] is not None,flush=True)
    out=ROOT/"outputs"/"choice"/"CR2"/"oracle_functional_roles"/"conditional_nullspace"
    out.mkdir(parents=True,exist_ok=True)
    path=out/"automatic_nullspace_discovery__report.json"
    path.write_text(json.dumps(report,indent=2,default=_json),encoding="utf-8")
    print(path)


if __name__=="__main__":
    main()
