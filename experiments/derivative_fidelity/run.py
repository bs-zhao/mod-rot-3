"""Run the C1/C2/C3/CR3 derivative fidelity benchmark.

One self-contained output root: outputs/derivative_fidelity/. Existing outputs
and frozen checkpoints are read only. Use python -m with repository root cwd.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .data import DEFAULTS,load_case,make_function,input_pairs,strict_x_interior,save_metadata,ROOT
from .derivatives import BLOCKS,write_field,derivative_function
from .estimators import METHODS,build_estimator,fit_value_surrogate,aggregate_gaussian_stein_u
from .metrics import block_metrics,field_accuracy,metric_accuracy,assemble_energy
from .finite_scale import taylor_tests
from .report import make_report

BASE=ROOT/"outputs"/"derivative_fidelity"
DEFAULT_METHODS=("ad_mixed","ad_forward","fd2","hybrid","gaussian_smoothing")
HEAVY_METHODS=("fd4","richardson","local_poly","spectral_chebyshev","krr","gp","mlp")
FAMILY={"U":"U-only","X":"X-only","UU":"U-only","UX":"mixed",
        "XX":"X-only","UUU":"U-only","UUX":"mixed","UXX":"mixed","XXX":"X-only"}


def log_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=str),encoding="utf-8")


def npz_metrics(path,d):
    np.savez_compressed(path,**{k:np.asarray(v) for k,v in d.items()})


def load_metrics(path):
    a=np.load(path)
    return {k:np.asarray(a[k]) for k in a.files}


def compare_metrics(path, got, expected, block, source):
    records=[]
    for metric in expected:
        if metric not in got:
            records.append({"metric":metric,"block":block,"method":source,"status":"not_available"})
            continue
        try: verdict=metric_accuracy(got[metric],expected[metric])
        except Exception as exc:verdict={"status":"failed","reason":str(exc)}
        records.append({"metric":metric,"block":block,"method":source,**verdict})
    with open(path,"a",encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r)+"\n")
    return records


def set_domain(case,X,domain):
    if case.model!="CR3" or domain=="all":
        return X,{"domain":domain,"boundary_policy":"all derivative values are branch-dependent at non-smooth inputs"}
    mask=strict_x_interior(case,X)
    if X.ndim==2: X=X[mask]
    else:
        keep=np.all(mask,axis=0)
        X=X[:,keep]
    if len(X)==0 or (X.ndim==3 and X.shape[1]==0):
        raise RuntimeError("CR3 has no strict-interior X rows; use --domain all for U-only or redesign X probes")
    return X,{"domain":"strict_interior","retained_stimuli":X.shape[-2],
              "excluded_stimuli":"non-smooth X rows (zero amount / probability boundary)"}


def run_case(model,repeat,channel,args):
    case=load_case(model,repeat,args.subjects,args.stimuli,args.seed,args.grid,
                   args.device,args.beta,args.zdim,args.pca)
    folder=BASE/args.tag/model/f"repeat{repeat:02d}"/channel/("standardized" if args.standardized else "native")/args.grid/args.domain
    folder.mkdir(parents=True,exist_ok=True)
    save_metadata(folder,case)
    V,X=input_pairs(case,standardized=args.standardized)
    X,domain_info=set_domain(case,X,args.domain)
    log_json(folder/"domain.json",domain_info)
    log_json(folder/"config.json",{**vars(args),"model":model,"repeat":repeat,"channel":channel})
    f_true=make_function(case,"analytic_truth",channel,"float64",args.standardized)
    f_mapped=make_function(case,"analytic_mapped",channel,"float64",args.standardized)
    f_nn=make_function(case,"nn",channel,args.precision,args.standardized)
    truth_v,_=input_pairs(case,standardized=args.standardized,reference="truth")
    if args.blocks=="all":blocks=BLOCKS
    else:
        blocks=tuple(b.strip() for b in args.blocks.split(","))
        if any(b not in BLOCKS for b in blocks):raise ValueError("unknown block")
    selected=tuple(METHODS) if args.methods=="all" else tuple(x.strip() for x in args.methods.split(","))
    if any(m not in METHODS for m in selected):raise ValueError("unknown method")
    dtype=torch.float32 if args.precision=="float32" else torch.float64
    references={}
    # All reference fields are persisted, allowing later methods to resume.
    for source,fn,u in (("analytic_truth",f_true,truth_v),("analytic_mapped",f_mapped,V)):
        rf=folder/source
        rf.mkdir(exist_ok=True)
        metrics={}
        for block in blocks:
            destination=rf/f"{block}.npy"
            if args.force or not destination.exists():
                write_field(destination,fn,u,X,block,"mixed","float64",args.device,args.chunk)
            field=np.load(destination,mmap_mode="r")
            try:
                m=block_metrics(field,block)
                npz_metrics(rf/f"{block}_metrics.npz",m)
                metrics[block]=np.asarray(m["A03.rms"])[None,None,...]
            except Exception as exc:
                log_json(rf/f"{block}_metrics_error.json",{"exception":repr(exc),
                         "traceback":traceback.format_exc()})
                if args.fail_fast:raise
        if metrics:
            npz_metrics(rf/"joint_metrics.npz",assemble_energy(metrics))
        references[source]=rf
        log_json(rf/"run_manifest.json",{"status":"reference","source":source,"blocks":list(blocks)})

    if args.stage=="reference":return folder
    expected_dir=references["analytic_mapped"]
    for method in selected:
        mf=folder/method
        mf.mkdir(exist_ok=True)
        manifest={"status":"incomplete","method":method,"finished":{},"errors":{}}
        if (mf/"run_manifest.json").exists() and not args.force:
            manifest=json.loads((mf/"run_manifest.json").read_text(encoding="utf-8"))
        if method=="stein_gaussian_u":
            try:
                yy=X if X.ndim==2 else X[0]
                record=aggregate_gaussian_stein_u(f_nn,V[:min(8,len(V))],yy,
                            sigma=args.sigma,samples=args.stein_samples,seed=args.seed,device=args.device)
                npz_metrics(mf/"stein_gaussian_u.npz",record)
                manifest.update({"status":"aggregate_only","note":"Gaussian smoothed U-only expectation, not empirical pointwise derivative"})
            except Exception as exc:
                manifest["status"]="failed"; manifest["errors"]["stein"]=repr(exc)
                if args.fail_fast:raise
            log_json(mf/"run_manifest.json",manifest)
            continue
        if method=="hvp_directional":
            try:
                from .directional import directional_report
                report=directional_report(f_nn,V,X,seed=args.seed,
                            device=args.device,precision=args.precision,
                            n_points=args.directional_points)
                log_json(mf/"directional.json",report)
                manifest["status"]="contraction"
            except Exception as exc:
                manifest["status"]="failed";manifest["errors"]["directional"]=repr(exc)
                if args.fail_fast:raise
            log_json(mf/"run_manifest.json",manifest)
            continue

        h=f_nn; fit_meta={}
        if method in ("krr","gp","mlp"):
            try:
                tr=case.data["train_ids"]
                T=case.data["V"][tr]
                if args.standardized:T=(T-case.v_mu)/case.v_sd
                # Surrogate training uses TRAIN latent data and requested X grid only.
                h,fit_meta=fit_value_surrogate(f_nn,T,X,kind=method,
                      n_train=args.surrogate_train,n_centers=args.centers,
                      seed=args.seed,device=args.device,dtype=dtype,epochs=args.epochs)
                log_json(mf/"surrogate_fit.json",fit_meta)
            except Exception as exc:
                manifest["status"]="failed";manifest["errors"]["fit"]=repr(exc)
                log_json(mf/"run_manifest.json",manifest)
                if args.fail_fast:raise
                continue
        field_paths={}
        energy_proxies={}
        for block in blocks:
            if manifest.get("finished",{}).get(block)=="complete" and not args.force:
                field_paths[block]=mf/f"{block}.npy"
                if (mf/f"{block}_metrics.npz").exists():
                    m=load_metrics(mf/f"{block}_metrics.npz")
                    energy_proxies[block]=np.asarray(m["A03.rms"])[None,None,...]
                continue
            started=time.monotonic()
            manifest["finished"][block]="running"
            log_json(mf/"run_manifest.json",manifest)
            try:
                estimator,est_meta=build_estimator(method,h,block,case.d,case.p,
                    step=args.step,sigma=args.sigma,mc=args.mc,
                    seed=args.seed,device=args.device,dtype=dtype)
                dest=mf/f"{block}.npy"
                write_field(dest,f_nn,V,X,block,precision=args.precision,
                    device=args.device,chunk=args.chunk,pointwise_fn=estimator)
                got=np.load(dest,mmap_mode="r")
                ref=np.load(expected_dir/f"{block}.npy",mmap_mode="r")
                observed=block_metrics(got,block)
                npz_metrics(mf/f"{block}_metrics.npz",observed)
                reference_metrics=load_metrics(expected_dir/f"{block}_metrics.npz")
                errors=field_accuracy(got,ref,zero_tol=args.zero_tol)
                errors.update({"elapsed_seconds":time.monotonic()-started,
                               "method":method,"block":block,"family":FAMILY[block]})
                log_json(mf/"field_comparisons.json",{
                     **(json.loads((mf/"field_comparisons.json").read_text(encoding="utf-8"))
                        if (mf/"field_comparisons.json").exists() else {}),
                        block:errors})
                compare_metrics(mf/"metric_comparisons.jsonl",observed,reference_metrics,block,method)
                energy_proxies[block]=np.asarray(observed["A03.rms"])[None,None,...]
                manifest["finished"][block]="complete"
                manifest.setdefault("metadata",{})[block]=est_meta
                if args.storage=="metrics":
                    dest.unlink(missing_ok=True)
                else:
                    field_paths[block]=dest
            except Exception as exc:
                manifest["finished"][block]="failed"
                manifest["errors"][block]={"error":repr(exc),"traceback":traceback.format_exc()}
                if args.fail_fast:
                    log_json(mf/"run_manifest.json",manifest)
                    raise
            finally:
                log_json(mf/"run_manifest.json",manifest)
        if energy_proxies:
            joint=assemble_energy(energy_proxies)
            npz_metrics(mf/"joint_metrics.npz",joint)
            if (expected_dir/"joint_metrics.npz").exists():
                compare_metrics(mf/"metric_comparisons.jsonl",joint,
                    load_metrics(expected_dir/"joint_metrics.npz"),"joint",method)
        manifest["status"]="complete" if all(manifest["finished"].get(b)=="complete" for b in blocks) else "incomplete"
        log_json(mf/"run_manifest.json",manifest)
    if args.taylor:
        for label,fn in (("analytic_mapped",f_mapped),("nn",f_nn)):
            try:
                rows=taylor_tests(fn,V,X,device=args.device,
                         n_points=args.taylor_points,precision="float64" if label=="analytic_mapped" else args.precision)
                log_json(folder/label/"taylor.json",rows)
            except Exception as exc:
                log_json(folder/label/"taylor_error.json",{"error":repr(exc)})
                if args.fail_fast:raise
    return folder


def parse():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models",nargs="+",default=["C1","C2","C3","CR3"])
    ap.add_argument("--repeats",nargs="+",type=int,default=[1,2])
    ap.add_argument("--channels",nargs="+",default=["logit","prob"])
    ap.add_argument("--stage",choices=("reference","run","report"),default="run")
    ap.add_argument("--tag",default="baseline")
    ap.add_argument("--subjects",type=int,default=32)
    ap.add_argument("--stimuli",type=int,default=64)
    ap.add_argument("--seed",type=int,default=20261116)
    ap.add_argument("--grid",choices=("shared","paired"),default="shared")
    ap.add_argument("--domain",choices=("interior","all"),default="interior")
    ap.add_argument("--standardized",action="store_true")
    ap.add_argument("--precision",choices=("float32","float64"),default="float64")
    ap.add_argument("--device",default="auto")
    ap.add_argument("--beta",type=float,default=None)
    ap.add_argument("--zdim",type=int,default=None)
    ap.add_argument("--pca",type=int,default=0)
    ap.add_argument("--blocks",default="all")
    ap.add_argument("--methods",default=",".join(DEFAULT_METHODS))
    ap.add_argument("--step",type=float,default=.03)
    ap.add_argument("--sigma",type=float,default=.05)
    ap.add_argument("--mc",type=int,default=16)
    ap.add_argument("--stein-samples",type=int,default=2048)
    ap.add_argument("--surrogate-train",type=int,default=600)
    ap.add_argument("--centers",type=int,default=120)
    ap.add_argument("--epochs",type=int,default=180)
    ap.add_argument("--chunk",type=int,default=64)
    ap.add_argument("--zero-tol",type=float,default=1e-8)
    ap.add_argument("--directional-points",type=int,default=32)
    ap.add_argument("--storage",choices=("metrics","all"),default="metrics")
    ap.add_argument("--taylor",action="store_true")
    ap.add_argument("--taylor-points",type=int,default=20)
    ap.add_argument("--force",action="store_true")
    ap.add_argument("--fail-fast",action="store_true")
    return ap.parse_args()


def main():
    args=parse()
    if args.device=="auto": args.device="cuda" if torch.cuda.is_available() else "cpu"
    if args.stage=="report":
        count=make_report(BASE/args.tag)
        print("Report written:",BASE/args.tag,count)
        return
    for model in args.models:
        for repeat in args.repeats:
            for channel in args.channels:
                print("START",model,repeat,channel,flush=True)
                try:
                    path=run_case(model,repeat,channel,args)
                    print("DONE",path,flush=True)
                except Exception as exc:
                    print("ERROR",model,repeat,channel,repr(exc),flush=True)
                    if args.fail_fast:raise
    print("Report", make_report(BASE/args.tag), BASE/args.tag/"REPORT.md")


if __name__=="__main__":
    main()
