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
from .metrics_extended import enrich,summarize_fullw
from .finite_scale import taylor_tests
from .audits import audit_derivatives,audit_probability_chain
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
        try:
            a=np.asarray(got[metric]); b=np.asarray(expected[metric])
            if a.dtype.kind not in "fiu" or b.dtype.kind not in "fiu":
                verdict={"status":"not_applicable","reason":"non_numeric_status"}
            else:
                verdict=metric_accuracy(a,b)
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
    f_nn64=make_function(case,"nn",channel,"float64",args.standardized)
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
    for source,fn,u in (("analytic_truth",f_true,truth_v),
                        ("analytic_mapped",f_mapped,V),
                        ("nn_ad64",f_nn64,V)):
        rf=folder/source
        rf.mkdir(exist_ok=True)
        metrics={}
        for block in blocks:
            destination=rf/f"{block}.npy"
            if args.force or not destination.exists():
                write_field(destination,fn,u,X,block,"mixed","float64",args.device,args.chunk)
            field=np.load(destination,mmap_mode="r")
            try:
                m=enrich(field,block,block_metrics(field,block))
                npz_metrics(rf/f"{block}_metrics.npz",m)
                metrics[block]=np.asarray(m["A03.rms"])[None,None,...]
            except Exception as exc:
                log_json(rf/f"{block}_metrics_error.json",{"exception":repr(exc),
                         "traceback":traceback.format_exc()})
                if args.fail_fast:raise
        if metrics:
            npz_metrics(rf/"joint_metrics.npz",assemble_energy(metrics))
        try:
            field_arrays={b:np.load(rf/f"{b}.npy",mmap_mode="r") for b in blocks}
            fullw=summarize_fullw(field_arrays)
            if fullw:npz_metrics(rf/"fullw_metrics.npz",fullw)
        except Exception as exc:
            log_json(rf/"fullw_error.json",{"error":repr(exc),"traceback":traceback.format_exc()})
            if args.fail_fast:raise
        references[source]=rf
        log_json(rf/"run_manifest.json",{"status":"reference","source":source,
                  "reference_kind":("nn_numerical_reference" if source=="nn_ad64" else
                                    "true_theta" if source=="analytic_truth" else "supervised_mapped_proxy"),
                  "blocks":list(blocks)})

    if getattr(args,"audit",True):
        for source,fn,u in (("analytic_truth",f_true,truth_v),
                            ("analytic_mapped",f_mapped,V),("nn_ad64",f_nn64,V)):
            try:
                check=audit_derivatives(fn,u,X,device=args.device,
                        precision="float64",points=getattr(args,"audit_points",2))
                log_json(folder/source/"numerical_audit.json",check)
                if channel=="logit":
                    prob=make_function(case,("nn" if source=="nn_ad64" else source),
                                       "prob","float64",args.standardized)
                    chain=audit_probability_chain(fn,prob,u,X,device=args.device,
                              precision="float64",points=getattr(args,"audit_points",2))
                    log_json(folder/source/"sigmoid_chain_audit.json",chain)
            except Exception as exc:
                log_json(folder/source/"audit_error.json",
                         {"exception":repr(exc),"traceback":traceback.format_exc()})
                if args.fail_fast:raise

    if args.stage=="reference":return folder
    expected_dir=references["analytic_mapped"]
    nn_reference_dir=references["nn_ad64"]
    # G1->G2 is an aggregate structural comparison only (different inputs).
    # Never compute pointwise derivative error between theta and V.
    diagnostics=folder/"truth_vs_mapped_structural.jsonl"
    diagnostics.write_text("",encoding="utf-8")
    for block in blocks:
        left=load_metrics(references["analytic_mapped"]/f"{block}_metrics.npz")
        right=load_metrics(references["analytic_truth"]/f"{block}_metrics.npz")
        compare_metrics(diagnostics,left,right,block,"mapped_vs_truth_structure")
    for method in selected:
        mf=folder/method
        mf.mkdir(exist_ok=True)
        manifest={"status":"incomplete","method":method,"finished":{},"errors":{}}
        if (mf/"run_manifest.json").exists() and not args.force:
            manifest=json.loads((mf/"run_manifest.json").read_text(encoding="utf-8"))
        if args.force:
            for stale in ("metric_comparisons.jsonl","metric_comparisons_nn_ad.jsonl"):
                (mf/stale).unlink(missing_ok=True)
        if method=="stein_gaussian_u":
            try:
                yy=X if X.ndim==2 else X[0]
                record=aggregate_gaussian_stein_u(f_nn,V[:min(8,len(V))],yy,
                            sigma=args.sigma,samples=args.stein_samples,seed=args.seed,device=args.device)
                npz_metrics(mf/"stein_gaussian_u.npz",record)
                ref_record=aggregate_gaussian_stein_u(f_mapped,V[:min(8,len(V))],yy,
                            sigma=args.sigma,samples=args.stein_samples,
                            seed=args.seed,device=args.device)
                npz_metrics(mf/"stein_analytic_mapped.npz",ref_record)
                log_json(mf/"aggregate_comparison.json",
                      {k:metric_accuracy(record[k],ref_record[k]) for k in ref_record})
                manifest.update({"status":"aggregate_only",
                    "note":"Gaussian-smoothed U-only derivative expectations at NN and mapped analytic; both integrated with the same MC probes, not empirical pointwise derivatives"})
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
                ref_report=directional_report(f_mapped,V,X,seed=args.seed,
                            device=args.device,precision="float64",
                            n_points=args.directional_points)
                log_json(mf/"directional_analytic_mapped.json",ref_report)
                log_json(mf/"directional_comparison.json",
                         {k:metric_accuracy(report[k],ref_report[k])
                          for k in ref_report if k.endswith("_mean") or k.endswith("_rms")})
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
                nn_ref=np.load(nn_reference_dir/f"{block}.npy",mmap_mode="r")
                observed=enrich(got,block,block_metrics(got,block))
                npz_metrics(mf/f"{block}_metrics.npz",observed)
                reference_metrics=load_metrics(expected_dir/f"{block}_metrics.npz")
                errors=field_accuracy(got,ref,zero_tol=args.zero_tol)
                against_nn_ad=field_accuracy(got,nn_ref,zero_tol=args.zero_tol)
                errors.update({f"{key}_vs_nn_ad64":val for key,val in against_nn_ad.items()
                               if key in ("mae","rmse","relative_l2","pearson","bias")})
                errors.update({"elapsed_seconds":time.monotonic()-started,
                               "method":method,"block":block,"family":FAMILY[block]})
                log_json(mf/"field_comparisons.json",{
                     **(json.loads((mf/"field_comparisons.json").read_text(encoding="utf-8"))
                        if (mf/"field_comparisons.json").exists() else {}),
                        block:errors})
                compare_metrics(mf/"metric_comparisons.jsonl",observed,reference_metrics,block,method)
                compare_metrics(mf/"metric_comparisons_nn_ad.jsonl",observed,
                                load_metrics(nn_reference_dir/f"{block}_metrics.npz"),block,method)
                energy_proxies[block]=np.asarray(observed["A03.rms"])[None,None,...]
                manifest["finished"][block]="complete"
                manifest.setdefault("metadata",{})[block]=est_meta
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
                compare_metrics(mf/"metric_comparisons_nn_ad.jsonl",joint,
                    load_metrics(nn_reference_dir/"joint_metrics.npz"),"joint",method)
        # Genuine full-W gradient/Hessian/third tensor, from the SAME points.
        # This is separate from full-W block-energy aggregation.
        try:
            present={b:np.load(mf/f"{b}.npy",mmap_mode="r")
                     for b in blocks if (mf/f"{b}.npy").exists()}
            fullw=summarize_fullw(present)
            if fullw:
                npz_metrics(mf/"fullw_metrics.npz",fullw)
                if (expected_dir/"fullw_metrics.npz").exists():
                    compare_metrics(mf/"metric_comparisons.jsonl",fullw,
                          load_metrics(expected_dir/"fullw_metrics.npz"),"full_W",method)
                    compare_metrics(mf/"metric_comparisons_nn_ad.jsonl",fullw,
                          load_metrics(nn_reference_dir/"fullw_metrics.npz"),"full_W",method)
        except Exception as exc:
            manifest["errors"]["full_W"]={"error":repr(exc),"traceback":traceback.format_exc()}
            if args.fail_fast:raise
        if args.storage=="metrics":
            for path in field_paths.values():
                Path(path).unlink(missing_ok=True)
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
    ap.add_argument("--no-audit",action="store_false",dest="audit",default=True)
    ap.add_argument("--audit-points",type=int,default=2)
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
