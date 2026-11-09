"""Paired Gap2+Gap3 with matched W, estimates and rotation-score comparisons.

Usage:
  python math/10_multimodel_hessian_simplicity/run_gaps.py --models C1 C2 C3 CR3
Requires the original S1/S3/S5 assets; does not train NP or use torch
autodiff. Gap1 results are needed to freeze the metric shortlist.
"""
from __future__ import annotations
import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np

from oracle import DIMENSIONS
from run import ROOT, rotation, rotate_h, angles_from_spec, write_csv
from metrics import compute_metrics
from selected_metrics import compute_selected
from numeric import (resolve_latent, choose_subject_trials, numeric_channels,
                     sieve_fit, rbf_fit, analytic_test_hessian, compare_field)

DEFAULT_OUT=ROOT/"outputs"/"math"/"10_multimodel_hessian_simplicity"

CHANNELS=("logit","probability")
SOURCES=("analytic_T_hat","formula_Y_sieve","decoder_Y_sieve")


def _csv(path):
    with Path(path).open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def choose_metrics(gap1_root,model,limit=50,global_limit=80):
    """Frozen Gap1 shortlist, including common and per-model winners.

    Exclude value_* (numerical estimators here only supply Hessians, not
    callable counterfactual Y for every rotated coordinate) and metrics
    requiring 3rd/4th derivatives that the current numerical fits do not
    estimate. No truth leakage from Gap2/3 scores.
    """
    loc=Path(gap1_root)/model/"ranking_by_model_channel.csv"
    glob=Path(gap1_root)/"cross_model_ranking.csv"
    if not loc.is_file() or not glob.is_file():
        raise FileNotFoundError(f"Run Gap1 first; missing {loc if not loc.is_file() else glob}")
    top=set()
    def compatible(name):
        return (name.endswith("_loss") and not name.startswith("value_")
                and not any(q in name for q in ("third","fourth")))
    for ch in CHANNELS:
        rows=[r for r in _csv(loc) if r["channel"]==ch and compatible(r["metric"])]
        rows=sorted(rows,key=lambda r:(-float(r["success_fraction"]),
                                      int(r["flat_planes"]),r["metric"]))
        top.update(r["metric"] for r in rows[:limit])
    rows=[r for r in _csv(glob) if compatible(r["metric"]) and int(r["full_coverage"])==1]
    rows=sorted(rows,key=lambda r:(-float(r["min_success_fraction"]),
                                  -float(r["mean_success_fraction"]),
                                  int(r["multiway_better_than_true"]),r["metric"]))
    top.update(r["metric"] for r in rows[:global_limit])
    # Additional predefined structural controls, never selected on Gap2/3 truth.
    top.update((
        "zz_rms_offdiag_l1_loss","zz_rms_offdiag_l2_loss",
        "zx_rms_total_l1_loss","zx_rms_row_tail1_k1_loss",
        "zx_rms_row_tail2_k1_loss","zx_rms_row_tail2_k2_loss",
        "zx_rms_row_tail2sq_k2_loss","zx_rms_row_relative_tail2_k2_loss",
        "zx_rms_row_softl0_rational_loss","zx_rms_column_logsum_loss",
        "zz_rms_row_off_tail2_k1_loss","zz_local_offdiag_energy_loss",
        "zx_localfirst_tail2_k1_loss","zx_gram_offdiag_l2sq_loss",
        "zx_local_frobenius_sq_diagnostic"))
    return sorted(top)


def select_estimator(method,parts,y,args,seed):
    """Choose ONE estimator configuration across BOTH sources, both Y channels.

    Validation criterion: mean of four normalized Y RMSEs; never Hessian
    error, metric recovery, theoretical axis, or TEST labels.
    """
    configs=[]
    if method=="sieve":
        for degree in args.degree_grid:
            for lam in args.ridge_grid:
                configs.append(dict(degree=degree,ridge=lam))
    elif method=="rbf":
        for bandwidth in args.bandwidth_grid:
            for lam in args.rbf_ridge_grid:
                configs.append(dict(bandwidth=bandwidth,ridge=lam,landmarks=args.landmarks,seed=seed))
    else:
        raise ValueError(method)
    if not configs:
        raise ValueError(f"{method}: empty hyperparameter grid")
    sd=np.maximum(np.std(y["validation"],axis=0),1e-3)
    candidates=[]
    best=None
    for conf in configs:
        fit=sieve_fit(parts,y,**conf) if method=="sieve" else rbf_fit(parts,y,**conf)
        score=float(np.mean(fit["validation_rmse"]/sd))
        row=dict(method=method,**conf,validation_score=score,
                 **{f"val_rmse_{k}":float(v) for k,v in zip(
                     ("formula_logit","formula_probability","nn_logit","nn_probability"),
                     fit["validation_rmse"])})
        candidates.append(row)
        if best is None or score<best[0]:
            best=(score,conf,fit)
    return best[2],best[1],candidates


def curve_metrics(zz,zx,R,allowed,metric_mode="shortlist"):
    a,b=rotate_h(zz,zx,R)
    metrics=(compute_metrics(a,b) if metric_mode=="all" else compute_selected(a,b,allowed))
    return {k:metrics[k] for k in allowed if k in metrics}


def make_curves(model,method,ref,fit,angles,shortlist,planes="all",metric_mode="shortlist"):
    """All channels/sources computed from exactly the SAME TEST W anchors."""
    d=ref["logit"][0].shape[1]
    curves=[]
    sources={
        "analytic_T_hat":ref,
        "gap2_formula_Y":{},
        "gap3_nn_Y":{},
    }
    for ch,ci in zip(CHANNELS,(0,1)):
        sources["gap2_formula_Y"][ch]=(fit["zz"][:,ci],fit["zx"][:,ci])
        sources["gap3_nn_Y"][ch]=(fit["zz"][:,ci+2],fit["zx"][:,ci+2])
    for i in range(d):
        for j in range(i+1,d):
            plane=f"{i+1}{j+1}"
            if planes!="all" and plane not in planes.split(","):
                continue
            for angle in angles:
                R=rotation(d,i,j,angle)
                for ch in CHANNELS:
                    for source,bych in sources.items():
                        zz,zx=bych[ch]
                        vals=curve_metrics(zz,zx,R,shortlist,metric_mode)
                        curves.append(dict(model=model,method=method,channel=ch,
                                           source=source,plane=plane,angle=angle,
                                           **vals))
    return curves


def compare_landscapes(curves,shortlist,endpoint_tolerance=10):
    """Gap2-vs-oracle and Gap3-vs-Formula/Oracle shape and minimum agreement."""
    groups={}
    for r in curves:
        groups.setdefault((r["model"],r["method"],r["channel"],r["plane"],r["source"]),[]).append(r)
    res=[]
    for (model,method,ch,plane,source),rows in groups.items():
        if source!="analytic_T_hat":
            continue
        base=sorted(rows,key=lambda r:float(r["angle"]))
        bysource={}
        for src in ("analytic_T_hat","gap2_formula_Y","gap3_nn_Y"):
            bysource[src]=sorted(groups[(model,method,ch,plane,src)],
                                 key=lambda r:float(r["angle"]))
        angles=np.array([float(r["angle"]) for r in base])
        for metric in shortlist:
            if metric not in base[0]:
                continue
            y=np.array([float(r[metric]) for r in bysource["analytic_T_hat"]])
            f=np.array([float(r[metric]) for r in bysource["gap2_formula_Y"]])
            n=np.array([float(r[metric]) for r in bysource["gap3_nn_Y"]])
            target_scale=max(float(np.ptp(y)),1e-12)
            def stats(a,b):
                aa=a-a.mean(); bb=b-b.mean()
                na=float(np.linalg.norm(aa));nb=float(np.linalg.norm(bb))
                pearson=float(aa@bb/(na*nb)) if na>1e-10 and nb>1e-10 else float("nan")
                mina=float(angles[np.argmin(a)]);minb=float(angles[np.argmin(b)])
                flat_a=bool(np.ptp(a)<1e-9*max(1.,abs(float(np.max(a)))))
                flat_b=bool(np.ptp(b)<1e-9*max(1.,abs(float(np.max(b)))))
                return dict(curve_r=pearson,min_angle_a=mina,min_angle_b=minb,
                            minima_distance_degrees=abs(mina-minb),
                            normalized_shape_rmse=float(np.sqrt(np.mean(
                                ((a-a.min())/max(float(np.ptp(a)),1e-12)
                                 -(b-b.min())/max(float(np.ptp(b)),1e-12))**2))),
                            a_flat=int(flat_a),b_flat=int(flat_b),
                            a_near_true_axis=int(not flat_a and
                                min(abs(mina),abs(90-mina))<=endpoint_tolerance+1e-8),
                            b_near_true_axis=int(not flat_b and
                                min(abs(minb),abs(90-minb))<=endpoint_tolerance+1e-8))
            for name,a,b in (("gap2_vs_analytic",y,f),
                             ("gap3_vs_gap2",f,n),
                             ("gap3_vs_analytic",y,n)):
                res.append(dict(model=model,method=method,channel=ch,plane=plane,
                                comparison=name,metric=metric,**stats(a,b)))
    return res


def plot_selected(curves,ranking,output,plots=5):
    if plots<=0:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=Path(output)/"plots";out.mkdir(parents=True,exist_ok=True)
    scores={}
    for row in ranking:
        key=(row["channel"],row["metric"])
        scores.setdefault(key,[]).append(row)
    ranked=sorted(scores,key=lambda k:-np.nanmean(
        [float(r.get("curve_r","nan")) for r in scores[k] if r["comparison"]=="gap3_vs_analytic"]))
    for ch,metric in ranked[:plots]:
        planes=sorted({r["plane"] for r in curves})
        for plane in planes:
            fig,ax=plt.subplots(figsize=(7,4))
            for source,label in (("analytic_T_hat","exact oracle"),
                                 ("gap2_formula_Y","fitted formula-Y"),
                                 ("gap3_nn_Y","fitted frozen-NN-Y")):
                group=sorted((r for r in curves if r["source"]==source and r["channel"]==ch
                              and r["plane"]==plane),key=lambda r:r["angle"])
                if not group or metric not in group[0]:
                    continue
                ax.plot([r["angle"] for r in group],[r[metric] for r in group],label=label)
            ax.set(xlabel="Rotation angle (degrees)",ylabel="Candidate loss",
                   title=f"{ch}/{plane}: {metric}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out/f"{ch}_{plane}_{metric[:80]}.png",dpi=130)
            plt.close(fig)


def process_model(model,args,root,out):
    d,p=DIMENSIONS[model]
    beta=args.beta_cr3 if model=="CR3" else args.beta_c
    zdim=d if args.zdim is None else args.zdim
    mapping=resolve_latent(root,model,beta,zdim,args.repeat,args.lambda_sparse,
                            args.context_min,args.context_max,pca=bool(args.pca))
    parts,domain=choose_subject_trials(model,mapping,args,root)
    # Compute both Y sources once, jointly across the exactly same sets.
    merged_t=np.concatenate([parts[k]["t"] for k in ("train","validation","test")])
    merged_x=np.concatenate([parts[k]["x"] for k in ("train","validation","test")])
    merged_z=np.concatenate([parts[k]["z"] for k in ("train","validation","test")])
    print(f"[{model}] generating four Y channels on shared W (formula & NN)",flush=True)
    full=numeric_channels(merged_t,merged_x,merged_z,model,
                          mapping["checkpoint"],device=args.device,batch_size=args.decoder_batch)
    y={};start=0
    for split in ("train","validation","test"):
        n=parts[split]["nsubjects"]*parts[split]["ntrials"]
        y[split]=full.reshape(-1,4)[start:start+n]
        start+=n
    assert start==full.shape[0]*full.shape[1]
    Ytruth=y["test"][:,:2]
    Ynn=y["test"][:,2:]
    yeval=dict(
        formula_vs_nn_logit_rmse=float(np.sqrt(np.mean((Ytruth[:,0]-Ynn[:,0])**2))),
        formula_vs_nn_probability_rmse=float(np.sqrt(np.mean((Ytruth[:,1]-Ynn[:,1])**2))),
        formula_vs_nn_logit_r=float(np.corrcoef(Ytruth[:,0],Ynn[:,0])[0,1]),
        formula_vs_nn_probability_r=float(np.corrcoef(Ytruth[:,1],Ynn[:,1])[0,1]))
    test_oracle=analytic_test_hessian(parts,model,domain["st"],domain["sx"])
    selected=choose_metrics(args.gap1_dir,model,args.top_per_model,args.top_global)
    base_dir=Path(out)/model/(mapping["condition"]+f"_pca{int(args.pca)}")
    base_dir.mkdir(parents=True,exist_ok=True)
    # Freeze the shortlist BEFORE any Gap2/Gap3 data-driven Hessian metrics.
    write_csv(base_dir/"gap1_frozen_metric_shortlist.csv",[dict(metric=m) for m in selected])
    meta=dict(model=model,beta=beta,zdim=zdim,pca=int(args.pca),repeat=args.repeat,
              gap1_shortlist_count=len(selected),gap1_dir=str(args.gap1_dir),
              checkpoint_sha256=mapping["checkpoint_hash"],
              latent_sha256=mapping["latent_hash"],dataset_sha256=mapping["dataset_hash"],
              mapping=mapping["mapping"],regression_condition=mapping["z_condition"],
              domain=domain,Y_test_agreement=yeval,
              estimand="Hessian of E[Y|W], W=(standardized TRAIN-mapped T_hat, standardized X)",
              gap2="oracle f(T_hat,X) numerical Y, fitted WY",
              gap3="frozen decoder d(Z,X) numerical Y, fitted on IDENTICAL WY points",
              no_neural_autograd=True,
              model_selection="VALIDATION numeric Y RMSE pooled standardized four channels, no Hessian truth",
              caveat="full-trial Z is encoded using all subject trials; not held-out-trial NP prediction")
    (base_dir/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    allrows=[];accuracy=[];landscapes=[]; hyper=[]
    for method in args.methods:
        print(f"[{model}] {method}: selecting common hyperparameters, then Gap2+3 concurrently",flush=True)
        fit,choice,grid=select_estimator(method,parts,y,args,args.seed+sum(map(ord,model)))
        for r in grid:
            hyper.append(r)
        candidate={ch:(fit["zz"][:,i],fit["zx"][:,i]) for ch,i in zip(CHANNELS,(0,1))}
        nn={ch:(fit["zz"][:,i],fit["zx"][:,i]) for ch,i in zip(CHANNELS,(2,3))}
        for ch in CHANNELS:
            for source,field in (("gap2_vs_analytic",candidate[ch]),
                                 ("gap3_vs_analytic",nn[ch])):
                for block,v in compare_field(test_oracle[ch],field).items():
                    accuracy.append(dict(model=model,method=method,channel=ch,
                                         comparison=source,block=block,**v))
            for block,v in compare_field(candidate[ch],nn[ch]).items():
                accuracy.append(dict(model=model,method=method,channel=ch,
                                     comparison="gap3_vs_gap2",block=block,**v))
        rows=make_curves(model,method,test_oracle,fit,angles_from_spec(args.angles),selected,args.planes,args.metric_mode)
        available=set(rows[0])-{"model","method","channel","source","plane","angle"}
        missing=[m for m in selected if m not in available]
        write_csv(base_dir/f"{method}__missing_oracle_metrics.csv",
                  [dict(metric=m,reason="Not a second-order ZZ/ZX numeric-WY metric") for m in missing],
                  fields=["metric","reason"])
        allrows.extend(rows)
        comparisons=compare_landscapes(rows,sorted(available),args.endpoint_tolerance)
        landscapes.extend(comparisons)
        plot_selected(rows,comparisons,base_dir/method,args.plots)
        np.savez_compressed(base_dir/f"{method}__test_hessians.npz",
             formula_zz=fit["zz"][:,:2],formula_zx=fit["zx"][:,:2],
             nn_zz=fit["zz"][:,2:],nn_zx=fit["zx"][:,2:],
             oracle_logit_zz=test_oracle["logit"][0],
             oracle_logit_zx=test_oracle["logit"][1],
             oracle_probability_zz=test_oracle["probability"][0],
             oracle_probability_zx=test_oracle["probability"][1])
    write_csv(base_dir/"hyperparameter_validation.csv",hyper)
    write_csv(base_dir/"field_errors.csv",accuracy)
    write_csv(base_dir/"threeway_curves.csv",allrows)
    write_csv(base_dir/"threeway_landscape_comparisons.csv",landscapes)
    return landscapes,meta,base_dir


def cross_summary(rows, expected_models=None):
    """Coverage-aware comparison: never rank C1-only wins as 4-model wins.

    'conditions' means plane x output channel x estimator. Number of
    pairwise planes depends on actual theory dimension: C3 has 15, not 6.
    Each stage comparison is ranked separately, and flat curves retain NaN
    correlations rather than being declared a match.
    """
    expected_models=list(expected_models or sorted({r["model"] for r in rows}))
    methods=sorted({r["method"] for r in rows})
    channels=sorted({r["channel"] for r in rows})
    expected_n=sum(DIMENSIONS[m][0]*(DIMENSIONS[m][0]-1)//2
                   for m in expected_models)*len(methods)*len(channels)
    buckets={}
    for r in rows:
        buckets.setdefault((r["metric"],r["comparison"]),[]).append(r)
    summary=[]
    for (metric,comp),v in buckets.items():
        present={r["model"] for r in v}
        finite=[float(r["curve_r"]) for r in v
                if np.isfinite(float(r["curve_r"]))]
        per_model=[]
        for model in expected_models:
            sub=[r for r in v if r["model"]==model]
            if not sub:
                continue
            rvals=[float(r["curve_r"]) for r in sub
                   if np.isfinite(float(r["curve_r"]))]
            per_model.append(dict(
                model=model,near=np.mean([int(r["b_near_true_axis"]) for r in sub]),
                both=np.mean([int(r["a_near_true_axis"]) and
                              int(r["b_near_true_axis"]) for r in sub]),
                median_r=float(np.median(rvals)) if rvals else float("nan")))
        coverage=float(len(v)/expected_n) if expected_n else 0.
        mm=[x["median_r"] for x in per_model if np.isfinite(x["median_r"])]
        summary.append(dict(
            metric=metric,comparison=comp,
            models_covered=len(present),
            model_names_covered=";".join(sorted(present)),
            model_coverage_fraction=len(present)/len(expected_models),
            conditions=len(v),expected_conditions=expected_n,
            condition_coverage_fraction=coverage,
            full_coverage=int(len(present)==len(expected_models) and
                              len(v)==expected_n),
            finite_r_conditions=len(finite),
            median_curve_r=float(np.median(finite)) if finite else float("nan"),
            near_axis_fraction=float(np.mean([int(r["b_near_true_axis"]) for r in v])),
            both_near_axis_fraction=float(np.mean([
                int(r["a_near_true_axis"]) and int(r["b_near_true_axis"]) for r in v])),
            min_model_near_axis_fraction=float(min(x["near"] for x in per_model)),
            min_model_both_near_fraction=float(min(x["both"] for x in per_model)),
            min_model_median_curve_r=float(min(mm)) if mm else float("nan"),
            median_shape_rmse=float(np.median([
                float(r["normalized_shape_rmse"]) for r in v]))))
    # FIRST reject partial coverage, THEN compare the weakest model.
    def score(r):
        mr=r["min_model_median_curve_r"]
        return (r["comparison"],-r["full_coverage"],
                -r["model_coverage_fraction"],-r["condition_coverage_fraction"],
                -r["min_model_both_near_fraction"],
                -r["min_model_near_axis_fraction"],
                -mr if np.isfinite(mr) else float("inf"),r["metric"])
    summary.sort(key=score)
    return summary

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models",nargs="+",default=["C1","C2","C3","CR3"])
    p.add_argument("--gap1-dir",type=Path,default=DEFAULT_OUT)
    p.add_argument("--output",type=Path,default=DEFAULT_OUT/"gaps23")
    p.add_argument("--beta-c",type=float,default=.0005)
    p.add_argument("--beta-cr3",type=float,default=.00005)
    p.add_argument("--zdim",type=int,default=None,
                   help="Must equal true parameter count: 4/4/6/4")
    p.add_argument("--pca",type=int,choices=[0,1],default=0)
    p.add_argument("--repeat",type=int,default=1)
    p.add_argument("--lambda-sparse",type=float,default=0.)
    p.add_argument("--context-min",type=float,default=.25)
    p.add_argument("--context-max",type=float,default=.8)
    p.add_argument("--train-subjects",type=int,default=100)
    p.add_argument("--validation-subjects",type=int,default=24)
    p.add_argument("--test-subjects",type=int,default=24)
    p.add_argument("--trials-per-subject",type=int,default=24)
    p.add_argument("--methods",nargs="+",choices=["sieve","rbf"],default=["sieve","rbf"])
    p.add_argument("--degree-grid",nargs="+",type=int,default=[2,3])
    p.add_argument("--ridge-grid",nargs="+",type=float,default=[1e-4,1e-6])
    p.add_argument("--bandwidth-grid",nargs="+",type=float,default=[1.5,2.5])
    p.add_argument("--rbf-ridge-grid",nargs="+",type=float,default=[1e-4,1e-6])
    p.add_argument("--landmarks",type=int,default=128)
    p.add_argument("--device",default="auto")
    p.add_argument("--decoder-batch",type=int,default=32)
    p.add_argument("--angles",default="0:90:10")
    p.add_argument("--planes",default="all",help="all (default), or 1-based pair identifiers 12,13")
    p.add_argument("--endpoint-tolerance",type=float,default=10.)
    p.add_argument("--top-per-model",type=int,default=30)
    p.add_argument("--top-global",type=int,default=50)
    p.add_argument("--metric-mode",choices=("shortlist","all"),default="shortlist")
    p.add_argument("--plots",type=int,default=3)
    p.add_argument("--seed",type=int,default=20261109)
    args=p.parse_args(argv)
    if min(args.train_subjects,args.validation_subjects,args.test_subjects)<3 or args.trials_per_subject<2:
        p.error("Need >=3 subjects per split and >=2 trials/subject")
    if min(args.degree_grid)<2 or min(args.ridge_grid)<=0 or min(args.rbf_ridge_grid)<=0:
        p.error("Degree >=2 and regularizers >0 required")
    if not 0<=args.endpoint_tolerance<=45:
        p.error("endpoint-tolerance must be 0..45")
    return args


def main(argv=None):
    args=parse_args(argv)
    root=ROOT
    modelnames=[]
    for a in args.models:
        for model in a.upper().split(","):
            if model=="CR1":
                raise ValueError("CR1 not implemented; use real CR3 generator explicitly for Gap2/3")
            if model not in DIMENSIONS:
                raise ValueError(f"Unknown {model}; supported {list(DIMENSIONS)}")
            if model not in modelnames:
                modelnames.append(model)
    allrows=[];manifest=[]
    for model in modelnames:
        rows,meta,out=process_model(model,args,root,args.output)
        allrows.extend(rows)
        manifest.append(dict(model=model,output=str(out),checkpoint=meta["checkpoint_sha256"],
                             metric_shortlist=meta["gap1_shortlist_count"]))
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/"all_models_landscape_comparisons.csv",allrows)
    write_csv(args.output/"cross_model_threeway_ranking.csv",cross_summary(allrows,modelnames))
    (args.output/"manifest.json").write_text(json.dumps(
        dict(models=modelnames,artifact_source="real S1/S3/S5 only",runs=manifest,
             gap1_directory=str(args.gap1_dir),sequential_stage="Gap1 then paired Gap2+Gap3",
             same_W_and_estimator=True,NN_autograd=False),indent=2),encoding="utf-8")
    print("Gap2 and Gap3 paired numeric WY completed:",args.output,flush=True)


if __name__=="__main__":
    main()
