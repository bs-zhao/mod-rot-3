"""07: rich LOCAL ZZ/ZX Hessian complexity under pairwise A* rotations.

All 9 sources are evaluated at identical U(subject) × X(stimulus) grids:
  exact analytic CR3 Hessian of [E(choice)=p, logit, probability],
  six-Y original generator regression [choice, logit, probability],
  six-Y frozen neural decoder regression [choice, logit, probability].

No NN weights or NN derivatives are accessed. Data-only derivative fields
are exact derivatives of the newly refitted tensor polynomial surrogate.

Full example, from repository root:
  python math/07_cr3_local_hessian_complexity/run.py --planes all --angles 0:90:5
Cheap smoke:
  python math/07_cr3_local_hessian_complexity/run.py --planes 01 --angles 0,45,90 --eval-subjects 12 --eval-stimuli 8 --anchors 3
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SYS06=ROOT/"math"/"06_cr3_hessian_rotation_landscapes"
if str(SYS06) not in sys.path:sys.path.insert(0,str(SYS06))

from fast_tensor import RotationTensor
from loader import load_source
from rotation import candidate_mapping,parse_angles,parse_planes,plane_name
from screen import select_sparsity_metrics
from local_polynomial import polynomial_field_shake
from analytic_reference import reference_fields,rotate_fields
from controlled_sweeps import (analytic_sweeps,polynomial_sweeps,
                               sweep_metrics)
from local_metrics import complexity_metrics
from comparisons import compare_all_sources

import analytic_reference as oracle_module

SOURCES=("analytic","formula_y","nn_y")
TARGETS=("choice","logit","probability")
FIELDS=("value","zz","zx","zzz","zzx","zxx","zzzz","zzxx","zxxx")


def to_json(o):
    if isinstance(o,np.ndarray):return to_json(o.tolist())
    if isinstance(o,np.integer):return int(o)
    if isinstance(o,np.bool_):return bool(o)
    if isinstance(o,np.floating):return to_json(float(o))
    if isinstance(o,float):return o if np.isfinite(o) else None
    if isinstance(o,tuple):return [to_json(x) for x in o]
    if isinstance(o,list):return [to_json(x) for x in o]
    if isinstance(o,dict):return {str(k):to_json(v) for k,v in o.items()}
    return o


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with open(path,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:to_json(row.get(k)) for k in keys})


def choose_evaluation(X,Z,heldout,*,subjects,stimuli,anchors,seed):
    """Fixed valid derivative domain and fixed held-out subject grid."""
    rng=np.random.default_rng(seed)
    X=np.asarray(X,dtype=float)
    money=(X[:,[0,1,3,4]]>.5).all(axis=1)
    probability=((X[:,[2,5]]>.02)&(X[:,[2,5]]<.98)).all(axis=1)
    choices=np.flatnonzero(money&probability)
    if len(choices)<3:
        raise ValueError("Not enough positive-money / interior-probability X stimuli")
    use_stim=np.sort(rng.choice(choices,size=min(len(choices),stimuli),replace=False))
    use_subject=np.sort(rng.choice(heldout,size=min(len(heldout),subjects),replace=False))
    use_anchor=np.sort(rng.choice(use_subject,size=min(len(use_subject),anchors),
                                  replace=False))
    if not len(use_subject) or not len(use_anchor):
        raise ValueError("Empty held-out local evaluation grid")
    return use_subject,use_stim,use_anchor


def field_slice(fields,idx):
    return {key:np.asarray(value[idx],dtype=float) for key,value in fields.items()}


def domain_convergence(reference1,reference2,keys=("zzz","zzx","zxx","zzzz","zzxx","zxxx")):
    output=[]
    for key in keys:
        if key not in reference1:continue
        a=np.asarray(reference1[key])
        b=np.asarray(reference2[key])
        if a.shape!=b.shape:raise ValueError("FD convergence shape mismatch")
        diff=np.linalg.norm(a-b)
        rmse=np.sqrt(np.mean((a-b)**2))
        normal=np.linalg.norm(b)
        output.append(dict(derivative_field=key,
                           relative_FD_change=float(diff/max(normal,1e-15)),
                           absolute_FD_rmse=float(rmse),
                           reference_relative_scale=float(normal/max(np.linalg.norm(reference1["zz"]),1e-15))))
    return output


def evaluate_rotation(A,R,Z,X_bank,Y,estimator,analytic_reference,
                      eval_subj,eval_stim,anchors,b,include_fourth,sweep_levels,scales_x):
    X_eval=np.asarray(X_bank)[eval_stim]
    poly,fit_local,_,_,fit_rmse=polynomial_field_shake(
        estimator,A,Z,b,eval_subj,X_eval,fourth=include_fourth)
    analytic_local=rotate_fields(analytic_reference,R)
    U=np.asarray(Z,dtype=float)@A.T+b
    us=np.maximum(U.std(axis=0),1e-12)
    xs=np.maximum(np.asarray(scales_x,dtype=float),1e-12)
    per={}
    for source in SOURCES:
        for target_idx,target in enumerate(TARGETS):
            c=target_idx if source!="nn_y" else target_idx+3
            channels=analytic_local if source=="analytic" else fit_local
            source_fields=field_slice(channels,c)
            per[(source,target)]=complexity_metrics(
                source_fields,us,xs,include_fourth=include_fourth)
    # Value-based response complexity, at fixed Z_{-i}, X;
    # do not infer self-Z dependence by averaging Hessian over the
    # evaluation subject bank.
    fitted_sweeps=polynomial_sweeps(poly,U,anchors,levels=sweep_levels)
    true_sweeps=analytic_sweeps(U,b,R,anchors,X_eval,oracle_module,
                               levels=sweep_levels)
    for source in SOURCES:
        for ti,target in enumerate(TARGETS):
            ci=ti if source!="nn_y" else ti+3
            sweep=fitted_sweeps if source!="analytic" else true_sweeps
            per[(source,target)].update(sweep_metrics({
                "value":sweep["value"][ci],
                "curvature":sweep["curvature"][ci],
                "position":sweep["position"]}))
    comparisons=compare_all_sources(analytic_local,fit_local,None,
                                    include_fourth=include_fourth)
    return per,comparisons,fit_rmse,analytic_local,fit_local


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition",default="beta5em05_lambda0_ctx0p25to0p8_L4")
    p.add_argument("--repeat",type=int,default=1)
    p.add_argument("--source",type=Path,help="original completed step-04 repeat01 dir or ZIP")
    p.add_argument("--output",type=Path)
    p.add_argument("--planes",default="all")
    p.add_argument("--angles",default="0:90:5")
    p.add_argument("--degree-u",type=int,default=4)
    p.add_argument("--degree-x",type=int,default=4)
    p.add_argument("--ridge-u",type=float,default=1e-6)
    p.add_argument("--ridge-x",type=float,default=1e-6)
    p.add_argument("--basis",choices=("monomial","chebyshev"),default="monomial")
    p.add_argument("--seed",type=int,default=20261107)
    p.add_argument("--eval-subjects",type=int,default=48)
    p.add_argument("--eval-stimuli",type=int,default=24)
    p.add_argument("--anchors",type=int,default=6)
    p.add_argument("--sweep-levels",type=int,default=9)
    p.add_argument("--no-fourth",action="store_true",
                   help="Only local Hessian and 3rd derivative tensors")
    p.add_argument("--no-fd-convergence",action="store_true",
                   help="Skip analytical 3rd/4th finite-difference half-step validation")
    p.add_argument("--no-screen",action="store_true")
    p.add_argument("--no-plots",action="store_true")
    p.add_argument("--no-heldout-rmse",action="store_true")
    p.add_argument("--corr",type=float,default=.9)
    p.add_argument("--endpoint-tolerance",type=float,default=10.)
    p.add_argument("--min-range",type=float,default=.02)
    args=p.parse_args(argv)
    if min(args.eval_subjects,args.eval_stimuli,args.anchors)<1:
        p.error("Nonpositive local grid dimensions")
    if args.sweep_levels<6:p.error("At least 6 sweep levels required")
    if min(args.degree_u,args.degree_x)<1 or min(args.ridge_u,args.ridge_x)<=0:
        p.error("Invalid surrogate degree/ridge")
    angles=parse_angles(args.angles)
    planes=parse_planes(args.planes)
    if args.source is None:
        args.source=(ROOT/"outputs"/"math"/"cr3_zx_hessian_threeway"/
                     args.condition/f"repeat{args.repeat:02d}")
    if args.output is None:
        args.output=(ROOT/"outputs"/"math"/"07_cr3_local_hessian_complexity"/
                     args.condition/f"repeat{args.repeat:02d}"/
                     f"{args.basis}_u{args.degree_u}_x{args.degree_x}_"
                     f"lambu{args.ridge_u:g}_lambx{args.ridge_x:g}")
    data=load_source(args.source)
    args.output.mkdir(parents=True,exist_ok=True)
    Z,X,Y=data["Z"],data["X"],data["Y"]
    eval_subj,eval_stim,anchors=choose_evaluation(
        X,Z,data["heldout"],subjects=args.eval_subjects,stimuli=args.eval_stimuli,
        anchors=args.anchors,seed=args.seed)
    X_eval=X[eval_stim]
    # Must choose valid X before analytic derivative computation.
    star_T=Z[eval_subj]@data["A_star"].T+data["b"]
    print(f"Local grid {len(eval_subj)} held-out Z × {len(eval_stim)} X; "
          f"independent anchor subjects {len(anchors)}",flush=True)
    reference,fd_cfg=reference_fields(star_T,X_eval,fourth=not args.no_fourth)
    fd_records=[]
    if not args.no_fd_convergence:
        check,_=reference_fields(
            star_T,X_eval,fourth=not args.no_fourth,
            tsteps=np.asarray(fd_cfg["tstep"])/2,
            xsteps=np.asarray(fd_cfg["xstep"])/2)
        fd_records=domain_convergence(reference,check)
        write_csv(args.output/"analytic_fd_convergence.csv",fd_records)
    # The ORIGINAL fixed X bank, not just the selected interior subset,
    # determines standardized derivative units in all angles.
    xscale=np.maximum(X.std(axis=0),1e-12)
    engine=RotationTensor(
        X,Y,data["train"],degree_u=args.degree_u,degree_x=args.degree_x,
        ridge_u=args.ridge_u,ridge_x=args.ridge_x,basis=args.basis,
        heldout_ids=data["heldout"],validation_limit=(
            0 if args.no_heldout_rmse else 64))
    all_metrics=[]
    all_gaps=[]
    all_rmse=[]
    curves_by_plane={}
    for i,j in planes:
        plane=plane_name(i,j)
        for step,angle in enumerate(angles):
            A,R,U=candidate_mapping(
                data["A_star"],data["b"],Z,i,j,angle)
            values,gaps,rmse,analytic_loc,fit_loc=evaluate_rotation(
                A,R,Z,X,Y,engine,reference,eval_subj,eval_stim,
                anchors,data["b"],not args.no_fourth,args.sweep_levels,xscale)
            for (source,target),metrics in values.items():
                all_metrics.append(dict(
                    plane=plane,parameter_i=i,parameter_j=j,
                    angle_degrees=float(angle),source=source,target=target,
                    block="LOCAL",**metrics))
            for gap in gaps:
                all_gaps.append(dict(plane=plane,angle_degrees=float(angle),**gap))
            if rmse is not None:
                for k,channel in enumerate(("oracle_choice","oracle_logit",
                                            "oracle_probability","nn_choice",
                                            "nn_logit","nn_probability")):
                    all_rmse.append(dict(plane=plane,angle_degrees=float(angle),
                                         channel=channel,y_validation_rmse=float(rmse[k])))
            if step==0 or step==len(angles)-1 or (step+1)%5==0:
                print(f"  {plane}: {step+1}/{len(angles)} angles, {angle:g}°",flush=True)
        curves_by_plane[plane]=len(angles)
        write_csv(args.output/f"curves__{plane}.csv",
                  [r for r in all_metrics if r["plane"]==plane])
    write_csv(args.output/"all_local_metrics.csv",all_metrics)
    write_csv(args.output/"three_way_local_derivative_errors.csv",all_gaps)
    write_csv(args.output/"heldout_Y_rmse.csv",all_rmse)
    selected=[]
    groups=[]
    ranking=[]
    if not args.no_screen:
        selected,groups,ranking=select_sparsity_metrics(
            all_metrics,min_correlation=args.corr,
            endpoint_tolerance=args.endpoint_tolerance,
            min_relative_range=args.min_range)
        write_csv(args.output/"strict_screen_all.csv",selected)
        write_csv(args.output/"strict_screen_groups.csv",groups)
        write_csv(args.output/"strict_screen_ranking.csv",ranking)
    if not args.no_plots:
        from plots import plot_xz_sparsity
        # The pre-existing 06 plotting primitive shows three sources
        # separately per Y type and can draw any scalar metric on LOCAL.
        # This 07 experiment uses its own LOCAL plotting routine below.
        from plot_local import draw_local_charts
        draw_local_charts(all_metrics,args.output)
    summary=dict(
        experiment="07: local ZZ/ZX higher-order and adaptive sparsity rotation landscapes",
        A_star=data["A_star"],b=data["b"],source=str(args.source),
        source_sha256=data["sha256"],regression_R2=data["ols_r2"],
        planes=[plane_name(*pair) for pair in planes],angles=angles,
        sources=SOURCES,targets=TARGETS,
        total_local_derivative_methods=len(planes)*len(angles)*9,
        grid=dict(eval_subject_indices=eval_subj,
                  eval_stimulus_indices=eval_stim,anchor_indices=anchors,
                  X_validation="positive amounts, interior probabilities",
                  X_column_std=xscale),
        estimator=dict(degree_u=args.degree_u,degree_x=args.degree_x,
                       ridge_u=args.ridge_u,ridge_x=args.ridge_x,basis=args.basis),
        fd_reference=fd_cfg,fd_convergence=fd_records,
        derivative_interpretation=(
            "Exact reference Hessian is CR3 evaluated at T_hat=A*Z+b. "
            "Analytic order>=3 is finite-difference of exact CR3 Hessian, "
            "fitted formulaY/NNY fields are exact derivatives of fitted "
            "polynomials, NEVER gradients or derivatives of the NN."),
        failure_attribution=(
            "Local formulaY-vs-analytic mismatch can arise from polynomial "
            "sieve approximation, sampling and regularization, so NN-Y "
            "mismatch does NOT automatically isolate NN model error."),
        strict_screen=dict(corr=args.corr,endpoint_tolerance=args.endpoint_tolerance,
                           min_range=args.min_range,qualifying=sum(
                               bool(r["qualifies"]) for r in selected),
                           tested=len(selected)),
        metrics_count_per_method=len(values[("analytic","logit")]),
        computation_counts=dict(n_planes=len(planes),n_angles=len(angles),
                                n_derivative_analyses=len(all_metrics)))
    (args.output/"summary.json").write_text(
        json.dumps(to_json(summary),indent=2,allow_nan=False),encoding="utf-8")
    print(f"Saved {len(all_metrics)} local curves, {len(all_gaps)} local error rows to {args.output}",
          flush=True)
    return summary


if __name__=="__main__":main()
