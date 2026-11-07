"""Experiment 06: pairwise rotations of A_star, nine Hessians per angle.

Run from repository root:
 python math/06_cr3_hessian_rotation_landscapes/run.py
 python math/06_cr3_hessian_rotation_landscapes/run.py --planes 01 --angles 0:90:10

Nine outputs are 3 sources (exact analytic CR3, fitted analytic Y,
fitted frozen NN Y) × 3 responses (Choice, Logit, Probability).

All paths expose a common entry point run_nine(A,Z,X,Y,...).
The choice 'analytic Hessian' is that of E[Choice|T,X] = probability.
Y AND THE UNDERLYING GENERATING FUNCTION ARE FIXED across angles.
Only U = A(theta) Z + b changes; the six data-only regressions are refit.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from fast_tensor import RotationTensor
from loader import load_source, TARGETS, CHANNELS, ORACLE_TARGET
from metrics import metrics_for_source,equivariance_error,structure
from rotation import (candidate_mapping, original_z_to_candidate_u,
                      parse_angles,parse_planes,plane_name,full_block_rotation)
from plots import plot_dashboard,DEFAULT_METRICS


HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CONDITION="beta5em05_lambda0_ctx0p25to0p8_L4"
METHODS=("analytic","formula_y","nn_y")
PLOT_BLOCKS=("ZZ","ZX","XX")
ALL_BLOCKS=("ZZ","ZX","XX","FULL","ZZ_diag","ZZ_offdiag","XX_diag","XX_offdiag")


def as_native(o):
    if isinstance(o,np.ndarray):return as_native(o.tolist())
    if isinstance(o,np.integer):return int(o)
    if isinstance(o,np.bool_):return bool(o)
    if isinstance(o,np.floating):return as_native(float(o))
    if isinstance(o,float):return o if np.isfinite(o) else None
    if isinstance(o,(list,tuple)):return [as_native(v) for v in o]
    if isinstance(o,dict):return {str(k):as_native(v) for k,v in o.items()}
    return o


def csv_write(path,rows):
    if not rows:return
    cols=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore")
        w.writeheader()
        for row in rows:w.writerow({k:as_native(row.get(k)) for k in cols})


def run_nine(A,Z,X,Y,oracle_z,estimator,b):
    """Shared (A,Z,X,Y) interface for the requested 3×3 Hessian methods.

    Returns dict[(source,target)] -> mean 10x10 Hessian in candidate
    U=AZ+b, plus masked-domain validity and six-response Y-validation RMSE.

    The 'analytic' source transforms the exact step04 analytic CR3
    Hessian; it does not differentiate sampled 0/1 choice observations.
    """
    A=np.asarray(A,dtype=float)
    Z=np.asarray(Z,dtype=float)
    X=np.asarray(X,dtype=float)
    Y=np.asarray(Y,dtype=float)
    if A.shape!=(4,4) or Z.shape[1]!=4 or X.shape[1]!=6 or Y.shape!=(len(Z),len(X),6):
        raise ValueError("Expected A=(4,4), Z=(N,4), X=(M,6), Y=(N,M,6)")
    estimates,valid,fitted_rmse=estimator.fit_at(A,Z,b)
    nine={}
    reference={}
    for target in TARGETS:
        key=ORACLE_TARGET[target]
        reference[target]=original_z_to_candidate_u(oracle_z[key]["mean_ZX"],A)
        nine[("analytic",target)]=reference[target]
    for k,target in enumerate(TARGETS):
        nine[("formula_y",target)]=estimates[k]
        nine[("nn_y",target)]=estimates[3+k]
    assert len(nine)==9
    return nine,reference,valid,fitted_rmse


def plot_index(output,plots):
    doc=["# Hessian rotation landscape dashboards","",
         "Rows within each plot are Choice, Logit, Probability; columns are ZZ, ZX, XX.",
         "Each panel overlays exact analytic CR3, fitted analytic Y, fitted NN Y.",
         ""]
    for p in plots:
        rel=p.relative_to(output).as_posix()
        doc.append(f"- [{rel}]({rel})")
    (output/"PLOT_INDEX.md").write_text("\n".join(doc)+"\n",encoding="utf-8")


def calculate_best_angles(rows):
    """Best-angle diagnostics, not proof that the true axes minimize a loss."""
    metrics=[
        ("normalized_l1","min"),
        ("max_abs_over_frob","max"),
        ("offdiag_energy_ratio","min"),
        ("row_concentration","max"),
        ("row_entropy","min"),
        ("cosine_to_analytic","max"),
        ("relative_error_to_analytic","min"),
        ("equivariance_error","min"),
        ("sign_to_analytic","max"),
        ("abs_rank_to_analytic","max")
    ]
    groups={}
    for row in rows:
        key=(row["plane"],row["source"],row["target"],row["block"])
        groups.setdefault(key,[]).append(row)
    output=[]
    for (plane,src,target,block),series in groups.items():
        ordered=sorted(series,key=lambda r:r["angle_degrees"])
        for metric,direction in metrics:
            points=[(float(r["angle_degrees"]),float(r[metric]))
                    for r in ordered
                    if r.get(metric) is not None and np.isfinite(float(r[metric]))]
            if not points:continue
            winner=(min(points,key=lambda t:t[1]) if direction=="min"
                    else max(points,key=lambda t:t[1]))
            at0=next((v for a,v in points if abs(a)<1e-10),None)
            output.append(dict(plane=plane,source=src,target=target,block=block,
                               metric=metric,selection=direction,
                               optimum_angle_degrees=winner[0],optimum_value=winner[1],
                               value_at_zero=at0,
                               zero_is_optimum=bool(at0 is not None and
                                    np.isclose(at0,winner[1],atol=1e-8,rtol=1e-6))))
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition",default=CONDITION)
    p.add_argument("--repeat",type=int,default=1)
    p.add_argument("--source",type=Path,help="step04 repeat01 directory OR the original ZIP")
    p.add_argument("--output",type=Path)
    p.add_argument("--angles",default="0:90:5",
                   help="Inclusive degrees, e.g. 0:90:5 or 0,10,20,90")
    p.add_argument("--planes",default="all",
                   help="all; 01; 01,02; alpha:lambda; etc")
    p.add_argument("--degree-u",type=int,default=4)
    p.add_argument("--degree-x",type=int,default=4)
    p.add_argument("--ridge-u",type=float,default=1e-6)
    p.add_argument("--ridge-x",type=float,default=1e-6)
    p.add_argument("--basis",choices=("monomial","chebyshev"),default="monomial")
    p.add_argument("--validation-limit",type=int,default=128,
                   help="Max held-out subjects for Y RMSE at every rotation")
    p.add_argument("--no-plots",action="store_true")
    p.add_argument("--plot-metrics",nargs="+",default=list(DEFAULT_METRICS))
    p.add_argument("--no-save-matrices",action="store_true")
    args=p.parse_args(argv)
    if args.degree_u<1 or args.degree_x<1 or args.validation_limit<0:
        p.error("Invalid polynomial degree or validation limit")
    if args.ridge_u<=0 or args.ridge_x<=0:p.error("Ridge must be positive")
    angles=parse_angles(args.angles)
    planes=parse_planes(args.planes)
    if args.source is None:
        args.source=(ROOT/"outputs"/"math"/"cr3_zx_hessian_threeway"/
                     args.condition/f"repeat{args.repeat:02d}")
    if args.output is None:
        args.output=(ROOT/"outputs"/"math"/"06_cr3_hessian_rotation_landscapes"/
                     args.condition/f"repeat{args.repeat:02d}"/
                     f"{args.basis}_u{args.degree_u}_x{args.degree_x}_"
                     f"ridgeu{args.ridge_u:g}_ridgex{args.ridge_x:g}")
    output=args.output
    data=load_source(args.source)
    output.mkdir(parents=True,exist_ok=True)
    print("Step04 original Z,X,Y loaded; no NN backprop or regeneration",flush=True)
    print(f"Source={args.source}; Subjects={len(data['Z'])}; X={len(data['X'])}; "
          f"R²(T~A*Z+b)={data['ols_r2']}",flush=True)
    print("Six 2D planes and angles:",len(planes),len(angles),
          "=>",len(planes)*len(angles)*9,"Hessians",flush=True)
    engine=RotationTensor(
        data["X"],data["Y"],data["train"],
        degree_u=args.degree_u,degree_x=args.degree_x,
        ridge_u=args.ridge_u,ridge_x=args.ridge_x,basis=args.basis,
        heldout_ids=data["heldout"],validation_limit=args.validation_limit)
    all_rows=[]
    all_yerrors=[]
    plotted=[]
    trace={}
    for i,j in planes:
        plane=plane_name(i,j)
        plane_rows=[]
        tensors=np.empty((len(angles),len(METHODS)*len(TARGETS),10,10),dtype=np.float64)
        candidate_A=np.empty((len(angles),4,4),dtype=np.float64)
        reference_at_zero={}
        H_at_zero={}
        for index,angle in enumerate(angles):
            A,R,U=candidate_mapping(data["A_star"],data["b"],data["Z"],i,j,angle)
            candidate_A[index]=A
            nine,reference,fraction,yerror=run_nine(
                A,data["Z"],data["X"],data["Y"],data["oracle"],engine,data["b"])
            if index==0:
                H_at_zero={k:v.copy() for k,v in nine.items()}
                reference_at_zero={key:ref.copy() for key,ref in reference.items()}
            # Fundamental CR3 analytic-coordinate invariants:
            # the orthogonal R rotates first 4 dimensions only; the
            # Frobenius norm of full H, ZZ, ZX, XX must be unchanged.
            for target in TARGETS:
                for block in ("ZZ","ZX","XX","FULL"):
                    original=structure(reference_at_zero[target],block)["frob_norm"]
                    current=structure(reference[target],block)["frob_norm"]
                    if not np.isclose(original,current,atol=1e-8,rtol=1e-8):
                        raise AssertionError(f"Analytic reference rotation invariant failed: {plane} {angle} {target} {block}")
            for s,source in enumerate(METHODS):
                for t,target in enumerate(TARGETS):
                    H=nine[(source,target)]
                    tensors[index,s*3+t]=H
                    ref=reference[target]
                    ref_fraction=data["oracle"][ORACLE_TARGET[target]]["fraction_defined"]
                    blocks=metrics_for_source(H,ref,ref_fraction,ALL_BLOCKS)
                    drift=equivariance_error(H,H_at_zero[(source,target)],R)
                    for block,values in blocks.items():
                        baseline_norm=structure(H_at_zero[(source,target)],block)["frob_norm"]
                        norm=values["frob_norm"]
                        row=dict(plane=plane,parameter_i=i,parameter_j=j,
                                 angle_degrees=float(angle),source=source,
                                 target=target,block=block,**values,
                                 equivariance_error=drift,
                                 frobenius_norm_drift=float(norm/max(baseline_norm,1e-15)-1.))
                        plane_rows.append(row)
            if yerror is not None:
                for k,channel in enumerate(CHANNELS):
                    all_yerrors.append(dict(plane=plane,angle_degrees=float(angle),
                                            channel=channel,validation_y_rmse=float(yerror[k])))
            if index==0 or (index+1)%max(1,len(angles)//5)==0 or index==len(angles)-1:
                print(f" {plane}: {index+1}/{len(angles)} angles, current={angle:g}°",flush=True)
        if not args.no_save_matrices:
            np.savez_compressed(output/f"hessians__{plane}.npz",
                angles=angles,A_candidates=candidate_A,
                H=tensors,method_names=np.array(METHODS),
                target_names=np.array(TARGETS),
                reference_fraction_defined_logit=data["oracle"]["logit"]["fraction_defined"],
                reference_fraction_defined_probability=data["oracle"]["probability"]["fraction_defined"])
        csv_write(output/f"curves__{plane}.csv",plane_rows)
        all_rows.extend(plane_rows)
        trace[plane]={"n_angles":len(angles),"angles":angles,
                      "n_hessians":len(angles)*9,
                      "matrix_file":f"hessians__{plane}.npz" if not args.no_save_matrices else None}
        if not args.no_plots:
            for metric in args.plot_metrics:
                if metric not in plane_rows[0]:
                    raise ValueError(f"Unknown plot metric: {metric}")
                file=output/f"dashboard__{plane}__{metric}.png"
                plot_dashboard(plane_rows,plane,angles,metric,file)
                plotted.append(file)
        print("Completed pairwise plane:",plane,flush=True)
    csv_write(output/"all_curves.csv",all_rows)
    csv_write(output/"heldout_y_rmse.csv",all_yerrors)
    best=calculate_best_angles(all_rows)
    csv_write(output/"best_angles.csv",best)
    if plotted:plot_index(output,plotted)
    cfg=dict(
        experiment="06: A-star anchored pairwise rotations and 3x3 Hessian landscapes",
        A_star=data["A_star"],b=data["b"],A_star_condition=float(np.linalg.cond(data["A_star"])),
        coordinate_formula="U(theta)=R_ij(theta)A_star Z+b; original T_hat=A_star Z+b",
        angle_degrees=angles,planes=[plane_name(*pair) for pair in planes],
        hessian_sources=METHODS,y_types=TARGETS,
        note_choice_analytic="H[ E[Choice|Z,X] ] = H[Probability], not a derivative of sampled 0/1 decisions",
        source=data["original_source"],source_sha256=data["sha256"],
        regression_R2=data["ols_r2"],total_hessians=len(planes)*len(angles)*9,
        estimator=dict(type="factorized tensor polynomial",
                       basis=args.basis,degree_u=args.degree_u,degree_x=args.degree_x,
                       ridge_u=args.ridge_u,ridge_x=args.ridge_x,
                       heldout_y_rmse_subject_count=min(len(data["heldout"]),args.validation_limit)),
        methodology=[
            "Nine methods share (A,Z,X,Y) through run_nine; no network gradients.",
            "All six Y arrays and the original CR3 function are fixed across rotations.",
            "Exact Hessians are coordinate pullbacks of the original step04 analytical Z,X mean Hessians.",
            "Each rotated U is regressed afresh; one X SVD and Y×Ux projection are reused for speed.",
            "Analytic choice means Bernoulli conditional expectation p, same Hessian as analytic probability.",
            "The original step04 affine A is refitted from Z and T_true using the recorded map-fit subjects and cross-checked.",
            "Hessian structural simplicity may NOT be minimized at 0 degrees in true CR3 coordinates.",
            "Some amount derivatives at X=0 are undefined; the original conditional-mean validity mask is retained.",
            "Cosine against the CR3 reference does NOT measure the unknown NN's own Hessian.",
            "The rotation-invariant full/ZZ/ZX/XX Frobenius norms of analytic Hessians are asserted at every angle.",
            "A*, not its left-rotation, maps the fixed encoder Z to original CR3 parameters.",
        ],
        files=dict(curves="all_curves.csv",best_angles="best_angles.csv",
                   heldout="heldout_y_rmse.csv",
                   dashboards="dashboard__{plane}__{metric}.png"),
        trace=trace,
    )
    (output/"summary.json").write_text(json.dumps(as_native(cfg),indent=2,allow_nan=False),
                                       encoding="utf-8")
    print(f"Saved {len(all_rows)} Hessian metrics, {len(best)} angle summaries to {output}",flush=True)
    return cfg


if __name__=="__main__":
    main()
