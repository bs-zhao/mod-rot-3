"""Step 4: BLIND local-WY structural discovery; initially C1 and CR3 only.

Compare anonymous Step-2 trial matrices using multiple *local* response
fingerprints B(U). No simulator formula, true theta, neural decoder, R_best,
or per-case rotation matrices are read for score calculation.

Neighborhoods, local linear operators and response profiles are all fitted
once in an arbitrary anonymous reference W; orthogonal registration from W
alone transforms the local response-gradient field to each other W. Case
angles are not read until after all loss values have been written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
STEP3=HERE.parent/"step3_blind_matrix"
for folder in (str(HERE),str(STEP3)):
    if folder not in sys.path:
        sys.path.insert(0,folder)

from profiles import CHANNELS, load_profiles, profile_views
from metrics import normalize_response
from local import (orthogonal_map,select_anchors,build_local_operators,
                   global_fingerprint,scores)

PLOTTED=(
    "local_point_entropy",
    "local_rms_entropy",
    "nonlinear_local_point_entropy",
    "nonlinear_local_rms_entropy",
    "local_gram_offdiag_fraction",
    "global_local_point_entropy",
)
DEFAULT_K=(96,256)
CONTROL_NAMES={
    "control_local_gradient_fro",
    "control_local_gradient_gram_spectrum",
    "control_local_variation_fraction",
    "global_control_local_gradient_fro",
    "global_control_local_gradient_gram_spectrum",
}


def tagnumber(x):
    return f"{float(x):g}".replace("-","m").replace(".","p").replace("+","")


def condition(beta,lamb,cmin,cmax,dim):
    return (f"beta{tagnumber(beta)}_lambda{tagnumber(lamb)}_"
            f"ctx{tagnumber(cmin)}to{tagnumber(cmax)}_L{dim}")


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(2*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def save_csv(path,rows,cols):
    with Path(path).open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=cols,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)


def save_json(path,payload):
    Path(path).write_text(json.dumps(payload,indent=2,ensure_ascii=False,
                                     allow_nan=False)+"\n",encoding="utf-8")


def load_cases(step2,model):
    stage=Path(step2)
    indexfile=stage/"index.json"
    if not indexfile.is_file() or not (stage/"Y.npz").is_file():
        raise FileNotFoundError(f"Step-2 shared Y or index missing: {stage}")
    index=json.loads(indexfile.read_text(encoding="utf-8"))
    if index.get("model") != model or not index.get("complete"):
        raise ValueError("Step 2 must be complete for the requested model")
    expected=37 if model in ("C1","CR3") else -1
    paths=sorted((stage/"cases").glob("*/W.npy"))
    if len(paths)!=expected or {p.parent.name for p in paths}!= {
            c["id"] for c in index["cases"]}:
        raise ValueError("Step-2 W cases missing or inconsistent with index")
    return index,paths


def get_u(wpath,sid,offsets,reference,*,d,m,check_rows):
    w=np.load(wpath,mmap_mode="r",allow_pickle=False)
    if w.ndim!=2 or w.shape != (m,reference.shape[1]):
        raise ValueError(f"W shape differs from shared source: {wpath}")
    u=np.asarray(w[offsets[:-1],:d],dtype=np.float64)
    verify=np.unique(np.r_[check_rows,offsets[:-1],offsets[1:]-1])
    if not np.allclose(w[verify,:d],u[sid[verify]],atol=1e-6,rtol=1e-6):
        raise ValueError(f"W subject U does not align with original trial rows: {wpath}")
    if not np.array_equal(w[verify,d:],reference[verify,d:]):
        raise ValueError(f"X columns changed across rotations: {wpath}")
    if not np.isfinite(u).all():
        raise ValueError("Non-finite candidate latent U")
    return u


def plot_curves(path,rows,model,channel,view,k,metric):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.7,5.2))
    for plane in sorted({r["plane"] for r in rows}):
        seq=sorted((r for r in rows if r["plane"]==plane),
                   key=lambda x:x["angle_deg"])
        ax.plot([r["angle_deg"] for r in seq],
                [r["loss"] for r in seq],lw=1.5,marker="o",
                markersize=3.5,label=plane)
    ax.set_xticks([0,15,30,45,60,75,90])
    ax.set(xlabel="Additional rotation relative to supervised R* (degrees)",
           ylabel="WY-only simplicity loss (lower = simpler)",
           title=f"{model} · {channel} · {view} · k={k}\n{metric}")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8,ncol=3)
    fig.tight_layout()
    buf=io.BytesIO()
    try:
        fig.savefig(buf,format="png",dpi=150)
        path.write_bytes(buf.getvalue())
    finally: plt.close(fig)


def plot_fingerprint(path,mat,model,detail):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mat=np.asarray(mat)
    d,p=mat.shape
    ind=np.linspace(0,p-1,min(p,120),dtype=int)
    arr=mat[:,ind]
    fig,ax=plt.subplots(figsize=(10,3.6))
    ax.imshow(arr,aspect="auto",cmap="viridis")
    ax.set_yticks(range(d),[f"U{i+1}" for i in range(d)])
    ax.set_xlabel(f"Response-profile columns ({len(ind)} of {p})")
    ax.set_title(f"{model}: {detail}")
    fig.tight_layout()
    output=io.BytesIO()
    try:
        fig.savefig(output,format="png",dpi=150)
        path.write_bytes(output.getvalue())
    finally: plt.close(fig)


def summarize(scored,case_index,folder):
    """Unblind only after scores have been materialized to CSV."""
    plane_cases=defaultdict(dict)
    for c in case_index["cases"]:
        if c["id"]=="baseline":continue
        i,j=c["pair_1based"]
        plane_cases[f"U{i}-U{j}"][c["angle_deg"]]=c["id"]

    collection=defaultdict(dict)
    for record in scored:
        key=(record["channel"],record["view"],record["k"],record["metric"])
        collection[key][record["case"]]=record["loss"]

    plotrows=[]
    ranking=[]
    summaries=[]
    for (ch,view,k,metric),by_case in sorted(collection.items()):
        if "baseline" not in by_case:
            raise ValueError("Missing baseline in revealed scores")
        baseline=float(by_case["baseline"])
        names=sorted(by_case,key=lambda key:(by_case[key],key))
        scale=max(1.,abs(baseline))
        tol=1e-6*scale
        rank=1+sum(value<baseline-tol for value in by_case.values())
        for number,name in enumerate(names,1):
            ranking.append(dict(channel=ch,view=view,k=k,metric=metric,
                                rank=number,case=name,loss=by_case[name],
                                is_baseline=int(name=="baseline"),
                                is_control=int(metric in CONTROL_NAMES)))
        passed=0;positive=0;endgap=[];midgaps=[];infos=[]
        for plane,lookup in sorted(plane_cases.items()):
            if not all(angle in lookup for angle in (15,30,45,60,75,90)):
                raise ValueError(f"Incomplete angle grid: {plane}")
            ninety=float(by_case[lookup[90]])
            interior=np.array([by_case[lookup[a]] for a in (15,30,45,60,75)])
            envelope=np.r_[baseline,ninety,interior]
            width=max(float(np.ptp(envelope)),1e-12)
            contrast=float(np.median(interior)-.5*(baseline+ninety))
            endpoint_gap=abs(baseline-ninety)
            # Sign/permutation invariance, meaningful non-flat curvature,
            # and BOTH 0/90 endpoints below all intermediate angles.
            good=(width>1e-5*scale and
                  endpoint_gap<=.05*width and
                  float(np.min(interior))-max(baseline,ninety)>.02*width)
            passed+=int(good)
            positive+=int(contrast>.05*width and width>1e-5*scale)
            midgaps.append(contrast);endgap.append(endpoint_gap)
            for angle in (0,15,30,45,60,75,90):
                name="baseline" if angle==0 else lookup[angle]
                plotrows.append(dict(channel=ch,view=view,k=k,metric=metric,
                                     plane=plane,angle_deg=angle,case=name,
                                     loss=by_case[name],
                                     delta_from_baseline=by_case[name]-baseline,
                                     endpoint_favored=int(good)))
        summaries.append(dict(
            channel=ch,view=view,k=k,metric=metric,
            metric_family="global" if metric.startswith("global_")
            else ("nonlinear_gradient" if metric.startswith("nonlinear_")
                  else "local_gradient"),
            is_control=int(metric in CONTROL_NAMES),
            baseline_rank=rank,
            baseline_loss=baseline,
            winner_case=names[0],
            n_cases=len(names),n_planes=len(plane_cases),
            endpoint_favored_planes=passed,
            positive_interior_contrast_planes=positive,
            median_interior_minus_endpoint=float(np.median(midgaps)),
            max_endpoint_gap=float(max(endgap)),
        ))
    save_csv(folder/"plane_curves.csv",plotrows,[
        "channel","view","k","metric","plane","angle_deg",
        "case","loss","delta_from_baseline","endpoint_favored"])
    save_csv(folder/"rankings.csv",ranking,[
        "channel","view","k","metric","rank","case","loss",
        "is_baseline","is_control"])
    save_csv(folder/"metric_summary.csv",summaries,[
        "channel","view","k","metric","metric_family","is_control",
        "baseline_rank","baseline_loss","winner_case","n_cases",
        "n_planes","endpoint_favored_planes","positive_interior_contrast_planes",
        "median_interior_minus_endpoint","max_endpoint_gap"])
    return plotrows,summaries


def analyze(*,step2,output,model,channels=CHANNELS,
            k_grid=DEFAULT_K,anchors=128,seed=31,train_fraction=.7,
            local_ridge=.025,global_ridge=.025,profile_ridge=.001,
            overwrite=False,plots=True):
    """Compute blind losses using ONLY W/Y; result identity reveal is post hoc."""
    step2=Path(step2);output=Path(output)
    if model not in ("C1","CR3"):
        raise ValueError("Initial experiment limited to C1 and CR3")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}; --overwrite to replace")
    if (not .5<train_fraction<.9 or anchors<1 or local_ridge<=0 or
            global_ridge<=0 or profile_ridge<=0 or not k_grid or
            any(k<8 for k in k_grid)):
        raise ValueError("Invalid training/anchor/ridge/neighborhood configuration")
    case_index,paths=load_cases(step2,model)
    n=int(case_index["n_subjects"]);m=int(case_index["n_trials"])
    dz=int(case_index["Z_dim"])
    if dz!=4 or n<20:
        raise ValueError("Expected 4-dimensional C1/CR3 data with >=20 subjects")
    rng=np.random.default_rng(seed)
    perm=rng.permutation(len(paths))
    blind=[(f"B{i+1:03d}",paths[k]) for i,k in enumerate(perm)]
    profiles,profile_info,(counts,sid,tid,offsets)=load_profiles(
        step2,blind[0][1].parent,channels=channels,ridge=profile_ridge)
    if profile_info["n_subjects"]!=n or profile_info["n_trials"]!=m:
        raise ValueError("Step 2 and Step 3 profile dimensions disagree")
    sids=rng.permutation(n)
    split=int(round(n*train_fraction))
    train=np.sort(sids[:split]);test=np.sort(sids[split:])
    if len(train)<=max(k_grid):
        raise ValueError(f"Not enough TRAIN subjects for neighborhoods: {k_grid}")
    anchor_ids=select_anchors(n,train,test,seed+37,anchors)

    response={}
    response_info={}
    for ch,F in profiles.items():
        for view,data in profile_views(F,profile_info["profile_kind"]).items():
            norm,meta=normalize_response(data,train)
            if norm is None:
                raise ValueError(f"All response profile features constant: {ch}/{view}")
            response[(ch,view)]=norm
            response_info[f"{ch}/{view}"]=meta
    random_rows=rng.choice(m,size=min(1024,m),replace=False)
    ref_w=np.load(blind[0][1],mmap_mode="r",allow_pickle=False)
    u0=get_u(blind[0][1],sid,offsets,ref_w,d=dz,m=m,
             check_rows=random_rows)
    # All neighborhoods are based solely on the subject U geometry.
    operators={k:build_local_operators(
        u0,train,anchor_ids,neighbors=k,ridge_fraction=local_ridge)
        for k in k_grid}
    # Cache local slopes and global slopes once per Y channel/profile view.
    # The arbitrary reference case is blinded: nothing about baseline/R* is read.
    cached={}
    fit_quality=[]
    for key,F in response.items():
        ch,view=key
        Bglob,training_mse=global_fingerprint(
            u0,F,train,ridge_fraction=global_ridge)
        cached[("global",ch,view)]=Bglob
        for k,op in operators.items():
            B,mse=op.apply(F)
            cached[(k,ch,view)]=B
            fit_quality.append(dict(
                channel=ch,view=view,k=k,
                profile_columns=int(F.shape[1]),
                n_local_anchors=op.n_anchors,
                heldout_anchor_profile_mse=mse,
                global_train_profile_mse=training_mse,
                median_neighbor_radius=float(np.median(op.bandwidths)),
                median_local_condition_number=float(np.median(op.condition_numbers)),
                max_local_condition_number=float(np.max(op.condition_numbers))))

    output.parent.mkdir(parents=True,exist_ok=True)
    # Stage on the same filesystem as the final destination so atomic rename
    # works for custom --output paths and Windows drives too.
    with tempfile.TemporaryDirectory(
            prefix=".step4_blind_build_",dir=output.parent) as temp:
        pending=Path(temp)/"step4_local_structure"
        pending.mkdir()
        (pending/"plots").mkdir()
        # Blind scoring phase. Only anonymous W and cached B(U) are allowed.
        scores_blind=[]
        geometry_checks=[]
        for number,(anon,path) in enumerate(blind,1):
            ut=get_u(path,sid,offsets,ref_w,d=dz,m=m,
                     check_rows=random_rows)
            h,alignment_error=orthogonal_map(u0,ut)
            geometry_checks.append(dict(
                blind_id=anon,orthogonal_registration_rmse=alignment_error))
            for (ch,view),F in response.items():
                bglob=h@cached[("global",ch,view)]
                global_scores=scores(bglob[None,:,:],include_variation=False)
                # The global score is the exact same structural metric
                # at a single (linear) slope matrix rather than a local field.
                global_scores={
                    "global_"+name:score for name,score in global_scores.items()
                }
                for metric,value in global_scores.items():
                    scores_blind.append(dict(
                        blind_id=anon,channel=ch,view=view,k="global",
                        metric=metric,loss=float(value),
                        is_control=int(metric in CONTROL_NAMES)))
                for k in k_grid:
                    localb=np.einsum("ij,ajp->aip",h,cached[(k,ch,view)],
                                     optimize=True)
                    vals=scores(localb)
                    for metric,value in vals.items():
                        scores_blind.append(dict(
                            blind_id=anon,channel=ch,view=view,k=str(k),
                            metric=metric,loss=float(value),
                            is_control=int(metric in CONTROL_NAMES)))
            if number==1 or number%5==0 or number==len(blind):
                print(f"[Step4 blind] {model}: {number}/{len(blind)} "
                      "anonymous W cases scored",flush=True)

        save_csv(pending/"blind_scores.csv",scores_blind,[
            "blind_id","channel","view","k","metric","loss","is_control"])
        save_csv(pending/"geometry_checks.csv",geometry_checks,[
            "blind_id","orthogonal_registration_rmse"])
        save_csv(pending/"local_fit_quality.csv",fit_quality,[
            "channel","view","k","profile_columns","n_local_anchors",
            "heldout_anchor_profile_mse","global_train_profile_mse",
            "median_neighbor_radius","median_local_condition_number",
            "max_local_condition_number"])
        # Unlock case identity only AFTER blind_scores.csv is written.
        reveal={anon:path.parent.name for anon,path in blind}
        save_csv(pending/"blind_key.csv",[
            {"blind_id":b,"case":name} for b,name in reveal.items()],
            ["blind_id","case"])
        identified=[{**row,"case":reveal[row["blind_id"]]}
                    for row in scores_blind]
        save_csv(pending/"revealed_scores.csv",identified,[
            "blind_id","case","channel","view","k","metric","loss",
            "is_control"])
        curves,summary=summarize(identified,case_index,pending)

        plot_errors={}
        if plots:
            for ch in ("logit_nn","p_nn"):
                if ch not in channels:continue
                for view in ("contrast",):
                    for k in list(map(str,k_grid))+["global"]:
                        for metric in PLOTTED:
                            if metric.startswith("global_")!=(k=="global"):
                                continue
                            record=[row for row in curves if row["channel"]==ch
                                    and row["view"]==view and row["k"]==k
                                    and row["metric"]==metric]
                            if not record:continue
                            path=pending/"plots"/f"curve_{ch}_{view}_k{k}_{metric}.png"
                            try:
                                plot_curves(path,record,model,ch,view,k,metric)
                            except OSError as e:
                                plot_errors[path.name]=str(e)
            # Detailed gradient heatmaps in selected angles are DIAGNOSTIC
            # output only; they cannot enter already-frozen blind scores.
            diag=pending/"diagnostic_gradient_maps"
            diag.mkdir()
            if "logit_nn" in channels:
                cases=["baseline"]+[c["id"] for c in case_index["cases"]
                        if c["angle_deg"]==45]
                k=max(k_grid)
                for case in cases:
                    wc=step2/"cases"/case/"W.npy"
                    ut=get_u(wc,sid,offsets,ref_w,d=dz,m=m,
                             check_rows=random_rows)
                    h,_=orthogonal_map(u0,ut)
                    B=np.einsum("ij,ajp->aip",h,
                                cached[(k,"logit_nn","contrast")],
                                optimize=True)
                    rms=np.sqrt(np.mean(B**2,axis=0))
                    variability=np.sqrt(np.mean(
                        (B-B.mean(axis=0,keepdims=True))**2,axis=0))
                    np.savez_compressed(diag/f"{case}_k{k}.npz",
                                        gradient_rms=rms,
                                        nonlinear_gradient_rms=variability,
                                        local_gradient_mean=B.mean(axis=0))
                    for label,mat in (
                        ("local_gradient_rms",rms),
                        ("nonlinear_gradient_rms",variability)):
                        img=diag/f"{case}_{label}.png"
                        try:
                            plot_fingerprint(img,mat,model,
                                             f"{case} · {label} · k={k}")
                        except OSError as e:
                            plot_errors[img.name]=str(e)

        save_json(pending/"summary.json",{
            "model":model,"n_subjects":n,"n_total_trials":m,
            "n_candidates":len(blind),"latent_dim":dz,
            "source_step2":str(step2),
            "source_step2_Y_sha256":sha256(step2/"Y.npz"),
            "source_step2_index_sha256":sha256(step2/"index.json"),
            "channels":list(channels),
            "subject_profile":profile_info,
            "normalized_profile_columns":response_info,
            "neighborhood_sizes":list(k_grid),
            "n_anchor_subjects":len(anchor_ids),
            "seed":seed,"train_fraction":train_fraction,
            "ridge_local_covariance_fraction":local_ridge,
            "ridge_global_covariance_fraction":global_ridge,
            "ridge_profile":profile_ridge,
            "reference_case_identity_hidden_until_reveal":True,
            "allowed_model_inputs":[
                "Step2 case W.npy","Step2 Y.npz",
                "Step2 original subject/trial indexing"],
            "disallowed_inputs_used":[],
            "orthogonal_coordinate_registration":
                "H=polar(center(U_reference).T@center(U_case)); "
                "B_case=H@B_ref; registration uses U only and not angles/R/T/Y",
            "local_regression":
                "Euclidean kNN TRAIN subjects; Gaussian weights; "
                "isotropic local weighted ridge slopes; "
                "all anchors selected from TEST subjects; Y values from "
                "TRAIN neighbors only; local W~Y response gradients "
                "are not exact model derivatives",
            "metric_semantics":
                "All scalar outputs smaller-is-sparser except controls. "
                "Nonlinear-gradient scores are computed on local "
                "B(U) - mean_anchor[B(U)]; they capture changes of "
                "estimated slopes over subjects, not the exact Hessian.",
            "comparison":
                "Global, pointwise local, anchor-RMS, anchor-meanabs and "
                "nonlinear-variation gradient concentration. Stronger "
                "CR3 alignment is an empirical question, not assumed.",
            "leakage_caveat":
                "S5 subject Z used all original outcomes; sample-wise "
                "anchor holdout is a structural fit diagnostic, not "
                "independent behavioral response prediction.",
            "plot_errors":plot_errors,
        })
        train_set=set(train.tolist())
        anchor_set=set(anchor_ids.tolist())
        save_csv(pending/"subject_split.csv",[
            {"subject_id":int(i),"partition":"train" if i in train_set else "test",
             "is_local_anchor":int(i in anchor_set)}
            for i in range(n)],
            ["subject_id","partition","is_local_anchor"])
        # Only successful complete outputs are published.
        if output.exists():
            backup=output.with_name(output.name+"_previous")
            if backup.exists():
                raise FileExistsError(f"Stale previous output: {backup}")
            os.replace(output,backup)
        else:
            backup=None
        try:
            output.parent.mkdir(parents=True,exist_ok=True)
            os.replace(pending,output)
        except BaseException:
            if backup is not None and not output.exists():
                os.replace(backup,output)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    print(f"[Step4] {model}: {len(blind)} cases; "
          f"{len(summary)} metric summaries -> {output}",flush=True)
    return {"output":str(output),"n_cases":len(blind),
            "n_metric_summaries":len(summary)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices=("C1","CR3"),required=True)
    p.add_argument("--beta",type=float)
    p.add_argument("--lambda-sparse",type=float,default=0.)
    p.add_argument("--context-min",type=float,default=.25)
    p.add_argument("--context-max",type=float,default=.8)
    p.add_argument("--repeat",type=int,default=1)
    p.add_argument("--objective",choices=("abs_mse","signed_mse"),
                   default="abs_mse")
    p.add_argument("--step2-dir",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--channels",default=",".join(CHANNELS))
    p.add_argument("--neighborhoods",default="96,256")
    p.add_argument("--anchors",type=int,default=128)
    p.add_argument("--seed",type=int,default=31)
    p.add_argument("--train-fraction",type=float,default=.7)
    p.add_argument("--local-ridge",type=float,default=.025)
    p.add_argument("--global-ridge",type=float,default=.025)
    p.add_argument("--profile-ridge",type=float,default=.001)
    p.add_argument("--no-plots",action="store_true")
    p.add_argument("--overwrite",action="store_true")
    args=p.parse_args(argv)
    if args.beta is None:
        args.beta=.00005 if args.model=="CR3" else .0005
    try:
        grid=tuple(int(k.strip()) for k in args.neighborhoods.split(","))
    except ValueError:
        p.error("Invalid --neighborhoods: comma-separated integers required")
    if not grid or len(set(grid))!=len(grid) or any(k<8 for k in grid):
        p.error("Invalid neighborhood sizes")
    channels=tuple(c.strip() for c in args.channels.split(",") if c.strip())
    if not channels or len(set(channels))!=len(channels) or not set(channels)<=set(CHANNELS):
        p.error(f"--channels must be a subset of {CHANNELS}")
    if args.repeat<1 or args.beta<0 or args.lambda_sparse<0:
        p.error("Invalid model configuration")
    standard=(ROOT/"outputs"/"supervised_abs_rotation"/args.model/
              condition(args.beta,args.lambda_sparse,args.context_min,
                        args.context_max,4)/
              f"repeat{args.repeat:02d}"/args.objective)
    step2=args.step2_dir or standard/"step2_pairwise_sweep"
    output=args.output or standard/"step4_local_structure"
    analyze(step2=step2,output=output,model=args.model,channels=channels,
            k_grid=grid,anchors=args.anchors,seed=args.seed,
            train_fraction=args.train_fraction,local_ridge=args.local_ridge,
            global_ridge=args.global_ridge,profile_ridge=args.profile_ridge,
            overwrite=args.overwrite,plots=not args.no_plots)


if __name__=="__main__":
    main()
