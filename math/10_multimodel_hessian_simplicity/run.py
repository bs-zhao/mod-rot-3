"""10: independent cross-model Hessian simplicity landscape experiment.

Run at repository root:
  python math/10_multimodel_hessian_simplicity/run.py --models C1 C2 C3 CR3
No imports from another math experiment. Oracle-only initial stage: no NP
checkpoint or numerical WY inference is silently substituted for analytics.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from oracle import DIMENSIONS, derivatives, valid_x_mask
from metrics import compute_metrics, metric_descriptor, INVARIANT_DIAGNOSTICS
from functional import controlled_effects

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT/"outputs"/"math"/"10_multimodel_hessian_simplicity"


def models_from_args(items):
    raw=[v.strip().upper() for token in items for v in token.split(",") if v.strip()]
    resolved=[]
    alias=False
    for name in raw:
        if name == "CR1":
            # This repository has no CR1 generator; make alias conspicuous.
            name="CR3"
            alias=True
        if name not in DIMENSIONS:
            raise ValueError(f"Unknown model {name}. Repository generators: C1,C2,C3,CR3.")
        if name not in resolved:
            resolved.append(name)
    if alias:
        warnings.warn("CR1 IS NOT A GENERATOR IN mod-rot-3. Explicitly using CR3 for the CR1 CLI alias. "
                      "This is NOT a true CR1 experiment.",stacklevel=2)
        print("WARNING: CLI CR1 means repository CR3 here; no CR1 data or formula exists.")
    return resolved, alias


def angles_from_spec(s):
    if ":" in s:
        a,b,h=map(float,s.split(":"))
        if h <= 0 or b < a:
            raise ValueError("Angles require start:end:positive_step")
        a=np.arange(a,b+0.1*h,h)
        if not np.isclose(a[-1],b):
            a=np.r_[a,b]
    else:
        a=np.array([float(v) for v in s.split(",")],float)
    if not len(a) or not np.isfinite(a).all() or np.min(a)<0 or np.max(a)>90:
        raise ValueError("Angles must be in [0,90]")
    return sorted(set(np.round(a,8)))


def rotation(d, i, j, angle):
    R=np.eye(d)
    rad=np.deg2rad(angle)
    c,s=np.cos(rad),np.sin(rad)
    R[i,i]=R[j,j]=c
    R[i,j]=-s
    R[j,i]=s
    return R


def rotate_h(zz,zx,R):
    # Local chain rule. NOT R @ RMS(H) @ R.T.
    return (np.einsum("ia,nab,jb->nij",R,zz,R,optimize=True),
            np.einsum("ia,nab->nib",R,zx,optimize=True))


def _sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def load_samples(model, subjects, trials, seed, *, debug=False):
    rng=np.random.default_rng(seed)
    path=ROOT/"outputs"/"choice"/model/"s1_generate"/"s1_generate__dataset.npz"
    d,p=DIMENSIONS[model]
    if not path.is_file():
        if not debug:
            raise FileNotFoundError(
                f"{model}: missing original generated dataset {path}\n"
                f"Run 'python np/s1_generate.py --model {model}' first; this independent "
                "experiment will not regenerate or overwrite training data.")
        t=rng.normal(0,.5,size=(subjects,d))
        if model=="CR3":
            raise FileNotFoundError("CR3 debug fallback disabled: official fixed bank is required")
        x=rng.uniform(-1,1,size=(subjects*trials,p))
        tt=np.repeat(t,trials,axis=0)
        return dict(t=tt,x=x,st=np.full(d,.5),sx=np.full(p,1/np.sqrt(3)),
                    subject_ids=np.repeat(np.arange(subjects),trials),dataset_hash="SYNTHETIC_DEBUG",
                    n_subjects=subjects,valid_fraction=1.,dataset_source="SYNTHETIC_DEBUG")
    with np.load(path,allow_pickle=False) as dat:
        t_all=np.asarray(dat["parameters"],float)
        counts=np.asarray(dat["trial_counts"],int)
        x_all=np.asarray(dat["x"],float)
        ids=np.asarray(dat["subject_index"],int)
    if t_all.ndim!=2 or t_all.shape[1]!=d or x_all.shape[1]!=p or len(t_all)!=len(counts):
        raise ValueError(f"{model}: incompatible s1 dataset parameter or X shapes")
    if int(counts.sum())!=len(x_all) or not np.array_equal(ids,np.repeat(np.arange(len(counts)),counts)):
        raise ValueError("Saved trials not grouped by subject; refuse to misalign T and X")
    eligible=np.flatnonzero(counts>0)
    selected=np.sort(rng.choice(eligible,size=min(subjects,len(eligible)),replace=False))
    offsets=np.r_[0,np.cumsum(counts)]
    t_rows=[]; x_rows=[]; subject_ids=[]
    total_valid=0; total_trial=0
    common_cr3_indices=None
    if model=="CR3":
        # Keep identical X positions for every selected subject. This permits
        # genuine subject x fixed-X conditional decomposition later.
        common_valid=np.flatnonzero(valid_x_mask(x_all[:counts[0]],model))
        common_cr3_indices=np.sort(rng.choice(
            common_valid,size=min(trials,len(common_valid)),replace=False))
    for sid in selected:
        lo,hi=offsets[sid:sid+2]
        valid=np.flatnonzero(valid_x_mask(x_all[lo:hi],model))
        total_valid+=len(valid); total_trial+=hi-lo
        if len(valid)<1:
            raise ValueError(f"{model}: subject {sid} has no differentiable stimuli")
        chosen=(common_cr3_indices if model=="CR3" else
                np.sort(rng.choice(valid,size=min(trials,len(valid)),replace=False)))
        if model=="CR3" and not np.all(valid_x_mask(x_all[lo+chosen],model)):
            raise ValueError("CR3 fixed stimuli disagree across subjects")
        x_rows.append(x_all[lo+chosen])
        t_rows.append(np.repeat(t_all[sid:sid+1],len(chosen),axis=0))
        subject_ids.extend([int(sid)]*len(chosen))
    sx=(np.std(x_all[:counts[0]],axis=0) if model=="CR3" else
        np.full(p,1/np.sqrt(3.)))
    # Full CR3 T reference, not only selected 32 subjects. Full saved data are
    # used for scaling; no TEST prediction or learning stage is performed.
    st=(np.std(t_all,axis=0) if model=="CR3" else np.full(d,.5))
    if np.any(st<=0) or np.any(sx<=0):
        raise ValueError("Zero variance in coordinate units")
    if model=="CR3":
        bank=ROOT/"outputs"/"choice"/"CR3"/"s0_design"/"CR3_4P_fixed500.npz"
        if not bank.is_file():
            raise FileNotFoundError(f"Official CR3 bank missing: {bank}")
        with np.load(bank,allow_pickle=False) as b:
            xb=np.asarray(b["X"],float)
        if xb.shape!=(500,6) or not np.allclose(xb,x_all[:500],rtol=0,atol=1e-12):
            raise ValueError("CR3 saved dataset disagrees with official 500-trial bank")
    return dict(t=np.vstack(t_rows),x=np.vstack(x_rows),st=st,sx=sx,
                subject_ids=np.array(subject_ids,int),dataset_hash=_sha256(path),
                n_subjects=len(selected),valid_fraction=total_valid/total_trial,
                dataset_source=str(path.relative_to(ROOT)))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows and fields is None:
        return
    if fields is None:
        fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def global_ranking(out, models):
    """Compare the same metric across all models and both channels.

    Metrics absent in a model (e.g. tail_k beyond dimension) are explicitly
    labelled as partially covered, never counted as a universal success.
    """
    buckets={}
    for model in models:
        path=out/model/"ranking_by_model_channel.csv"
        with path.open(newline="",encoding="utf-8") as f:
            rows=list(csv.DictReader(f))
        for r in rows:
            buckets.setdefault(r["metric"],[]).append(r)
    report=[]
    for metric,rows in buckets.items():
        mod={r["model"] for r in rows}
        frac=[float(r["success_fraction"]) for r in rows]
        fl=[int(r["flat_planes"]) for r in rows]
        full=(len(mod)==len(models) and len(rows)==2*len(models))
        report.append(dict(metric=metric,models_covered=len(mod),
                           conditions_covered=len(rows),full_coverage=int(full),
                           mean_success_fraction=float(np.mean(frac)),
                           min_success_fraction=float(np.min(frac)),
                           max_success_fraction=float(np.max(frac)),
                           conditions_all_planes_success=int(sum(x==1.0 for x in frac)),
                           total_flat_planes=sum(fl)))
    report.sort(key=lambda r:(-r["full_coverage"],-r["min_success_fraction"],
                              -r["mean_success_fraction"],r["total_flat_planes"],r["metric"]))
    write_csv(out/"cross_model_ranking.csv",report)
    return report


def third_fields(base_t, x, model, st, sx, step):
    """Finite-difference analytic 2nd derivatives along standardized T axes.

    The 2nd derivatives themselves remain exact; third derivatives are FD
    approximations and must be labeled, not called closed-form oracle.
    """
    d=len(st); n=len(base_t)
    v={}
    for ch in ("logit","probability"):
        v[ch]={"uuu":np.zeros((n,d,d,d)),"uux":np.zeros((n,d,d,len(sx)))}
    for k in range(d):
        delta=np.zeros(d)
        delta[k]=step*st[k]
        plus=derivatives(base_t+delta[None,:],x,model)
        minus=derivatives(base_t-delta[None,:],x,model)
        for ch in v:
            v[ch]["uuu"][:,:,:,k]=(
                (plus["tt_"+ch]-minus["tt_"+ch])/(2*step)
                *st[None,:,None]*st[None,None,:])
            v[ch]["uux"][:,:,k,:]=(
                (plus["tx_"+ch]-minus["tx_"+ch])/(2*step)
                *st[None,:,None]*sx[None,None,:])
    return v


def higher_metrics(tensors,R):
    a=tensors["uuu"]; b=tensors["uux"]
    uuu=np.einsum("ia,jb,kc,nabc->nijk",R,R,R,a,optimize=True)
    uux=np.einsum("ia,jb,nabp->nijp",R,R,b,optimize=True)
    n,d,_,_=uuu.shape
    ii=np.arange(d)
    own=uuu[:,ii,ii,ii]
    other=np.stack([uuu[:,i,i,j] for i in range(d) for j in range(d) if j!=i],axis=1)
    diff=np.stack([uuu[:,i,j,k] for i in range(d) for j in range(d)
                   for k in range(d) if len({i,j,k})==3],axis=1) if d>=3 else own[:, :0]
    zxown=uux[:,ii,ii,:]
    zxother=np.stack([uux[:,i,j,:] for i in range(d) for j in range(d) if i!=j],axis=1)
    return {"zz_self_third":own,"zz_other_third":other,
            "zz_three_distinct_third":diff,"zx_own_u_third":zxown,
            "zx_other_u_third":zxother}


def fourth_metrics(t,x,model,R,st,sx,step):
    # Direct second FD of analytic H_uu along own rotated coordinate.
    # Each +/- shift in standardized U maps to +/- step*(st*R[i,:]) in T.
    d=len(st)
    own={"logit":[],"probability":[]}
    for i in range(d):
        move=step*st*R[i]
        pp=derivatives(t+move,x,model)
        mm=derivatives(t-move,x,model)
        center=derivatives(t,x,model)
        for ch in own:
            h4=((pp["tt_"+ch]-2*center["tt_"+ch]+mm["tt_"+ch])/(step*step)
                *st[None,:,None]*st[None,None,:])
            projected=np.einsum("a,nab,b->n",R[i],h4,R[i],optimize=True)
            own[ch].append(projected)
    return {ch:{"zz_self_fourth":np.stack(own[ch],axis=1)} for ch in own}


def sample_rotations(d,num,rng):
    """Embedded Haar-distributed SO(3)/SO(4) perturbations, plus full SO(d)."""
    for subset_d in sorted({min(3,d),min(4,d),d}):
        for idx in range(num):
            axes=np.sort(rng.choice(d,subset_d,replace=False))
            A=rng.normal(size=(subset_d,subset_d))
            Q,RR=np.linalg.qr(A)
            Q=Q@np.diag(np.sign(np.diag(RR)))
            if np.linalg.det(Q)<0:
                Q[:,0]*=-1
            R=np.eye(d)
            R[np.ix_(axes,axes)]=Q
            yield f"SO{subset_d}",idx,"-".join(str(int(i+1)) for i in axes),R


def summarize(curves, output, angles, endpoint_tolerance=10.0):
    """Endpoint minima (0 or 90) are necessary, not sufficient, for recovery."""
    buckets={}
    for row in curves:
        for key,v in row.items():
            if not key.endswith("_loss"):
                continue
            id=(row["model"],row["channel"],row["plane"],key)
            buckets.setdefault(id,[]).append((float(row["angle"]),float(v)))
    screens=[]; grouped={}
    for (model,channel,plane,metric),v in buckets.items():
        v=sorted(v)
        a=np.array([x[0] for x in v]);y=np.array([x[1] for x in v])
        ymin=float(y.min()); ymax=float(y.max())
        span=ymax-ymin
        flat=bool(span<=1e-9*max(1.,abs(ymax)))
        min_a=float(a[np.argmin(y)])
        endpoint=bool(np.isclose(min_a,0) or np.isclose(min_a,90))
        near_endpoint=bool(min(abs(min_a),abs(90-min_a))<=endpoint_tolerance+1e-9)
        # Never count invariant flat losses as correct axis identification.
        success=endpoint and not flat
        near_success=near_endpoint and not flat
        at0=y[np.isclose(a,0)]; at90=y[np.isclose(a,90)]
        screens.append(dict(model=model,channel=channel,plane=plane,metric=metric,
            minimum_angle=min_a, minimum_value=ymin,
            value_0=float(at0[0]) if len(at0) else float("nan"),
            value_90=float(at90[0]) if len(at90) else float("nan"),
            maximum_value=ymax,relative_range=float(span/max(abs(ymax),1e-12)),
            flat=int(flat),endpoint_minimum=int(endpoint),
            nonflat_endpoint_minimum=int(success),
            near_endpoint_minimum=int(near_endpoint),
            nonflat_near_endpoint_minimum=int(near_success),
            endpoint_tolerance_degrees=endpoint_tolerance))
        g=grouped.setdefault((model,channel,metric),[])
        g.append((success,near_success,flat,endpoint,span))
    rankings=[]
    for (model,channel,metric),values in grouped.items():
        rankings.append(dict(model=model,channel=channel,metric=metric,n_planes=len(values),
            success_planes=sum(int(a) for a,_,_,_,_ in values),
            near_success_planes=sum(int(b) for _,b,_,_,_ in values),
            flat_planes=sum(int(c) for _,_,c,_,_ in values),
            endpoint_minimum_planes=sum(int(d) for _,_,_,d,_ in values),
            success_fraction=float(np.mean([a for a,_,_,_,_ in values])),
            near_success_fraction=float(np.mean([b for _,b,_,_,_ in values])),
            median_curve_range=float(np.median([v for *_,v in values]))))
    rankings.sort(key=lambda x:(x["model"],x["channel"],-x["success_fraction"],
                                x["flat_planes"],-x["median_curve_range"],x["metric"]))
    write_csv(output/"screens.csv",screens)
    write_csv(output/"ranking_by_model_channel.csv",rankings)
    return rankings


def plot_top(curves,rankings,out,max_plots=4):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    groups={}
    for r in rankings:
        groups.setdefault((r["model"],r["channel"]),[]).append(r)
    p=out/"plots";p.mkdir(exist_ok=True)
    for (model,ch),group in groups.items():
        for mr in group[:max_plots]:
            metric=mr["metric"]
            subset=[r for r in curves if r["model"]==model and r["channel"]==ch]
            fig,ax=plt.subplots(figsize=(8,4.5))
            for plane in sorted({r["plane"] for r in subset}):
                v=sorted([r for r in subset if r["plane"]==plane],key=lambda t:float(t["angle"]))
                ax.plot([t["angle"] for t in v],[t[metric] for t in v],label=plane)
            ax.set(title=f"{model} {ch} | {metric}",xlabel="Pairwise rotation angle (degrees)",
                   ylabel="Loss (lower = candidate simpler)")
            ax.legend(fontsize=6,ncol=3)
            fig.tight_layout()
            safe=metric[:110]
            fig.savefig(p/f"{model}_{ch}_{safe}.png",dpi=140)
            plt.close(fig)


def run_one(model,args,out,seed):
    print(f"  {model}: reading saved s1 simulation",flush=True)
    ds=load_samples(model,args.subjects,args.trials_per_subject,seed,debug=args.synthetic_debug)
    t,x,st,sx=(ds[k] for k in ("t","x","st","sx"))
    src=derivatives(t,x,model)
    raw={}
    for ch in ("logit","probability"):
        raw[ch]=(
            src["tt_"+ch]*st[None,:,None]*st[None,None,:],
            src["tx_"+ch]*st[None,:,None]*sx[None,None,:])
    hd=third_fields(t,x,model,st,sx,args.fd_step) if args.higher else None
    d=len(st)
    planes=list(itertools.combinations(range(d),2))
    if args.planes!="all":
        selected=set(args.planes.split(","))
        planes=[(i,j) for i,j in planes if f"{i+1}{j+1}" in selected]
        if not planes:
            raise ValueError(f"No matching 1-based plane identifiers: {args.planes}")
    modelout=out/model
    modelout.mkdir(parents=True,exist_ok=True)
    meta=dict(model=model,source="exact_analytic_oracle",samples=len(t),
              subjects=ds["n_subjects"],trials_per_subject=args.trials_per_subject,
              input_dim=len(sx),parameter_dim=d,
              data_source=ds["dataset_source"],dataset_sha256=ds["dataset_hash"],
              valid_x_fraction=ds["valid_fraction"],
              cr3_zero_amount_policy="conditional_on_differentiable_X" if model=="CR3" else "all_X_valid",
              scale_reference_T=st.tolist(),scale_reference_X=sx.tolist(),
              output_channels=["logit","probability"],
              binary_choice_note="Analytic Hessian of Bernoulli choice is undefined; E[choice|T,X]=p",
              third_derivatives="finite differences of exact analytic second derivatives" if args.higher else "disabled",
              fourth_derivatives="finite differences of exact analytic Hessian" if args.fourth else "disabled",
              function_value_contrasts="local 3-point/4-corner exact function evaluations; NOT Sobol" if args.functional else "disabled",
              function_value_anchors=args.function_anchors, function_value_step=args.function_step,
              angle_grid=angles_from_spec(args.angles),
              planes=[f"{i+1}{j+1}" for i,j in planes],zero_angle="original TRUE T parameter axes",
              supervision_note="TRUE T oracle only: no learned Z, no NN, no WY Hessian estimator")
    (modelout/"metadata.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    rows=[]
    metrics_seen=set()
    angles=angles_from_spec(args.angles)
    for i,j in planes:
        label=f"{i+1}{j+1}"
        for angle in angles:
            R=rotation(d,i,j,angle)
            fourth=fourth_metrics(t,x,model,R,st,sx,args.fd_step) if args.fourth else None
            functional, n_function_anchors = (
                controlled_effects(t,x,model,R,st,sx,args.function_step,args.function_anchors,
                                   args.relative_threshold) if args.functional else ({},0))
            for ch in ("logit","probability"):
                zz,zx=rotate_h(*raw[ch],R)
                high=higher_metrics(hd[ch],R) if hd is not None else {}
                if fourth is not None:
                    high.update(fourth[ch])
                score=compute_metrics(zz,zx,threshold=args.relative_threshold,higher=high)
                score.update(functional.get(ch,{}))
                metrics_seen.update(score)
                rows.append(dict(model=model,channel=ch,plane=label,angle=float(angle),**score))
        print(f"    plane {label}: {len(angles)} angles x 2 channels",flush=True)
    write_csv(modelout/"curves.csv",rows)
    write_csv(modelout/"metric_catalog.csv",
              [metric_descriptor(x) for x in sorted(metrics_seen)])
    ranks=summarize(rows,modelout,angles,args.endpoint_tolerance)
    plot_top(rows,ranks,modelout,args.plots)
    # Multi-axis stress test: SAME samples and target, oracle T axes at identity.
    if args.multiway>0:
        rng=np.random.default_rng(seed+999)
        rr=[]
        for ch in ("logit","probability"):
            zz,zx=raw[ch]
            high=higher_metrics(hd[ch],np.eye(d)) if hd is not None else {}
            identity=compute_metrics(zz,zx,args.relative_threshold,high)
            rr.append(dict(model=model,channel=ch,group="identity",index=-1,axes="all",
                           **{k:v for k,v in identity.items() if k.endswith("_loss")}))
        for group,idx,axes,R in sample_rotations(d,args.multiway,rng):
            for ch in ("logit","probability"):
                zz,zx=rotate_h(*raw[ch],R)
                high=higher_metrics(hd[ch],R) if hd is not None else {}
                scores=compute_metrics(zz,zx,args.relative_threshold,high)
                rr.append(dict(model=model,channel=ch,group=group,index=idx,axes=axes,
                               **{k:v for k,v in scores.items() if k.endswith("_loss")}))
        write_csv(modelout/"multiway_random_rotations.csv",rr)
        comparisons=[]
        for ch in ("logit","probability"):
            ref=next(r for r in rr if r["channel"]==ch and r["group"]=="identity")
            for metric in (k for k in ref if k.endswith("_loss")):
                for group in sorted({r["group"] for r in rr if r["group"]!="identity"}):
                    vals=[float(r[metric]) for r in rr if r["channel"]==ch and r["group"]==group]
                    comparisons.append(dict(model=model,channel=ch,metric=metric,rotation_group=group,
                        identity=float(ref[metric]),random_min=float(min(vals)),
                        n_random=len(vals),random_below_identity=int(sum(v<ref[metric]-1e-9 for v in vals))))
        write_csv(modelout/"multiway_comparison.csv",comparisons)
    # Exact orthogonal invariants, checked across all tested angles.
    for ch in ("logit","probability"):
        for metric in INVARIANT_DIAGNOSTICS:
            vals=[float(r[metric]) for r in rows if r["channel"]==ch]
            if max(vals)-min(vals)>1e-7*max(1.,np.mean(vals)):
                raise AssertionError(f"Orthogonal Hessian invariant drift: {model} {ch} {metric}")
    return len(metrics_seen),len(rows),meta


def main(argv=None):
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--models",nargs="+",default=["C1","C2","C3","CR3"],
                   help="C1 C2 C3 CR3 (CR1 is a clearly warned CLI alias of CR3, NOT a true model)")
    a.add_argument("--subjects",type=int,default=36)
    a.add_argument("--trials-per-subject",type=int,default=12)
    a.add_argument("--angles",default="0:90:10")
    a.add_argument("--planes",default="all",help="all, or 1-based pairs e.g. 12,13")
    a.add_argument("--relative-threshold",type=float,default=.10)
    a.add_argument("--endpoint-tolerance",type=float,default=10.,
                   help="Degree tolerance for a separate NEAR-endpoint screen; exact gate retained")
    a.add_argument("--fd-step",type=float,default=.01)
    a.add_argument("--higher",action=argparse.BooleanOptionalAction,default=True)
    a.add_argument("--functional",action=argparse.BooleanOptionalAction,default=True,
                   help="Exact oracle 3-point self and 4-corner interactions, NOT Sobol")
    a.add_argument("--function-anchors",type=int,default=24)
    a.add_argument("--function-step",type=float,default=.04)
    a.add_argument("--fourth",action="store_true",help="Slow fourth derivative finite difference extension")
    a.add_argument("--multiway",type=int,default=6,help="Random SO(3), SO(4), SO(d) rotations per group")
    a.add_argument("--plots",type=int,default=3,help="Top metric plots per model/channel; 0 disables")
    a.add_argument("--output",default=str(DEFAULT_OUT))
    a.add_argument("--seed",type=int,default=20261109)
    a.add_argument("--synthetic-debug",action="store_true",
                   help="Debug C1-C3 only; never count this as a real scientific run")
    args=a.parse_args(argv)
    if (args.subjects<3 or args.trials_per_subject<2 or args.fd_step<=0 or args.relative_threshold<=0
        or args.multiway<0 or args.function_anchors<2 or args.function_step<=0
        or not 0<=args.endpoint_tolerance<=45):
        a.error("subjects>=3, trials-per-subject>=2, fd-step/threshold>0, multiway>=0")
    models,alias=models_from_args(args.models)
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for mi,model in enumerate(models):
        n,row_count,meta=run_one(model,args,out,args.seed+mi*100)
        summary.append(dict(model=model,metrics=n,rows=row_count,subject_count=meta["subjects"],
                            valid_x_fraction=meta["valid_x_fraction"],
                            data_source=meta["data_source"]))
    write_csv(out/"all_models_summary.csv",summary)
    global_ranking(out,models)
    (out/"manifest.json").write_text(json.dumps(dict(models=models,cr1_alias_used=alias,
        modes=["oracle_exact_TT_TX"],args=vars(args),
        caution="This is analytic-oracle screening, NOT NN/WY recovery or blind rotation. "
        "Choice Hessian is the probability Hessian of its conditional mean.",
        runs=summary),indent=2,ensure_ascii=False),encoding="utf-8")
    print("Complete. Outputs:",out)
    for row in summary:
        print(" ",row["model"],row["metrics"],"metrics,",row["rows"],"angle/channel rows")


if __name__=="__main__":
    main()
