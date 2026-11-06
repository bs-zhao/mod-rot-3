"""Per-plane rank curves and opt-in FIT-only stress tests.

Failure-mode tests for function identifiability are separate in test_suite.py.
All stress refits use FIT-subject ZXY only; no theta is ever read in this module.
"""
from __future__ import annotations
import csv
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from .surrogate import make_proxy
from .optimizer import optimize


def best_match_rows(reference,other):
    from scipy.optimize import linear_sum_assignment
    cos=np.abs(np.asarray(reference)@np.asarray(other).T)
    i,j=linear_sum_assignment(-cos)
    matched=cos[i,j]
    return dict(mean_abs_axis_cos=float(np.mean(matched)),
                minimum_abs_axis_cos=float(np.min(matched)),
                axis_assignment=list(map(int,j)),axis_cos=matched.tolist())


def plane_curves(obj,best,steps=73,max_pairs=6):
    steps=int(steps)
    if steps<3:raise ValueError("plane_steps must be >=3")
    pairs=list(combinations(range(obj.k),2))[:int(max_pairs)]
    rows=[]
    for i,j in pairs:
        for degrees in np.linspace(-90.,90.,steps):
            rad=np.deg2rad(degrees)
            B=np.eye(obj.k)
            B[i,i]=B[j,j]=np.cos(rad)
            B[i,j]=-np.sin(rad)
            B[j,i]=np.sin(rad)
            for frame,mat in (("identity",B),("best",B@best)):
                pieces=obj.components(mat)
                rows.append(dict(pair=f"u{i+1}-u{j+1}",frame=frame,
                                 angle_degrees=float(degrees),
                                 loss=pieces["loss"],
                                 rank_excess=pieces["rank_excess"],
                                 weak_axis_penalty=pieces["weak_axis_penalty"]))
    return rows


def stress_runs(proxy,inputs,reference,config):
    """Refit the response-profile proxy and R for multiple FIT sample sizes/seeds.

    This directly tests a potential F'/eta' derivative-instability source.
    All q/X/Y/mask inputs here originate from preselected NP TRAIN respondents.
    """
    q,x,y,mask=inputs
    runs=[]
    seeds=[int(v) for v in config.get("stress_seeds",[31,47,73])]
    counts=[int(v) for v in config.get("stress_subject_counts",[320,640,1280])]
    modes=[str(v) for v in config.get("stress_channel_modes",["raw","unitvar"])]
    metrics=[str(v) for v in config.get("stress_rank_metrics",["entropy","participation","tail"])]
    if any(v<24 for v in counts):raise ValueError("Stress sample count must be >=24")
    for seed in seeds:
        rng=np.random.default_rng(seed+283)
        indices=rng.permutation(len(q))
        for count in counts:
            n=min(count,len(q))
            if n<24:continue
            subset=indices[:n]
            for metric in metrics:
                for mode in modes:
                    local=dict(config,seed=seed,rank_metric=metric,channel_mode=mode,
                               starts=int(config.get("stress_starts",2)),
                               maxiter=int(config.get("stress_maxiter",90)))
                    fitted=make_proxy(q[subset],x[subset],y[subset],mask[subset],local)
                    R,info,obj=optimize(fitted,local)
                    runs.append(dict(seed=seed,subjects=n,
                                     rank_metric=metric,channel_mode=mode,
                                     surrogate_profile_r2=fitted.fidelity["heldout_profile_r2"],
                                     surrogate_y_ce=fitted.fidelity["heldout_y_cross_entropy"],
                                     objective_best=info["best"]["loss"],
                                     objective_identity=info["identity"]["loss"],
                                     R=R,alignment=best_match_rows(reference,R)))
    return runs


def save_postfit(folder,proxy,obj,inputs,config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder=Path(folder)
    with np.load(folder/"rotation__matrix.npz",allow_pickle=False) as f:
        R=np.asarray(f["R"],float)
    before=obj.components(np.eye(proxy.k))
    after=obj.components(R)
    report=dict(identity=before,best=after,
                surrogate_validation=proxy.fidelity,
                fixed_channel_metric=obj.channel_mode,
                rank_definition=obj.metric,
                caveat="Only FIT-subject proxy gradients; rank reduction is not unique CPT theta identification")
    (folder/"fingerprint__diagnostics.json").write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    rows=plane_curves(obj,R,config.get("plane_steps",73),config.get("plane_pairs",6))
    if rows:
        with (folder/"fingerprint__plane_curves.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    if config.get("make_plots",True) and rows:
        pairs=list(dict.fromkeys(r["pair"] for r in rows))
        fig,axes=plt.subplots(len(pairs),1,figsize=(9,2.4*len(pairs)),squeeze=False)
        for ax,pair in zip(axes[:,0],pairs):
            for frame in ("identity","best"):
                r=[v for v in rows if v["pair"]==pair and v["frame"]==frame]
                ax.plot([v["angle_degrees"] for v in r],[v["loss"] for v in r],label=frame)
            ax.set_title(pair);ax.set_xlabel("Incremental angle (degrees)")
            ax.set_ylabel("Rank complexity");ax.legend()
        fig.tight_layout()
        fig.savefig(folder/"fingerprint__plane_curves.png",dpi=170)
        plt.close(fig)
    if config.get("stress_enabled",False):
        tests=stress_runs(proxy,inputs,R,config)
        flat=[];saved={}
        for i,row in enumerate(tests):
            saved[f"R_{i:04d}"]=row["R"]
            flat.append({key:value for key,value in row.items()
                         if key not in ("R","alignment")})
            flat[-1].update({key:value for key,value in row["alignment"].items()
                             if key in ("mean_abs_axis_cos","minimum_abs_axis_cos")})
        np.savez_compressed(folder/"fingerprint__stress_rotations.npz",**saved)
        if flat:
            with (folder/"fingerprint__stress.csv").open("w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=list(flat[0]))
                w.writeheader();w.writerows(flat)
        (folder/"fingerprint__stress.json").write_text(
            json.dumps([{**{k:v for k,v in test.items() if k!="R"},
                         "R_name":f"R_{i:04d}"} for i,test in enumerate(tests)],
                       indent=2,allow_nan=False),encoding="utf-8")
