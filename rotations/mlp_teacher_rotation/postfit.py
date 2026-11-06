"""Save independent FIT rotations, 20 diagnostics, and strictly post-fit TEST audits."""
from __future__ import annotations
import csv
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from . import metrics
from .experiment import evaluate_after_fit


def _jsonable(obj):
    if isinstance(obj,dict): return {str(k):_jsonable(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)): return [_jsonable(v) for v in obj]
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,(np.floating,np.integer)): return obj.item()
    return obj


def _csv(path, rows):
    if not rows:
        return
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _loss_sweeps(bank, rotations, config):
    pairs=list(combinations(range(bank.k),2))[:int(config["sweep_pairs"])]
    scan=[name for name in config["sweep_candidates"] if name in rotations]
    steps=max(3,int(config["sweep_steps"]))
    rows=[]
    for name in scan:
        for i,j in pairs:
            for deg in np.linspace(-90.,90.,steps):
                radians=np.deg2rad(deg)
                B=np.eye(bank.k)
                B[i,i]=B[j,j]=np.cos(radians)
                B[i,j]=-np.sin(radians)
                B[j,i]=np.sin(radians)
                for frame,reference in (("identity",np.eye(bank.k)),("optimized",rotations[name])):
                    R=B@reference
                    rows.append(dict(candidate=name,plane=f"u{i+1}-u{j+1}",
                                     frame=frame,angle_deg=float(deg),
                                     loss=float(bank.score(R,name))))
    return rows


def save_diagnostics(folder, bank, rotations, optimizers, teacher_info, config):
    folder=Path(folder)
    with np.load(folder/"rotation__matrix.npz",allow_pickle=False) as file:
        Rmain=np.array(file["R"],float)
        Amain=np.array(file["A"],float)
        offset=np.array(file["offset"],float)
        pca_components=np.array(file["pca_components"],float)
        pca_mean=np.array(file["pca_mean"],float)
        eigenvalues=np.array(file["pca_eigenvalues"],float)
    with np.load(folder/"rotation__U.npz",allow_pickle=False) as file:
        Umain=np.array(file["U"],float)
        train=np.array(file["train_ids"],int)
        validation=np.array(file["validation_ids"],int)
        test=np.array(file["test_ids"],int)
    primary=str(config["primary"])
    if not np.allclose(Rmain,rotations[primary],atol=1e-8):
        raise RuntimeError("Primary saved rotation differs from FIT candidate")
    if not np.allclose(Rmain@Rmain.T,np.eye(bank.k),atol=1e-7):
        raise RuntimeError("Saved primary is not orthogonal")
    q_all=Umain@Rmain
    # Persist every candidate's full-subject transform BEFORE accessing TEST theta.
    paths={}
    for name,R in rotations.items():
        if name==primary:
            paths[name]=(folder,Umain)
            continue
        sub=folder/("candidate__"+name)
        sub.mkdir(exist_ok=True)
        U=q_all@R.T
        A=R@Rmain.T@Amain
        b=R@Rmain.T@offset
        if not np.allclose(U,Umain@(Rmain@R.T),atol=1e-7):
            raise RuntimeError("Rotation composition mismatch")
        np.savez_compressed(sub/"rotation__matrix.npz",R=R,A=A,offset=b,
                            pca_components=pca_components,pca_mean=pca_mean,
                            pca_eigenvalues=eigenvalues,variant=np.array(name))
        np.savez_compressed(sub/"rotation__U.npz",U=U,train_ids=train,
                            validation_ids=validation,test_ids=test)
        paths[name]=(sub,U)
    np.savez_compressed(folder/"teacher__candidate_rotations.npz",
                        **{("R_"+name):R for name,R in rotations.items()})

    raw_scores={}
    score_rows=[]
    reference=bank.diagnose(np.eye(bank.k))
    for name,R in rotations.items():
        scores=bank.diagnose(R)
        raw_scores[name]=scores
        for metric,value in scores.items():
            base=reference[metric]
            score_rows.append(dict(candidate=name,metric=metric,
                                   score=float(value),identity_score=float(base),
                                   delta=float(value-base)))
    _csv(folder/"teacher__all_metric_scores.csv",score_rows)

    proxy_diagnostics={}
    for name,R in (("identity",np.eye(bank.k)),(primary,rotations[primary])):
        vals,details=metrics.diagnostic_metrics(
            bank.proxy,R,config,include_expensive=bool(config["full_mdl"]))
        proxy_diagnostics[name]=dict(metrics=vals,details=details)
    # Metric19 requires discrete Y; soft teacher probabilities are not binary.
    (folder/"teacher__20_diagnostics.json").write_text(
        json.dumps(_jsonable(proxy_diagnostics),indent=2,allow_nan=False),
        encoding="utf-8")

    sweeps=_loss_sweeps(bank,rotations,config)
    _csv(folder/"teacher__plane_curves.csv",sweeps)
    evaluated=json.loads((folder/"rotation__complete.json").read_text())["evaluated"]
    reports={}
    if evaluated:
        with np.load(folder/"rotation__test_truth.npz",allow_pickle=False) as file:
            theta_test=np.asarray(file["theta_test"],float)
            test_truth_ids=np.asarray(file["test_ids"],int)
        if not np.array_equal(test_truth_ids,test):
            raise RuntimeError("TEST truth and candidate U splits differ")
        meta=json.loads((folder/"rotation__fit.json").read_text())
        theta=np.zeros((len(Umain),theta_test.shape[1]),dtype=float)
        theta[test]=theta_test
        for name,(path,U) in paths.items():
            if name==primary:
                reports[name]=json.loads((folder/"rotation__test_evaluation.json").read_text())
            else:
                reports[name]=evaluate_after_fit(
                    theta,U,test,path,meta["model"],"mlp_teacher_rotation__"+name)
    rows=[]
    for name in rotations:
        metrics_eval=reports.get(name,{})
        rows.append(dict(candidate=name,primary=(name==primary),
                         own_loss=optimizers[name].get("loss"),
                         identity_loss=optimizers[name].get("identity_loss"),
                         optimizer_converged=optimizers[name]["success"],
                         test_mean_r2=metrics_eval.get("matched_mean_r2"),
                         test_min_r2=metrics_eval.get("matched_min_r2"),
                         test_offtarget_r2=metrics_eval.get("mean_offtarget_r2")))
    _csv(folder/"teacher__candidate_comparison.csv",rows)
    report=dict(primary=primary,teacher=teacher_info,
                fit_candidate_names=list(rotations),
                optimizers=optimizers,fit_metric_scores=raw_scores,
                test_reports=reports,
                scientific_limits=[
                    "Candidate R optimization never observes theta",
                    "TEST theta is descriptive, not a metric-selector",
                    "Full-trial S5 Z is not unseen-trial generalization",
                    "Soft teacher Y removes Bernoulli sampling noise, not decoder misspecification",
                    "Profile proxy derivatives and direct decoder derivatives are distinct estimands",
                    "Profile metric19 is unavailable for non-binary soft targets",
                ])
    (folder/"teacher__summary.json").write_text(
        json.dumps(_jsonable(report),indent=2,allow_nan=False),
        encoding="utf-8")
    if config.get("make_plots",True):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(max(9,len(rows)*.57),4.4))
        values=[float(r["test_mean_r2"]) if r["test_mean_r2"] is not None else 0. for r in rows]
        ax.bar(range(len(rows)),values)
        ax.set_xticks(range(len(rows)),[r["candidate"] for r in rows],
                      rotation=65,ha="right",fontsize=8)
        ax.set_ylabel("TEST matched theta-axis Pearson r²")
        ax.set_ylim(0,1)
        ax.set_title("Descriptive TEST audit only; NOT used to select rotation")
        fig.tight_layout()
        fig.savefig(folder/"teacher__candidate_comparison.png",dpi=150)
        plt.close(fig)
    return folder
