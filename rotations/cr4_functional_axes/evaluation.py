"""Post-freeze-only CR3/CR4/CR5 evaluation. Not imported by objective."""
from __future__ import annotations

import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def corr(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    a=a-a.mean(0);b=b-b.mean(0)
    den=np.sqrt((a*a).sum(0)[:,None]*(b*b).sum(0)[None,:])
    return np.divide(a.T@b,den,out=np.zeros((a.shape[1],b.shape[1])),
                     where=den>1e-12)


def parameter_diagnostics(source,U,R,out):
    if not (out/"fit_complete.json").exists() or not (out/"rotation.npz").exists():
        raise RuntimeError("Persist theta-free rotation before evaluation")
    with np.load(source["dataset"],allow_pickle=False) as file:
        theta=np.asarray(file["parameters"],dtype=float)
    labels=(["alpha","lambda_loss","gamma","tau"]
            if source["model"]=="CR3" else ["lambda_loss","gamma","tau"])
    if theta.shape != (len(U),len(labels)):
        raise ValueError(f"{source['model']}: expected theta columns {len(labels)}, got {theta.shape}")
    tr,te=source["tr"],source["te"]
    scores={}
    for label,arr in (("candidate",U),("identity_pca",source["q"])):
        c_train=corr(theta[tr],arr[tr])
        c_test=corr(theta[te],arr[te])
        i,j=linear_sum_assignment(-c_train*c_train)
        matched=np.asarray([c_test[x,y]**2 for x,y in zip(i,j)])
        assignments={labels[x]:int(y) for x,y in zip(i,j)}
        parameter_r2={name:(float(c_test[x,assignments[name]]**2)
                            if name in assignments else None)
                      for x,name in enumerate(labels)}
        off=np.ones_like(c_test,dtype=bool)
        off[i,j]=False
        scores[label]=dict(
            labels=labels,rows=labels,
            columns=[f"u{k+1}" for k in range(U.shape[1])],
            signed_train_corr=c_train.tolist(),signed_test_corr=c_test.tolist(),
            matched_train_assignment=assignments,
            unmatched_parameters=[name for name in labels if name not in assignments],
            total_parameters=len(labels),matched_parameters=len(assignments),
            full_parameter_axis_recovery_possible=(len(labels)==U.shape[1]),
            parameter_test_r2=parameter_r2,
            mean_test_r2=float(np.mean(matched)),min_test_r2=float(np.min(matched)),
            mean_test_r2_scope="matched parameters only; NOT all four CR3 parameters if L3" if len(labels)!=U.shape[1] else "all parameters",
            off_target_mean_test_r2=float(np.mean(c_test[off]**2)),
            diagnostic_test_optimal_mean_r2=float(np.mean(
                np.square(c_test[linear_sum_assignment(-c_test*c_test)]))),
        )
        fig,ax=plt.subplots(figsize=(6.1,4.9 if len(labels)==4 else 4.5))
        im=ax.imshow(c_test,vmin=-1,vmax=1,cmap="coolwarm")
        ax.set(xticks=range(U.shape[1]),yticks=range(len(labels)),
               xticklabels=[f"u{k+1}" for k in range(U.shape[1])],
               yticklabels=labels,title=f"{source['model']} {label}: held-out TEST Pearson r")
        for ii in range(len(labels)):
            for jj in range(U.shape[1]):
                ax.text(jj,ii,f"{c_test[ii,jj]:+.2f}",ha="center",va="center",
                        fontsize=10)
        fig.colorbar(im,ax=ax,shrink=.78)
        fig.tight_layout();fig.savefig(out/(label+"_correlations.png"),dpi=165)
        plt.close(fig)
    (out/"evaluation.json").write_text(json.dumps(scores,indent=2),encoding="utf-8")
    return scores
