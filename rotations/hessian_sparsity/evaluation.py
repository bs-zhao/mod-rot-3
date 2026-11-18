"""Post-freeze ALL-SUBJECT parameter-alignment diagnostic, never used in R fit.

There is no train/test division in the rotation research phase. Parameter
matching and squared correlations are descriptive in-sample full-population
statistics. NP may still have historical training splits, but they are not
used here. True theta is read only AFTER the rotation has been frozen.
"""
from __future__ import annotations
import json
import numpy as np
from scipy.optimize import linear_sum_assignment


def corr(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    a=a-a.mean(0);b=b-b.mean(0)
    den=np.sqrt((a*a).sum(0)[:,None]*(b*b).sum(0)[None,:])
    return np.divide(a.T@b,den,out=np.zeros((a.shape[1],b.shape[1])),
                     where=den>1e-12)


def evaluate(source,U,out):
    if not (out/"fit_complete.json").exists():
        raise RuntimeError("Freeze theta-blind Hessian R before parameter evaluation")
    with np.load(source["dataset"],allow_pickle=False) as file:
        theta=np.asarray(file["parameters"],dtype=float)
    labels=(["alpha","lambda_loss","gamma","tau"] if source["model"]=="CR3"
            else ["lambda_loss","gamma","tau"])
    if theta.shape != (len(U),len(labels)):
        raise ValueError("Truth parameters and U must represent all subjects")
    reports={}
    for label,arr in (("hessian_sparsity",U),("identity_pca",source["q"])):
        c_all=corr(theta,arr)
        i,j=linear_sum_assignment(-c_all*c_all)
        per_param={labels[x]:float(c_all[x,y]**2) for x,y in zip(i,j)}
        off=np.ones_like(c_all,dtype=bool);off[i,j]=False
        reports[label]=dict(
            scope="ALL subjects; no held-out TEST; descriptive in-sample",
            n_subjects=int(len(U)),
            all_correlation=c_all.tolist(),
            matched_all_assignment={labels[x]:int(y) for x,y in zip(i,j)},
            parameter_all_r2=per_param,
            mean_all_r2=float(np.mean(list(per_param.values()))),
            min_all_r2=float(np.min(list(per_param.values()))),
            off_target_mean_all_r2=float(np.mean(c_all[off]**2)) if np.any(off) else 0.,
            labels=labels,axes=[f"u{k+1}" for k in range(U.shape[1])])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(6.1,5.))
        im=ax.imshow(c_all,vmin=-1,vmax=1,cmap="coolwarm")
        ax.set(xticks=range(U.shape[1]),yticks=range(len(labels)),
               xticklabels=[f"u{k+1}" for k in range(U.shape[1])],
               yticklabels=labels,
               title=f"{source['model']} {label}: ALL-subject Pearson r")
        for ii in range(len(labels)):
            for jj in range(U.shape[1]):
                ax.text(jj,ii,f"{c_all[ii,jj]:+.2f}",
                        ha="center",va="center",fontsize=10)
        fig.colorbar(im,ax=ax,shrink=.78)
        fig.tight_layout()
        fig.savefig(out/f"{label}_all_correlations.png",dpi=165)
        plt.close(fig)
    (out/"evaluation_all_subjects.json").write_text(
        json.dumps(reports,indent=2),encoding="utf-8")
    return reports
