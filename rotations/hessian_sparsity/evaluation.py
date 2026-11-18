"""POST-FREEZE true-parameter recovery (never imported from loss/optimizer).

Use TRAIN theta ONLY to define signed/permutation assignment, then report
the frozen assignment on TEST subjects. Report the diagnostic TEST-optimal
assignment separately, never use it to choose method/objective or R.
"""
from __future__ import annotations
import json
import numpy as np
from scipy.optimize import linear_sum_assignment


def corr(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    a=a-a.mean(0);b=b-b.mean(0)
    norm=np.sqrt((a*a).sum(0)[:,None]*(b*b).sum(0)[None,:])
    return np.divide(a.T@b,norm,out=np.zeros((a.shape[1],b.shape[1])),
                     where=norm>1e-15)


def evaluate(source,U,out):
    with np.load(source["dataset"],allow_pickle=False) as saved:
        theta=np.asarray(saved["parameters"],dtype=float)
    labels=(["alpha","lambda_loss","gamma","tau"] if source["model"]=="CR3"
            else ["lambda_loss","gamma","tau"])
    if theta.shape!=(len(U),len(labels)):
        raise ValueError("True parameter dimensions differ from model")
    if not (out/"fit_complete.json").exists():
        raise RuntimeError("Must freeze and persist R before true-theta evaluation")
    tr,te=source["train_ids"],source["test_ids"]
    reports={}
    for variant,data in (("hessian_sparsity",U),("identity_pca",source["q"])):
        ctrain=corr(theta[tr],data[tr])
        ctest=corr(theta[te],data[te])
        i,j=linear_sum_assignment(-ctrain**2)
        paired={labels[a]:int(b) for a,b in zip(i,j)}
        per_param={labels[a]:float(ctest[a,b]**2) for a,b in zip(i,j)}
        off=np.ones_like(ctest,dtype=bool);off[i,j]=False
        ii,jj=linear_sum_assignment(-ctest**2)
        reports[variant]=dict(
            train_correlation=ctrain.tolist(),
            test_correlation=ctest.tolist(),
            match_train_only=paired,
            matched_test_r2=per_param,
            mean_matched_test_r2=float(np.mean(list(per_param.values()))),
            min_matched_test_r2=float(np.min(list(per_param.values()))),
            off_target_mean_test_r2=float(np.mean(ctest[off]**2)) if np.any(off) else 0.,
            test_optimal_assignment_r2_DIAGNOSTIC_ONLY=float(
                np.mean(ctest[ii,jj]**2)),
            labels=labels,axis_names=[f"u{i+1}" for i in range(U.shape[1])])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(6.2,5.2))
        im=ax.imshow(ctest,vmin=-1,vmax=1,cmap="coolwarm")
        ax.set(xticks=np.arange(U.shape[1]),yticks=np.arange(len(labels)),
               xticklabels=[f"u{i+1}" for i in range(U.shape[1])],
               yticklabels=labels,
               title=f"{source['model']} {variant}: TEST Pearson r")
        for a in range(len(labels)):
            for b in range(U.shape[1]):
                ax.text(b,a,f"{ctest[a,b]:+.2f}",ha="center",va="center")
        fig.colorbar(im,ax=ax,shrink=.76)
        fig.tight_layout()
        fig.savefig(out/f"{variant}_test_correlations.png",dpi=160)
        plt.close(fig)
    (out/"evaluation.json").write_text(json.dumps(reports,indent=2),
                                         encoding="utf-8")
    return reports
