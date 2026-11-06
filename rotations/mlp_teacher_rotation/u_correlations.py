"""Pearson inter-axis U correlation audits for this independent rotation suite.

Descriptive only: U correlations do not participate in rotation optimization,
and no theta or decoder is accessed. Uses exactly the saved all-subject U.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def u_pearson_correlation(values: np.ndarray) -> np.ndarray:
    """Signed Pearson r between columns, with safe handling of constant axes.

    A constant coordinate has undefined Pearson correlation; it is assigned 0
    (including on the diagonal) and documented in the saved audit.
    """
    x=np.asarray(values,dtype=float)
    if x.ndim!=2 or x.shape[1]<1 or not np.isfinite(x).all():
        raise ValueError("U must be a finite two-dimensional array")
    k=x.shape[1]
    if len(x)<2:
        return np.zeros((k,k),dtype=float)
    centered=x-x.mean(axis=0,keepdims=True)
    gram=centered.T@centered
    norms=np.sqrt(np.maximum(np.diag(gram),0.))
    denom=norms[:,None]*norms[None,:]
    r=np.divide(gram,denom,out=np.zeros_like(gram),where=denom>1e-24)
    r=np.clip((r+r.T)/2,-1.,1.)
    np.fill_diagonal(r,np.where(norms>1e-12,1.,0.))
    return r



def save_u_pairwise_scatter(
    folder: Path,U_test: np.ndarray,*, filename="rotation__test_u_pairwise_scatter.png"
):
    """All unordered TEST U-axis pairs, with diagonal marginal histograms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    U=np.asarray(U_test,dtype=float)
    if U.ndim!=2 or U.shape[1]<2 or not np.isfinite(U).all():
        raise ValueError("TEST U needs at least two finite columns")
    k=U.shape[1]
    r=u_pearson_correlation(U)
    fig,axes=plt.subplots(k,k,figsize=(max(7,2.3*k),max(7,2.3*k)),
                          squeeze=False)
    for i in range(k):
        for j in range(k):
            ax=axes[i,j]
            if i==j:
                ax.hist(U[:,j],bins=25,color="tab:blue",alpha=.65)
                ax.set_title(f"u{j+1} distribution",fontsize=9)
            elif i>j:
                ax.scatter(U[:,j],U[:,i],s=6,alpha=.32,
                           rasterized=True)
                ax.text(.03,.95,f"r = {r[i,j]:+.3f}",transform=ax.transAxes,
                        ha="left",va="top",fontsize=9,
                        bbox=dict(facecolor="white",edgecolor="none",alpha=.7))
            else:
                ax.text(.5,.55,f"r = {r[i,j]:+.3f}",transform=ax.transAxes,
                        ha="center",va="center",fontsize=11)
                ax.axis("off")
            if i==k-1:
                ax.set_xlabel(f"u{j+1}",fontsize=9)
            if j==0 and i!=j:
                ax.set_ylabel(f"u{i+1}",fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle(f"TEST U-U pairwise scatter matrix (n={len(U)})",fontsize=13)
    fig.tight_layout()
    fig.savefig(Path(folder)/filename,dpi=155)
    plt.close(fig)


def save_theta_u_pairwise_scatter(
    folder: Path,theta_test: np.ndarray,U_test: np.ndarray,
    parameter_names: list[str], *,
    corr_matrix: np.ndarray | None = None,
    matching: dict[int,int] | None = None,
):
    """All theta_i–U_j TEST scatterplots. Descriptive ONLY, after R/U save."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    theta=np.asarray(theta_test,dtype=float)
    U=np.asarray(U_test,dtype=float)
    if (theta.ndim!=2 or U.ndim!=2 or theta.shape[0]!=U.shape[0]
        or theta.shape[1]!=len(parameter_names)
        or not np.isfinite(theta).all() or not np.isfinite(U).all()):
        raise ValueError("Invalid TEST theta/U arrays for pairwise plots")
    ntheta,k=theta.shape[1],U.shape[1]
    c=u_cross_pearson(theta,U) if corr_matrix is None else np.asarray(corr_matrix,float)
    if c.shape!=(ntheta,k):
        raise ValueError("Correlation dimensions do not match TEST theta/U")
    fig,axes=plt.subplots(ntheta,k,figsize=(max(8,k*2.65),
                                          max(6.5,ntheta*2.65)),
                          squeeze=False)
    for i in range(ntheta):
        for j in range(k):
            ax=axes[i,j]
            ax.scatter(U[:,j],theta[:,i],s=6,alpha=.28,
                       rasterized=True)
            ax.text(.03,.97,f"r={c[i,j]:+.3f}\nR²={c[i,j]**2:.3f}",
                    ha="left",va="top",transform=ax.transAxes,fontsize=8,
                    bbox=dict(facecolor="white",edgecolor="none",alpha=.75))
            if matching and matching.get(i)==j:
                ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,
                                       fill=False,edgecolor="red",lw=2.4))
            if i==ntheta-1:
                ax.set_xlabel(f"u{j+1}",fontsize=9)
            if j==0:
                ax.set_ylabel(parameter_names[i],fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle(f"TEST theta-U all pairwise scatterplots (n={len(U)})",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(Path(folder)/"rotation__test_theta_u_pairwise_scatter.png",
                dpi=155)
    plt.close(fig)


def u_cross_pearson(a: np.ndarray,b: np.ndarray) -> np.ndarray:
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.ndim!=2 or b.ndim!=2 or len(a)!=len(b):
        raise ValueError("Expected two aligned 2D arrays")
    ac=a-a.mean(0);bc=b-b.mean(0)
    den=np.sqrt(np.sum(ac**2,axis=0)[:,None]*np.sum(bc**2,axis=0)[None,:])
    return np.divide(ac.T@bc,den,out=np.zeros((a.shape[1],b.shape[1])),
                     where=den>1e-24)

def save_u_correlation_audit(
    folder: Path, U: np.ndarray, train_ids: np.ndarray,
    validation_ids: np.ndarray, test_ids: np.ndarray,
    *, make_plots: bool = True,
) -> dict:
    """Report TRAIN/VALIDATION/TEST/ALL raw U axis correlations.

    TRAIN PCA whitening and orthogonal rotation imply almost zero TRAIN
    correlations (unless eigenvalue flooring takes effect). Other splits
    are descriptive checks, not model-fitting objectives.
    """
    folder=Path(folder)
    U=np.asarray(U,dtype=float)
    if U.ndim!=2 or not np.isfinite(U).all():
        raise ValueError("U must have shape (subjects, axes) and be finite")
    ids=dict(
        train=np.asarray(train_ids,dtype=int),
        validation=np.asarray(validation_ids,dtype=int),
        test=np.asarray(test_ids,dtype=int),
    )
    for name,index in ids.items():
        if index.ndim!=1 or np.any(index<0) or np.any(index>=len(U)):
            raise ValueError(f"Invalid {name} subject ids")
    if not np.array_equal(np.sort(np.concatenate(list(ids.values()))),np.arange(len(U))):
        raise ValueError("Subject ids must be disjoint and exhaustive")
    folder.mkdir(parents=True,exist_ok=True)
    axis_names=[f"u{j+1}" for j in range(U.shape[1])]
    matrices={name:u_pearson_correlation(U[index]) for name,index in ids.items()}
    matrices["all"]=u_pearson_correlation(U)
    summary={}
    for name,m in matrices.items():
        target=folder/f"rotation__u_correlations_{name}.csv"
        with target.open("w",encoding="utf-8",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["axis",*axis_names])
            for j,row in enumerate(m):
                writer.writerow([axis_names[j],*(float(x) for x in row)])
        mask=~np.eye(len(m),dtype=bool)
        off=np.abs(m[mask])
        summary[name]=dict(n_subjects=(len(U) if name=="all" else len(ids[name])),
                           pearson_r=m.tolist(),
                           max_offdiag_abs_r=float(off.max()) if len(off) else 0.,
                           mean_offdiag_abs_r=float(off.mean()) if len(off) else 0.)
    result=dict(
        axis_names=axis_names,
        method="signed Pearson correlation of columns of the saved rotated U",
        constant_axis_policy="A constant U axis has undefined Pearson r and is shown as 0, including diagonal.",
        train_whitening_note="If the rotation uses TRAIN PCA whitening without active eigenvalue floor, TRAIN U-U Pearson correlations are zero by construction; this does not prove statistical independence.",
        splits=summary,
    )
    (folder/"rotation__u_correlations.json").write_text(
        json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    if make_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def draw(ax,matrix,title):
            im=ax.imshow(matrix,vmin=-1,vmax=1,cmap="coolwarm")
            ax.set_xticks(range(len(axis_names)),axis_names,rotation=45,ha="right")
            ax.set_yticks(range(len(axis_names)),axis_names)
            ax.set_title(title)
            for i in range(len(axis_names)):
                for j in range(len(axis_names)):
                    value=float(matrix[i,j])
                    ax.text(j,i,f"{value:+.2f}",ha="center",va="center",
                            fontsize=8,color="white" if abs(value)>.65 else "black")
            return im

        fig,ax=plt.subplots(figsize=(max(5.5,.8*len(axis_names)+2),max(4.5,.8*len(axis_names)+1)))
        im=draw(ax,matrices["test"],"TEST: Pearson correlation between U axes")
        fig.colorbar(im,ax=ax,label="Signed Pearson r",shrink=.85)
        fig.tight_layout()
        fig.savefig(folder/"rotation__test_u_corr.png",dpi=170)
        plt.close(fig)

        fig,axs=plt.subplots(2,2,figsize=(max(10,1.65*len(axis_names)+3),
                                          max(9,1.65*len(axis_names)+1)))
        for ax,name in zip(axs.flat,("train","validation","test","all")):
            im=draw(ax,matrices[name],f"{name.upper()} (n={summary[name]['n_subjects']})")
        fig.colorbar(im,ax=list(axs.flat),label="Signed Pearson r",shrink=.75)
        fig.subplots_adjust(left=.07,right=.88,bottom=.08,top=.93,wspace=.30,hspace=.40)
        fig.savefig(folder/"rotation__u_corr_by_split.png",dpi=170,bbox_inches="tight")
        plt.close(fig)
        save_u_pairwise_scatter(folder,U[ids["test"]])
    return result
