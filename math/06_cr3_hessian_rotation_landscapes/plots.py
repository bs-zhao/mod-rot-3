"""Each plot is a 3×3 dashboard (Y type × ZZ/ZX/XX).

Every cell contains all THREE Hessian sources: analytic CR3, regression
on analytic generated Y, regression on frozen NN generated Y.
Across three rows this visualizes all NINE requested source×Y methods.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

SOURCES=("analytic","formula_y","nn_y")
TARGETS=("choice","logit","probability")
BLOCKS=("ZZ","ZX","XX")
DEFAULT_METRICS=(
    "normalized_l1",
    "cosine_to_analytic",
    "relative_error_to_analytic",
    "max_abs_over_frob",
    "offdiag_energy_ratio",
    "row_concentration",
)
TITLES={
    "normalized_l1":"Normalized Hessian L1 (lower = sparser)",
    "cosine_to_analytic":"Cosine similarity to analytic CR3 Hessian",
    "relative_error_to_analytic":"Relative Frobenius error to analytic Hessian",
    "max_abs_over_frob":"Largest |H| / Frobenius norm",
    "offdiag_energy_ratio":"Off-diagonal Frobenius energy / total",
    "row_concentration":"Max Z-row effect / sum of Z-row effects",
    "sign_to_analytic":"Important-element sign agreement",
    "abs_rank_to_analytic":"Spearman of absolute element sizes",
    "row_entropy":"Normalized Z-row energy entropy",
    "effective_rank":"Effective Hessian rank",
    "equivariance_error":"Rotation-equivariance drift vs 0 degrees",
}


def plot_dashboard(curves,plane,angles,metric,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pair_data=[r for r in curves if r["plane"]==plane]
    fig,axs=plt.subplots(3,3,figsize=(15,10),sharex=True,
                         constrained_layout=True)
    for r,target in enumerate(TARGETS):
        for c,block in enumerate(BLOCKS):
            ax=axs[r,c]
            plotted=0
            for source in SOURCES:
                pairs=[(float(item["angle_degrees"]),item.get(metric))
                       for item in pair_data
                       if item["source"]==source and item["target"]==target and
                          item["block"]==block]
                byangle={a:np.nan if v is None else float(v) for a,v in pairs}
                ys=np.array([byangle.get(float(angle),np.nan) for angle in angles])
                if np.isfinite(ys).any():
                    ax.plot(angles,ys,marker=".",linewidth=1.5,markersize=3,
                            label={"analytic":"Exact analytic",
                                   "formula_y":"Fit: analytic Y",
                                   "nn_y":"Fit: NN Y"}[source])
                    plotted+=1
            ax.axvline(0,color="gray",linewidth=.8,linestyle=":")
            ax.grid(True,alpha=.25)
            ax.set_title(f"{target} / {block}",fontsize=10)
            if not plotted:
                ax.text(.5,.5,"Not defined for this block",
                        ha="center",va="center",transform=ax.transAxes)
            if c==0:ax.set_ylabel(metric.replace("_"," "),fontsize=8)
            if r==2:ax.set_xlabel("Rotation angle (degrees)")
    handles,labels=axs[0,0].get_legend_handles_labels()
    if handles:fig.legend(handles,labels,loc="outside upper center",
                          ncol=3,frameon=False)
    fig.suptitle(f"{plane}: {TITLES.get(metric,metric)}",fontsize=14)
    out=Path(output)
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=135)
    plt.close(fig)


def plot_heatmap_triplet(hessian,ref,title,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    blocks=(slice(0,4),slice(0,4)),(slice(0,4),slice(4,10)),(slice(4,10),slice(4,10))
    for ax,region,heading in zip(axs,blocks,("ZZ","ZX","XX")):
        a=hessian[region];b=ref[region]
        vmax=max(float(np.nanmax(abs(a))),float(np.nanmax(abs(b))),1e-12)
        m=ax.imshow(a,cmap="coolwarm",vmin=-vmax,vmax=vmax)
        ax.set_title(heading)
        fig.colorbar(m,ax=ax,shrink=.65)
    fig.suptitle(title)
    fig.savefig(path,dpi=135)
    plt.close(fig)
