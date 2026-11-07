"""Compact 3-panel figures: Choice / Logit / Probability, 3 sources each.

A plotted metric is ALWAYS an orientation-dependent candidate loss. We
include analytic, formula-Y fit, and NN-Y fit to avoid hiding the
numeric-Hessian inference error under a seemingly good NN curve.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

DEFAULT=(
    "sweep_mean_self_curvature_variance_normalized_loss",
    "sweep_mean_quadratic_approx_relative_mse_loss",
    "sweep_mean_self_curvature_tv_normalized_loss",
    "zz_self_z_third_rms_loss",
    "zz_other_z_third_rms_loss",
    "zz_diagonal_X_modulation_third_rms_loss",
    "zx_own_Z_modulation_third_rms_loss",
    "zx_own_X_modulation_third_rms_loss",
    "zx_local_row_mean_capped_l2_t0p005_loss",
    "zx_rms_zx_row_mean_logsum_t0p005_loss",
    "zz_offdiag_total_energy_fraction_loss",
)
LABELS={"analytic":"Analytic CR3","formula_y":"Fit to formula Y",
        "nn_y":"Fit to NN Y"}
COLORS={"analytic":"#233b59","formula_y":"#d18c2a","nn_y":"#34816d"}


def draw_local_charts(all_rows,output,metrics=DEFAULT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=Path(output)
    folder=output/"local_curves"
    folder.mkdir(parents=True,exist_ok=True)
    filenames=[]
    planes=sorted(set(r["plane"] for r in all_rows))
    for plane in planes:
        rows=[r for r in all_rows if r["plane"]==plane]
        ang=np.array(sorted(set(float(r["angle_degrees"]) for r in rows)))
        for metric in metrics:
            if metric not in rows[0]:continue
            fig,axs=plt.subplots(1,3,figsize=(14,4.2),sharex=True,
                                 constrained_layout=True)
            has_valid=False
            for ax,target in zip(axs,("choice","logit","probability")):
                for source in ("analytic","formula_y","nn_y"):
                    values={float(r["angle_degrees"]):r.get(metric)
                            for r in rows if r["target"]==target and
                            r["source"]==source}
                    vec=np.array([float(values.get(float(a)))
                                  if values.get(float(a)) is not None
                                  else np.nan for a in ang])
                    if np.isfinite(vec).any():
                        has_valid=True
                        ax.plot(ang,vec,label=LABELS[source],
                                color=COLORS[source],linewidth=1.75,
                                marker=".",markersize=4)
                        finite=np.where(np.isfinite(vec))[0]
                        if len(finite):
                            min_idx=finite[int(np.argmin(vec[finite]))]
                            ax.scatter([ang[min_idx]],[vec[min_idx]],s=44,
                                       facecolors="none",edgecolors=COLORS[source],
                                       zorder=5)
                ax.axvline(0,linestyle=":",color="gray",linewidth=.8)
                ax.axvline(90,linestyle=":",color="gray",linewidth=.8)
                ax.grid(alpha=.22)
                ax.set_xlabel("A* pair rotation (degrees)")
                ax.set_title(target.capitalize(),fontsize=10)
            axs[0].set_ylabel("Complexity loss (lower = simpler)")
            if has_valid:
                handles,labels=axs[0].get_legend_handles_labels()
                fig.legend(handles,labels,ncol=3,loc="upper center",
                           bbox_to_anchor=(.5,1.075),frameon=False)
                fig.suptitle(f"{plane} | {metric}",fontsize=11)
                path=folder/f"{plane}__{metric}.png"
                fig.savefig(path,dpi=140)
                filenames.append(path)
            plt.close(fig)
    (output/"PLOT_INDEX.md").write_text(
        "# Local Hessian complexity curves\n\n"+
        "\n".join(f"- [{p.name}](local_curves/{p.name})" for p in filenames)+
        "\n",encoding="utf-8")
    return filenames
