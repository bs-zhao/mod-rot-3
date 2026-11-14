"""EXP10 - three-gap U-T correlation heatmaps (post-hoc display, T used only for evaluation).

Needs the rotations saved by
  python three_gaps/exp06_two_stage/run.py ... --view GCat --losses absL1 --variants AB --tag hm --save-q
Shows, on TEST subjects, |Corr(U_i, Tstd_j)| for the frozen two-stage method at each gap, next to the
identity (no rotation), the supervised Oracle rotation, and the step5 WY-only candidates.
Columns of every matrix are reordered by Hungarian matching for display only.

python three_gaps/exp10_heatmaps/run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import io  # noqa: E402
from common.rotlib import corr_matrix  # noqa: E402

MODELS = [("CR3", 0.00005, 4), ("C1", 0.0005, 4), ("C2", 0.0005, 4), ("C3", 0.0005, 6)]
GAL = io.ROOT / "outputs" / "supervised_abs_rotation" / "retained_metric_rotation_gallery"
STEP5 = [("Step5 BU3 (NN-p)", "30_BU3_p_nn"), ("Step5 CK (NN-p)", "01_CK_p_nn")]
OUT = io.ROOT / "outputs" / "three_gaps" / "exp10"


def stats(C):
    r, c = linear_sum_assignment(-C)
    d = len(C)
    off = C.copy(); off[r, c] = np.nan
    return dict(mcc=float(C[r, c].mean()), worst=float(C[r, c].min()),
                leak_mean=float(np.nanmean(off)), leak_max=float(np.nanmax(off))), c


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, panels = [], {}
    for model, beta, zdim in MODELS:
        data, truth = io.load_all(model, beta, zdim, 0, need_checkpoint=False)
        te = data["test_ids"]
        V, Tst = data["V"], truth.std()
        cond = f"{model}_{data['condition']}_pca0"
        qdir = io.ROOT / "outputs" / "three_gaps" / "exp06" / "hm" / cond / "Q"
        R_or = np.load(io.ROOT / "outputs" / "three_gaps" / "exp00" / cond / "R_oracle_raw.npy")
        P = {}
        P["Identity"] = V[te]
        for lab, name in STEP5:
            f = GAL / name / model / "best_available_search" / "U.npy"
            if f.exists():
                P[lab] = np.load(f)[te]
        def Qu(src, ch, coords):
            f = qdir / f"{src}_{ch}_AB_absL1.npy"
            return coords[te] @ np.load(f).T if f.exists() else None
        for lab, src, ch, coords in [("Gap1 analytic logit", "analytic", "logit", Tst),
                                      ("Gap2 formula-WY logit", "formula_sieve", "logit", Tst),
                                      ("Gap3 decoder-WY logit", "decoder_sieve", "logit", V),
                                      ("Gap3 decoder-WY prob", "decoder_sieve", "prob", V),
                                      ("Gap3 decoder-WY binary", "decoder_sieve", "binary", V)]:
            u = Qu(src, ch, coords)
            if u is not None:
                P[lab] = u
        P["Oracle (supervised)"] = V[te] @ R_or.T
        panels[model] = {}
        for lab, U in P.items():
            C = np.abs(corr_matrix(U, Tst[te]))
            s, c = stats(C)
            panels[model][lab] = (C[:, c], s)
            rows.append(dict(model=model, panel=lab, **s))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "leakage_summary.csv", index=False)
    cols = ["Identity", "Step5 BU3 (NN-p)", "Step5 CK (NN-p)", "Gap1 analytic logit", "Gap2 formula-WY logit",
            "Gap3 decoder-WY logit", "Gap3 decoder-WY prob", "Gap3 decoder-WY binary", "Oracle (supervised)"]
    def draw(ax, M, s, title):
        ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        d = len(M)
        for i in range(d):
            for j in range(d):
                ax.text(j, i, f"{M[i, j]:.2f}".lstrip("0") if M[i, j] < 1 else "1", ha="center", va="center",
                        fontsize=8 if d == 4 else 6, color="w" if M[i, j] < .6 else "k")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title}\nmcc {s['mcc']:.3f}  leak {s['leak_mean']:.3f}/{s['leak_max']:.2f}", fontsize=7)
    fig, axs = plt.subplots(len(MODELS), len(cols), figsize=(2.3 * len(cols), 2.6 * len(MODELS)))
    for i, (model, _, _) in enumerate(MODELS):
        for j, lab in enumerate(cols):
            ax = axs[i, j]
            if lab in panels[model]:
                draw(ax, *panels[model][lab], lab if i == 0 else "")
                if i != 0:
                    ax.set_title(f"mcc {panels[model][lab][1]['mcc']:.3f}  leak {panels[model][lab][1]['leak_mean']:.3f}/{panels[model][lab][1]['leak_max']:.2f}", fontsize=7)
            else:
                ax.axis("off")
            if j == 0:
                ax.set_ylabel(model, fontsize=11, rotation=0, labelpad=22)
    fig.suptitle("|Corr(U_i, Tstd_j)| on TEST subjects; rows U axes, columns T (reordered by matching). "
                 "leak = mean/max off-diagonal", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "three_gap_heatmaps_all.png", dpi=150)
    plt.close(fig)
    for model, _, _ in MODELS:
        labs = [l for l in cols if l in panels[model]]
        fig, axs = plt.subplots(1, len(labs), figsize=(3.0 * len(labs), 3.4))
        for ax, lab in zip(np.atleast_1d(axs), labs):
            draw(ax, *panels[model][lab], lab)
        fig.suptitle(f"{model}: |Corr(U, Tstd)| TEST", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"heatmaps_{model}.png", dpi=150)
        plt.close(fig)
    piv = df.pivot(index="panel", columns="model", values="leak_mean").reindex(cols)
    print(df.pivot(index="panel", columns="model", values="mcc").reindex(cols).round(3).to_string())
    print(piv.round(3).to_string())
    print(df.pivot(index="panel", columns="model", values="leak_max").reindex(cols).round(3).to_string())


if __name__ == "__main__":
    main()
