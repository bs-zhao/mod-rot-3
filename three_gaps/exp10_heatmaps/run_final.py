"""EXP10b - U-T heatmaps of the EXP11 final pipeline vs the earlier frozen method and the Oracle (TEST subjects, post-hoc)."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from scipy.optimize import linear_sum_assignment
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import io
from common.rotlib import corr_matrix
MODELS = [("CR3", 0.00005, 4), ("C1", 0.0005, 4), ("C2", 0.0005, 4), ("C3", 0.0005, 6)]
EX = io.ROOT / "outputs" / "three_gaps"; OUT = EX / "exp10"; OUT.mkdir(exist_ok=True, parents=True)
def panel(U, T):
    C = np.abs(corr_matrix(U, T)); r, c = linear_sum_assignment(-C); off = C.copy(); off[r, c] = np.nan
    return C[:, c], C[r, c].mean(), np.nanmean(off), np.nanmax(off)
rows = []
for rep in [1, 2]:
    fig, axs = plt.subplots(4, 5, figsize=(12.5, 10.5))
    for i, (m, beta, zd) in enumerate(MODELS):
        data, truth = io.load_all(m, beta, zd, 0, rep, need_checkpoint=False); te = data["test_ids"]; V, T = data["V"], truth.std()
        cond = f"{m}_{data['condition']}_pca0"
        P = {"Identity": V[te], "Oracle (supervised)": V[te] @ np.load(EX / "exp00" / cond / "R_oracle_raw.npy").T}
        for ch in ["logit", "prob"]:
            P[f"New (decoder {ch})"] = V[te] @ np.load(EX / "exp11" / "final" / f"Q_{m}_rep{rep}_{ch}.npy").T
        if rep == 1:
            P["Earlier frozen (logit)"] = V[te] @ np.load(EX / "exp06" / "hm" / cond / "Q" / "decoder_sieve_logit_AB_absL1.npy").T
        cols = ["Identity", "Earlier frozen (logit)", "New (decoder logit)", "New (decoder prob)", "Oracle (supervised)"]
        for j, lab in enumerate(cols):
            ax = axs[i, j]
            if lab not in P: ax.axis("off"); continue
            M, mcc, lm, lx = panel(P[lab], T[te]); d = len(M)
            ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
            for a in range(d):
                for b in range(d):
                    ax.text(b, a, f"{M[a, b]:.2f}".lstrip("0"), ha="center", va="center", fontsize=8 if d == 4 else 6, color="w" if M[a, b] < .6 else "k")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title((lab + "\n" if i == 0 else "") + f"mcc {mcc:.3f} leak {lm:.3f}/{lx:.2f}", fontsize=8)
            if j == 0: ax.set_ylabel(m, rotation=0, labelpad=18, fontsize=11)
    fig.suptitle(f"|Corr(U, Tstd)| on TEST subjects, NP repeat {rep}", fontsize=10); fig.tight_layout()
    fig.savefig(OUT / f"final_heatmaps_rep{rep}.png", dpi=140); plt.close(fig)
print("saved")
