"""EXP13 heatmaps: |Corr(U, Tstd)| (TEST) of the role-free simplicity losses vs identity / Oracle (T used only for display).
python heatmaps.py [REPEAT]"""
import sys, numpy as np, torch
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "exp11_cr3_search"))
import harness as H, losses2 as L2, screen as SC
from common.rotlib import corr_matrix, effective_A, haar
from scipy.optimize import linear_sum_assignment
torch.set_num_threads(4)
rep = int(sys.argv[1]) if len(sys.argv) > 1 else 1
OUT = SC.OUT
def panel(U, T):
    C = np.abs(corr_matrix(U, T)); r, c = linear_sum_assignment(-C); off = C.copy(); off[r, c] = np.nan
    return C[:, c], C[r, c].mean(), np.nanmean(off), np.nanmax(off)
def search(c, names, SX, T4, starts=24):
    d = c["g"].shape[-1]; t_dev = torch.tensor(c["V"][c["ids"]], dtype=torch.float32)
    term = {k: (lambda A, k=k: L2.LOSSES[k](*L2.energies(A, SX, T4))) for k in names}
    rng = np.random.default_rng(1); Ms = [torch.tensor(haar(d, rng), dtype=torch.float32) for _ in range(200)]
    with torch.no_grad():
        sc = {k: np.std([float(term[k](effective_A(M, t_dev))) for M in Ms]) + 1e-9 for k in names}
    f = lambda b, A, M, cc: sum(term[k](A) / sc[k] for k in names)
    return H.optimize(c, f, "SO", F=None, starts=starts, steps=250, seed=7)[0][1]
COLS = [("Identity", None), ("UX sparsity (UX_neff)", ["UX_neff"]), ("U self-curvature share", ["UUd_share"]),
        ("UX_neff + curvature share", ["UX_neff", "UUd_share"]), ("Oracle (supervised)", "oracle")]
rows = []
for ch in ["logit", "prob"]:
    fig, axs = plt.subplots(4, 5, figsize=(12.5, 10.5))
    for i, model in enumerate(["CR3", "C1", "C2", "C3"]):
        c, cond = SC.load(model, rep, ch); d = c["g"].shape[-1]; SX, T4 = L2.moments(c)
        te = c["te"]; V = c["V"]; T = c["Tst"]
        R = np.load(SC.io.ROOT / "outputs" / "three_gaps" / "exp00" / f"{model}_{cond}_pca0" / "R_oracle_raw.npy")
        for j, (lab, spec) in enumerate(COLS):
            if spec is None: M = np.eye(d)
            elif spec == "oracle": M = R
            else:
                M = search(c, spec, SX, T4)
                np.save(OUT / f"Q_{model}_rep{rep}_{ch}_{'+'.join(spec)}.npy", M)
            Mx, mcc, lm, lx = panel(V[te] @ M.T, T[te])
            rows.append(dict(model=model, channel=ch, rep=rep, panel=lab, mcc=mcc, leak=lm, leak_max=lx))
            ax = axs[i, j]; ax.imshow(Mx, vmin=0, vmax=1, cmap="viridis")
            for a in range(d):
                for b in range(d):
                    ax.text(b, a, f"{Mx[a, b]:.2f}".lstrip("0"), ha="center", va="center", fontsize=8 if d == 4 else 6, color="w" if Mx[a, b] < .6 else "k")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title((lab + "\n" if i == 0 else "") + f"mcc {mcc:.3f} leak {lm:.3f}/{lx:.2f}", fontsize=8)
            if j == 0: ax.set_ylabel(model, rotation=0, labelpad=18, fontsize=11)
        print(model, ch, "done", flush=True)
    fig.suptitle(f"Role-free W-Y simplicity losses (NN decoder {ch}), |Corr(U, Tstd)| TEST, NP repeat {rep}", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / f"heatmaps_simplicity_{ch}_rep{rep}.png", dpi=140); plt.close(fig)
import pandas as pd; pd.DataFrame(rows).to_csv(OUT / f"heatmaps_simplicity_rep{rep}.csv", index=False)
print("saved")
