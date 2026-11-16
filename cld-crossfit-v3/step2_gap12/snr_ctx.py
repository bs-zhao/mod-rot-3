"""Step 2b - why does Gap 2 fail, and does pooling items into W-only stimulus contexts rescue it?

(1) Signal-to-noise of the per-item fits on binary choices: fields fitted on the observed choices vs the same fit on the noise-free
    true probabilities (truncation only), in the TRUE axes.  Reported for g and H_UU, and for the item-level energy cells.
(2) Pooled cells.  A cross-fitted energy is unbiased, so averaging it over the items of a context averages the noise away (a plain
    squared energy keeps its floor however many items are pooled).  Contexts use the stimulus columns only:
      item     every stimulus is its own cell (500)
      flags    pattern of "column sits at its modal value" flags (harness.mode_contexts with no merging)
      mode     harness.mode_contexts (small groups merged)        pure   harness.pure_cells (>= 2 flags)
      kmN      k-means on standardised X, N clusters
    Each (source, context) is searched with the item-level support losses and reported against the true axes.

python cld-crossfit-v3/step2_gap12/snr_ctx.py [FIT=lpm3] [STARTS=24] [TAG=v1]
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "common"))
import xfit as XF
from xfit import X, io
from common.analytic import make_h
from common.derivs import base_derivatives
torch.set_num_threads(6)

fit = sys.argv[1] if len(sys.argv) > 1 else "lpm3"
starts = int(sys.argv[2]) if len(sys.argv) > 2 else 24
tag = sys.argv[3] if len(sys.argv) > 3 else "v1"
RES = HERE / "results"; RES.mkdir(exist_ok=True)
dev = io.pick_device("auto"); model = "CR3"
du, link = {"logit2": (2, "logit"), "logit3": (3, "logit"), "lpm2": (2, "identity"), "lpm3": (3, "identity")}[fit]

data, truth = io.load_all(model, *XF.BETA[model], 0, 1, need_checkpoint=False)
c, cond, _ = X.load(model, 1, "prob")
ids, Xb, tr, te = c["ids"], c["X"].double(), data["train_ids"], data["test_ids"]
Tst = truth.std(); d = Tst.shape[1]; K = Xb.shape[0]
Q0 = XF.haar(d, np.random.default_rng(20261115)); W = Tst @ Q0.T; M_true = Q0.T
Y = data["y"].reshape(-1, K).astype(np.float64); P = data["predictive_true"].reshape(-1, K)
t_grid = torch.tensor(W[ids], dtype=torch.float64)
Qt = torch.tensor(Q0, dtype=torch.float64, device=dev)

# ---------------------------------------------------------------- fields
h0 = make_h(model, "prob", truth.mu, truth.sd)
b = base_derivatives(lambda w, x: h0(Qt.T @ w, x), t_grid.to(dev), Xb.to(dev), which=("g", "Htt"), chunk=2048, out_dtype=torch.float64)
F_an = [{k: v.cpu() for k, v in b.items()}]
F_tr = [XF.sieve_bank_fields(W, P, [tr], W[ids], du=du, link=link, ridge=0.0 if link == "logit" else 1e-10, dev=dev)[0][0]["prob"]]
F_wy = {Kf: [f["prob"] for f in XF.sieve_bank_fields(W, Y, XF.split_folds(tr, Kf, 20261115), W[ids], du=du, link=link, dev=dev)[0]] for Kf in (2, 5)}

# ---------------------------------------------------------------- (1) signal to noise in the true axes
A_true = XF.effective_A(torch.tensor(M_true, dtype=torch.float64), t_grid)
print(f"== {fit}: relative error of the binary-choice fit against the noise-free fit of the same polynomial (true axes)")
for Kf in (2, 5):
    for nm in ("g", "Htt"):
        ref = F_tr[0][nm]; one = F_wy[Kf][0][nm]; avg = sum(f[nm] for f in F_wy[Kf]) / Kf
        print(f"  folds {Kf}  {nm:3s}: one fold |err|^2/|signal|^2 = {float(((one - ref) ** 2).mean() / (ref ** 2).mean()):.2f}   "
              f"mean of folds = {float(((avg - ref) ** 2).mean() / (ref ** 2).mean()):.2f}")
e_ref = XF.e_uui(A_true, XF.moments(F_tr, cross=False)).flatten()
for Kf in (2, 5):
    ex = XF.e_uui(A_true, XF.moments(F_wy[Kf], cross=True)).flatten(); ep = XF.e_uui(A_true, XF.moments(F_wy[Kf], cross=False)).flatten()
    cc = lambda a: float(np.corrcoef(a.numpy(), e_ref.numpy())[0, 1])
    print(f"  folds {Kf}  item-level UU energy cells (K x d x d): corr with noise-free  cross {cc(ex):.3f}  plain {cc(ep):.3f} ; "
          f"mean energy  true {float(e_ref.mean()):.2e}  cross {float(ex.mean()):.2e}  plain {float(ep.mean()):.2e}", flush=True)

# ---------------------------------------------------------------- (2) contexts
Xn = Xb.numpy()
def flags(Xn, freq=0.10):
    n, p = Xn.shape; f = np.zeros((n, p), int)
    for j in range(p):
        v, cn = np.unique(Xn[:, j], return_counts=True); i = int(np.argmax(cn))
        if cn[i] / n >= freq: f[:, j] = (Xn[:, j] == v[i])
    keys = [tuple(r) for r in f]; uk = sorted(set(keys)); return np.array([uk.index(k) for k in keys])
def kmeans(Xn, k, seed=0, iters=100):
    Z = (Xn - Xn.mean(0)) / Xn.std(0); rng = np.random.default_rng(seed); C = Z[rng.choice(len(Z), k, replace=False)]
    for _ in range(iters):
        lab = ((Z[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.stack([Z[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return np.unique(lab, return_inverse=True)[1]
CTX = {"item": np.arange(K), "flags": flags(Xn), "mode": X.H.mode_contexts(Xn), "pure": X.H.pure_cells(Xn),
       "km8": kmeans(Xn, 8), "km16": kmeans(Xn, 16), "km32": kmeans(Xn, 32), "km64": kmeans(Xn, 64)}
print("contexts:", {k: int(v.max()) + 1 for k, v in CTX.items()}, flush=True)

COMBOS = [("UUI_softL0_03",), ("UUI_softL0_10",), ("UUI_softL0_30",), ("GI_softL0_10",), ("TOT_softL0_10",),
          ("UUI_softL0_10", "GI_softL0_10"), ("UUI_softL0_10", "UUd_share"), ("UUI_softL0_30", "UUd_share"), ("UUd_share",)]
SRC = {"analytic": (F_an, False), f"trunc:{fit}": (F_tr, False), f"wy:{fit}:2x": (F_wy[2], True), f"wy:{fit}:2p": (F_wy[2], False),
       f"wy:{fit}:5x": (F_wy[5], True), f"wy:{fit}:5p": (F_wy[5], False)}
rows = []
for sname, (F, cross) in SRC.items():
    m0 = XF.moments(F, cross=cross)
    for cname, lab in CTX.items():
        t0 = time.time(); m = XF.pool_items(m0, lab)
        out = XF.run_losses(m, t_grid, COMBOS, W, Tst, te, M_ref=M_true, starts=starts)
        for row, M in out:
            row.update(source=sname, ctx=cname, ncell=int(lab.max()) + 1, ang_true=XF.axis_error(M @ Q0)[0]); rows.append(row)
        print(f"{sname:14s} {cname:5s} " + " ".join(f"{r['loss'].replace('_softL0_', '')}:{r['ang_true']:.0f}" for r, _ in out) + f"  ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(RES / f"snr_ctx_{fit}_{tag}.csv", index=False)
print("done")
