"""Step 3c - Gap 3 without any decoder second derivative: per-item polynomial fits on the OBSERVED choices in the frozen-Z coordinates V.

Strict W-Y route for the UU part: (V, trial id, Y) of the TRAIN subjects only.  Losses
  L0   frozen EXP13 loss, UX from the original decoder (autodiff, as in EXP13), UU moment replaced by the cross-fitted fit
  L1   U x item interaction (UI_neff) + UUd_share, both from the fit
  L2   item-level UU support count (+ UUd_share / first-order support), items or W-only contexts as cells
T / Oracle only in the post-hoc report.

python cld-crossfit-v3/step3_gap3/run_sieve.py [REPS=1,2] [STARTS=24] [TAG=v1]
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "common"))
import xfit as XF
from xfit import X, io
torch.set_num_threads(4)
reps = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,2").split(",")]
starts = int(sys.argv[2]) if len(sys.argv) > 2 else 24
tag = sys.argv[3] if len(sys.argv) > 3 else "v1"
RES = HERE / "results"; RES.mkdir(exist_ok=True)
dev = io.pick_device("auto"); model = "CR3"
FITS = {"lpm2": (2, "identity"), "lpm3": (3, "identity"), "logit2": (2, "logit")}
COMBOS = [("UX_neff", "UUd_share"), ("UUd_share",), ("UI_neff", "UUd_share"),
          ("UUI_softL0_10",), ("UUI_softL0_10", "UUd_share"), ("UUI_softL0_10", "GI_softL0_10"), ("GI_softL0_10",),
          ("UX_neff", "UUd_share", "UUI_softL0_10")]
rows = []
for rep in reps:
    data, truth = io.load_all(model, *XF.BETA[model], 0, rep, need_checkpoint=False)
    c, cond, R = X.load(model, rep, "prob")
    V, Tst, tr, te, ids = data["V"], truth.std(), data["train_ids"], data["test_ids"], c["ids"]
    Xn = c["X"].numpy(); K = Xn.shape[0]; Y = data["y"].reshape(-1, K).astype(np.float64)
    t_grid = torch.tensor(V[ids], dtype=torch.float64)
    SX = XF.moments([dict(g=c["g"], Htt=c["Htt"], Htx=c["Htx"])], cross=False, per_item=False)["SX"]      # decoder UX, exactly as in EXP13
    CTX = {"item": np.arange(K), "mode": X.H.mode_contexts(Xn), "pure": X.H.pure_cells(Xn)}
    print(f"== CR3 rep{rep}: oracle %.3f/%.3f/%.3f" % XF.evaluate(V, Tst, te, R), flush=True)
    for fit, (du, link) in FITS.items():
        for Kf, cross in ((2, True), (5, True), (2, False)):
            F = [f["prob"] for f in XF.sieve_bank_fields(V, Y, XF.split_folds(tr, Kf, 20261115), V[ids], du=du, link=link, dev=dev)[0]]
            m0 = XF.moments(F, cross=cross); m0["SX"] = SX
            for cname, lab in CTX.items():
                t0 = time.time()
                out = XF.run_losses(XF.pool_items(m0, lab), t_grid, COMBOS if cname == "item" else [x for x in COMBOS if any("I_" in k for k in x)],
                                    V, Tst, te, M_ref=R, starts=starts)
                for row, M in out:
                    row.update(rep=rep, fit=fit, folds=Kf, cross=int(cross), ctx=cname); rows.append(row)
                print(f"  {fit}:{Kf}{'x' if cross else 'p'} {cname:5s} " + " | ".join(f"{r['loss'].replace('_softL0_', '')} {r['mcc']:.3f}/{r['leak']:.3f}/{r['leak_max']:.2f}" for r, _ in out) +
                      f"  ({time.time()-t0:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(RES / f"sieve_{tag}.csv", index=False)
print("done")
