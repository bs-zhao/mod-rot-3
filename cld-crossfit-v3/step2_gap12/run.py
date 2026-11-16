"""Step 2 - Gap 1 and Gap 2 for the cross-fitted item-level UX / UU sparsity losses (CR3).

Coordinates: W = Tstd Q0^T, a fixed Haar mixture of the standardised true parameters (so the truth is the rotation Q0^T and the
search cannot benefit from starting axis-aligned).  T is used to BUILD this benchmark (that is what Gap 1 / Gap 2 are) and for the
post-hoc report; the fits and losses only receive (W, trial id, Y).

Sources
  analytic        exact derivatives of the generating formula                        Gap 1   (logit, prob)
  trunc:<fit>     per-item polynomial fitted to the NOISE-FREE true probabilities    truncation error only
  wy:<fit>:Kf     per-item polynomial fitted to the OBSERVED binary choices          Gap 2
                  Kf = number of disjoint subject folds; 'x' = cross-fitted energies, 'p' = plain squares (noise floor kept)
<fit> = logit2 / logit3 (per-item logistic polynomial of total degree 2 / 3 in U) or lpm2 / lpm3 (linear-probability polynomial).

Pass criterion (EXP07): worst angle to the true axes < 5 deg.

python cld-crossfit-v3/step2_gap12/run.py [SOURCES=all] [STARTS=24] [TAG=v1]
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

want = sys.argv[1] if len(sys.argv) > 1 else "all"
starts = int(sys.argv[2]) if len(sys.argv) > 2 else 24
tag = sys.argv[3] if len(sys.argv) > 3 else "v1"
RES = HERE / "results"; RES.mkdir(exist_ok=True)
dev = io.pick_device("auto")
model = "CR3"

data, truth = io.load_all(model, *XF.BETA[model], 0, 1, need_checkpoint=False)
c, cond, _ = X.load(model, 1, "prob")                                       # only for the 300 grid subject ids and the stimulus bank
ids, Xb, tr, te = c["ids"], c["X"].double(), data["train_ids"], data["test_ids"]
Tst = truth.std(); d = Tst.shape[1]; K = Xb.shape[0]
Q0 = XF.haar(d, np.random.default_rng(20261115)); W = Tst @ Q0.T; M_true = Q0.T
Y = data["y"].reshape(-1, K).astype(np.float64); P = data["predictive_true"].reshape(-1, K)
t_grid = torch.tensor(W[ids], dtype=torch.float64)
Qt = torch.tensor(Q0, dtype=torch.float64, device=dev)

COMBOS = [("UX_neff", "UUd_share"), ("UUd_share",), ("UI_neff",), ("UI_neff", "UUd_share"),
          ("UUI_softL0_03",), ("UUI_softL0_10",), ("UUI_softL0_30",),
          ("GI_softL0_10",), ("TOT_softL0_03",), ("TOT_softL0_10",), ("TOT_softL0_30",),
          ("UUI_softL0_10", "UI_neff"), ("UUI_softL0_10", "GI_softL0_10"), ("UUI_softL0_10", "UUd_share")]

rows = []
def report(source, ch, m, extra=None):
    t0 = time.time()
    out = XF.run_losses(m, t_grid, COMBOS, W, Tst, te, M_ref=M_true, starts=starts)
    for row, M in out:
        row.update(source=source, channel=ch, ang_true=XF.axis_error(M @ Q0)[0]); row.update(extra or {}); rows.append(row)
    best = min(out, key=lambda z: z[0]["ang_true"])[0]
    print(f"{source:22s} {ch:5s} " + " | ".join(f"{r['loss']}: {r['ang_true']:.1f}deg {r['mcc']:.3f}/{r['leak']:.3f}" for r, _ in out[:4]) +
          f" || best {best['loss']} {best['ang_true']:.1f}deg ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(RES / f"gap12_{tag}.csv", index=False)


def on(name):
    return want == "all" or any(name.startswith(w) for w in want.split(","))


# ---------------------------------------------------------------- Gap 1: exact analytic derivatives in the mixed coordinates
if on("analytic"):
    for ch in ("prob", "logit"):
        h0 = make_h(model, ch, truth.mu, truth.sd)
        h = lambda w, x: h0(Qt.T @ w, x)                                    # t = Q0^T w
        b = base_derivatives(h, t_grid.to(dev), Xb.to(dev), which=("g", "Htt", "Htx"), chunk=2048, out_dtype=torch.float64)
        report("analytic", ch, XF.moments([{k: v.cpu() for k, v in b.items()}], cross=False))

# ---------------------------------------------------------------- polynomial fits
FITS = {"logit2": (2, "logit"), "logit3": (3, "logit"), "lpm2": (2, "identity"), "lpm3": (3, "identity")}
for fit, (du, link) in FITS.items():
    chans = ("prob", "logit") if link == "logit" else ("prob",)
    if on("trunc"):                                                        # noise-free targets, all TRAIN subjects, one fit
        F, _ = XF.sieve_bank_fields(W, P, [tr], W[ids], du=du, link=link, ridge=0.0 if link == "logit" else 1e-10, dev=dev)
        for ch in chans:
            report(f"trunc:{fit}", ch, XF.moments([F[0][ch]], cross=False))
    if on("wy"):
        for Kf in (2, 5):
            folds = XF.split_folds(tr, Kf, seed=20261115)
            F, _ = XF.sieve_bank_fields(W, Y, folds, W[ids], du=du, link=link, dev=dev)
            for ch in chans:
                fl = [f[ch] for f in F]
                report(f"wy:{fit}:{Kf}x", ch, XF.moments(fl, cross=True), dict(folds=Kf, cross=1))
                if Kf == 2:
                    report(f"wy:{fit}:{Kf}p", ch, XF.moments(fl, cross=False), dict(folds=Kf, cross=0))
        F, _ = XF.sieve_bank_fields(W, Y, [tr], W[ids], du=du, link=link, dev=dev)          # one fit on all TRAIN subjects, plain squares
        for ch in chans:
            report(f"wy:{fit}:1p", ch, XF.moments([F[0][ch]], cross=False), dict(folds=1, cross=0))
print("done")
