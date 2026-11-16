"""Step 3b - Gap 3 with decoders trained on DISJOINT halves of the TRAIN subjects (train_halves.py).

(1) How much of a decoder's derivative field is reproducible?  For g, H_UX, H_UU (prob and logit channels):
      seed share   = <F_seed_a, F_seed_b> / mean |F|^2      two seeds, SAME half      (removes optimisation noise only)
      half share   = <F_half0, F_half1> / mean |F|^2        seed-averaged fields of the two DISJOINT halves (removes sampling noise too)
    A share of 1 means fully reproducible, 0 means pure noise.  T-free.
(2) Rotation with cross-moments between the two halves (fold field = mean over that half's seeds), against plain squares.

python cld-crossfit-v3/step3_gap3/run_halves.py [MODELS=CR3] [REPS=1,2] [CHANNELS=prob,logit] [STARTS=24] [TAG=v1]
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "common"))
import xfit as XF
from xfit import X, io
torch.set_num_threads(6)

models = (sys.argv[1] if len(sys.argv) > 1 else "CR3").split(",")
reps = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,2").split(",")]
channels = (sys.argv[3] if len(sys.argv) > 3 else "prob,logit").split(",")
starts = int(sys.argv[4]) if len(sys.argv) > 4 else 24
tag = sys.argv[5] if len(sys.argv) > 5 else "v1"
RES = HERE / "results"; RES.mkdir(exist_ok=True)

COMBOS = [("UX_neff", "UUd_share"), ("UUd_share",), ("UX_neff",),
          ("UUI_softL0_03",), ("UUI_softL0_10",), ("UUI_softL0_30",), ("GI_softL0_10",),
          ("UUI_softL0_10", "GI_softL0_10"), ("UUI_softL0_10", "UUd_share"), ("UUI_softL0_30", "UUd_share"),
          ("UX_neff", "UUI_softL0_10"), ("UX_neff", "UUd_share", "UUI_softL0_10"), ("UX_neff", "UUd_share", "GI_softL0_10")]
rows, shares = [], []
for model in models:
    for rep in reps:
        data, truth = io.load_all(model, *XF.BETA[model], 0, rep, need_checkpoint=False)
        V, Tst, te = data["V"], truth.std(), data["test_ids"]
        for ch in channels:
            c, cond, R = X.load(model, rep, ch)
            t_grid = torch.tensor(V[c["ids"]], dtype=torch.float64)
            Xn = c["X"].numpy(); K = Xn.shape[0]
            ctxs = {"item": np.arange(K)}
            if len(np.unique(Xn[:, 0])) < K // 4:
                ctxs.update(mode=X.H.mode_contexts(Xn), pure=X.H.pure_cells(Xn))
            F = torch.load(XF.OUT / "dec" / f"{model}_{data['condition']}_h2" / f"fields_{ch}.pt")     # (halves, seeds, S, K, ...)
            nh, ns = F["g"].shape[:2]
            sh = dict(model=model, rep=rep, channel=ch)
            for nm in ("g", "Htx", "Htt"):
                A = F[nm].double(); tot = float((A ** 2).mean())
                sh[f"seed_{nm}"] = float(np.mean([float((A[h, a] * A[h, b]).mean()) for h in range(nh) for a in range(ns) for b in range(ns) if a != b])) / tot
                Hm = A.mean(1); sh[f"half_{nm}"] = float((Hm[0] * Hm[1]).mean()) / tot
                sh[f"orig_vs_half_{nm}"] = float((c[nm].double() * Hm.mean(0)).mean()) / float(((c[nm].double() ** 2).mean() * (Hm.mean(0) ** 2).mean()) ** 0.5)
            shares.append(sh)
            print(f"== {model} rep{rep} {ch}: reproducible share of one decoder's field energy  " +
                  " | ".join(f"{nm}: same half other seed {sh['seed_' + nm]:.2f}, other half {sh['half_' + nm]:.2f}, cos(orig, mean) {sh['orig_vs_half_' + nm]:.2f}" for nm in ("g", "Htx", "Htt")), flush=True)
            pd.DataFrame(shares).to_csv(RES / f"halves_shares_{tag}.csv", index=False)
            half = [dict(g=F["g"][h].mean(0), Htt=F["Htt"][h].mean(0), Htx=F["Htx"][h].mean(0)) for h in range(nh)]
            allm = [dict(g=F["g"].mean((0, 1)), Htt=F["Htt"].mean((0, 1)), Htx=F["Htx"].mean((0, 1)))]
            SRC = {"orig": ([dict(g=c["g"], Htt=c["Htt"], Htx=c["Htx"])], False), "halves_cross": (half, True),
                   "halves_plain": (half, False), "all_mean": (allm, False)}
            for sname, (fields, cross) in SRC.items():
                m0 = XF.moments(fields, cross=cross)
                for cname, lab in ctxs.items():
                    t0 = time.time(); m = XF.pool_items(m0, lab)
                    out = XF.run_losses(m, t_grid, COMBOS if cname == "item" else [x for x in COMBOS if any("I_" in k for k in x)],
                                        V, Tst, te, M_ref=R, starts=starts)
                    for row, M in out:
                        row.update(model=model, rep=rep, channel=ch, source=sname, ctx=cname); rows.append(row)
                    print(f"  {sname:12s} {cname:5s} " + " | ".join(f"{r['loss'].replace('_softL0_', '')} {r['mcc']:.3f}/{r['leak']:.3f}/{r['leak_max']:.2f}" for r, _ in out) +
                          f"  ({time.time()-t0:.0f}s)", flush=True)
                    pd.DataFrame(rows).to_csv(RES / f"halves_{tag}.csv", index=False)
print("done")
