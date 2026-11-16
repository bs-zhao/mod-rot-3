"""Step 3a - Gap 3 on NN decoders: cross-moments between independently trained decoders instead of squares of one decoder.

Uses the 10 decoders that x6_plan06/e06_decoder_ensemble trained on the FROZEN latents with the observed choices (different seeds,
same TRAIN subjects) and their cached derivative fields (prob channel, same 300 x K grid as the EXP13 cache).  Read-only.
Caveat: these decoders share the training data, so only initialisation / optimisation noise is independent between them; the part
of the error that comes from the finite sample of choices is common and does NOT cancel in a cross-moment.

Fields / moments
  orig        the original NP decoder, plain squares                      (= frozen EXP13; must reproduce its numbers)
  ensN_mean   square of the mean field of N decoders                      (what e06 did)
  ensN_plain  mean of squares of N decoders
  ensN_cross  mean over pairs a != b of products                          (noise-floor free w.r.t. seed noise)
Losses: frozen EXP13 (UX_neff + UUd_share) and the item-level support losses, with items pooled into W-only contexts.

python cld-crossfit-v3/step3_gap3/run_decoders.py [MODELS=CR3] [REPS=1,2] [STARTS=24] [TAG=v1]
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
starts = int(sys.argv[3]) if len(sys.argv) > 3 else 24
tag = sys.argv[4] if len(sys.argv) > 4 else "v1"
RES = HERE / "results"; RES.mkdir(exist_ok=True)

COMBOS = [("UX_neff", "UUd_share"), ("UUd_share",), ("UX_neff",),
          ("UUI_softL0_03",), ("UUI_softL0_10",), ("UUI_softL0_30",), ("GI_softL0_10",), ("TOT_softL0_10",),
          ("UUI_softL0_10", "GI_softL0_10"), ("UUI_softL0_10", "UUd_share"), ("UUI_softL0_30", "UUd_share"),
          ("UX_neff", "UUI_softL0_10"), ("UX_neff", "UUd_share", "UUI_softL0_10")]
rows = []
for model in models:
    for rep in reps:
        c, cond, R = X.load(model, rep, "prob")
        data, truth = io.load_all(model, *XF.BETA[model], 0, rep, need_checkpoint=False)
        V, Tst, te = data["V"], truth.std(), data["test_ids"]
        t_grid = torch.tensor(V[c["ids"]], dtype=torch.float64)
        Xn = c["X"].numpy(); K = Xn.shape[0]
        ctxs = {"item": np.arange(K)}
        if len(np.unique(Xn[:, 0])) < K // 4:                              # shared bank with repeated column values: W-only mode-flag contexts
            ctxs.update(mode=X.H.mode_contexts(Xn), pure=X.H.pure_cells(Xn))
        F = torch.load(X.ROOT / "outputs" / "x6_plan06" / "e06" / f"{model}_{data['condition']}_base" / "fields_prob.pt")
        Kd = F["g"].shape[0]
        fl = [dict(g=F["g"][k], Htt=F["Htt"][k], Htx=F["Htx"][k]) for k in range(Kd)]
        mean = lambda n: [dict(g=F["g"][:n].mean(0), Htt=F["Htt"][:n].mean(0), Htx=F["Htx"][:n].mean(0))]
        SRC = {"orig": ([dict(g=c["g"], Htt=c["Htt"], Htx=c["Htx"])], False),
               f"ens{Kd}_mean": (mean(Kd), False), f"ens{Kd}_plain": (fl, False), f"ens{Kd}_cross": (fl, True),
               "ens2_cross": (fl[:2], True), "ens5_cross": (fl[:5], True)}
        print(f"== {model} rep{rep}: oracle %.3f/%.3f/%.3f" % XF.evaluate(V, Tst, te, R), flush=True)
        for sname, (fields, cross) in SRC.items():
            m0 = XF.moments(fields, cross=cross)
            for cname, lab in ctxs.items():
                t0 = time.time(); m = XF.pool_items(m0, lab)
                out = XF.run_losses(m, t_grid, COMBOS if cname == "item" else [x for x in COMBOS if any("I_" in k for k in x)],
                                    V, Tst, te, M_ref=R, starts=starts)
                for row, M in out:
                    row.update(model=model, rep=rep, source=sname, ctx=cname); rows.append(row)
                print(f"  {sname:11s} {cname:5s} " + " | ".join(f"{r['loss'].replace('_softL0_', '')} {r['mcc']:.3f}/{r['leak']:.3f}/{r['leak_max']:.2f}" for r, _ in out) +
                      f"  ({time.time()-t0:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(RES / f"decoders_{tag}.csv", index=False)
print("done")
