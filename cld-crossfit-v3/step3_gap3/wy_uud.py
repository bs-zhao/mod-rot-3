"""Step 3d - confirmation of the one strict W-Y result that came close to the frozen baseline in run_sieve.py:
UUd_share computed from per-item quadratic LOGISTIC fits on the observed choices (no decoder anywhere), cross-fitted over subject folds.
It was picked after looking at TEST on repeats 1-2, so here it is run WITHOUT selection on every combination of
  NP repeat 1 / 2 / 3  x  folds 2 / 5 / 10  x  3 fold seeds  x  {cross-fitted, plain}
and every row is reported.  Grid subjects = the EXP13 rule (300 TRAIN subjects, seed 20261108).
python cld-crossfit-v3/step3_gap3/wy_uud.py [REPS=1,2,3]"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "common"))
import xfit as XF
from xfit import io
torch.set_num_threads(4)
reps = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,2,3").split(",")]
dev = io.pick_device("auto"); model = "CR3"; rows = []
for rep in reps:
    data, truth = io.load_all(model, *XF.BETA[model], 0, rep, need_checkpoint=False)
    V, Tst, tr, te = data["V"], truth.std(), data["train_ids"], data["test_ids"]
    ids = np.sort(np.random.default_rng(20261108).choice(tr, 300, replace=False))
    K = int((data["subject_index"] == 0).sum()); Y = data["y"].reshape(-1, K).astype(np.float64)
    t_grid = torch.tensor(V[ids], dtype=torch.float64)
    for du in (2, 3):
        for Kf in (2, 5, 10):
            for fs in range(3):
                F = [f["prob"] for f in XF.sieve_bank_fields(V, Y, XF.split_folds(tr, Kf, 100 + fs), V[ids], du=du, link="logit", dev=dev)[0]]
                for cross in (True, False):
                    m = XF.moments(F, cross=cross, per_item=False)
                    out = XF.run_losses(m, t_grid, [("UUd_share",)], V, Tst, te)
                    row = out[0][0]; row.update(rep=rep, du=du, folds=Kf, fold_seed=fs, cross=int(cross)); rows.append(row)
                pd.DataFrame(rows).to_csv(HERE / "results" / "wy_uud.csv", index=False)
        d = pd.DataFrame(rows); s = d[(d.rep == rep) & (d.du == du)]
        print(f"rep{rep} logit{du}\n" + s.groupby(["cross", "folds"])[["mcc", "leak", "leak_max"]].agg(["mean", "min", "max"]).round(3).to_string(), flush=True)
