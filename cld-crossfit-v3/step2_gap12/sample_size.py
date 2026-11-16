"""Step 2d - how much choice data would the item-level UU support loss need?   (Gap 2 as a function of the amount of data)

Simulation in the mixed true coordinates: every TRAIN subject answers every stimulus R times (Y = Binomial(R, p_true) / R), so the
information per (subject, stimulus) cell grows by R.  R = 1 is the real design.  Per-item cubic linear-probability fits on two
disjoint halves; cross-fitted energies ("x") against plain squares of the same two fits ("p"); items as cells and W-only mode-flag
contexts.  Reported: worst angle to the true axes (pass < 5 deg), 3 simulation seeds.
python cld-crossfit-v3/step2_gap12/sample_size.py [RS=1,4,16,64,256] [SEEDS=3]"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "common"))
import xfit as XF
from xfit import X, io
torch.set_num_threads(4)
Rs = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,4,16,64,256").split(",")]
nseed = int(sys.argv[2]) if len(sys.argv) > 2 else 3
dev = io.pick_device("auto"); model = "CR3"
data, truth = io.load_all(model, *XF.BETA[model], 0, 1, need_checkpoint=False)
c, cond, _ = X.load(model, 1, "prob"); ids, Xn, tr, te = c["ids"], c["X"].numpy(), data["train_ids"], data["test_ids"]
Tst = truth.std(); K = Xn.shape[0]
Q0 = XF.haar(4, np.random.default_rng(20261115)); W = Tst @ Q0.T; M_true = Q0.T
P = data["predictive_true"].reshape(-1, K); t_grid = torch.tensor(W[ids], dtype=torch.float64)
CTX = {"item": np.arange(K), "mode": X.H.mode_contexts(Xn), "pure": X.H.pure_cells(Xn)}
COMBOS = [("UUI_softL0_10",), ("UUI_softL0_10", "UUd_share"), ("UUI_softL0_10", "GI_softL0_10"), ("UUd_share",)]
folds = XF.split_folds(tr, 2, 20261115); rows = []
for R in Rs:
    for seed in range(nseed):
        rng = np.random.default_rng(1000 * R + seed); Y = rng.binomial(R, P) / R; t0 = time.time()
        F = [f["prob"] for f in XF.sieve_bank_fields(W, Y, folds, W[ids], du=3, link="identity", dev=dev)[0]]
        for cross in (True, False):
            m0 = XF.moments(F, cross=cross)
            for cname, lab in CTX.items():
                for row, M in XF.run_losses(XF.pool_items(m0, lab), t_grid, COMBOS, W, Tst, te, M_ref=M_true):
                    row.update(R=R, seed=seed, cross=int(cross), ctx=cname, ang_true=XF.axis_error(M @ Q0)[0]); rows.append(row)
        d = pd.DataFrame(rows); d.to_csv(HERE / "results" / "sample_size.csv", index=False)
        s = d[(d.R == R) & (d.seed == seed)]
        print(f"R={R:3d} seed{seed} ({time.time()-t0:.0f}s)  " + "  ".join(
            f"[{'x' if cr else 'p'} {cn}] " + " ".join(f"{a:.0f}" for a in s[(s.cross == cr) & (s.ctx == cn)].ang_true) for cr in (1, 0) for cn in CTX), flush=True)
d = pd.DataFrame(rows)
print(d.groupby(["R", "cross", "ctx", "loss"]).ang_true.median().unstack("R").round(1).to_string())
