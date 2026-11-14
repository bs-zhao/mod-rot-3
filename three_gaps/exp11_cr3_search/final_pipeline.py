"""EXP11 final pipeline (W-Y simplicity, NN-decoder fields, no T in the search):
  1. contexts: CR3 bank -> design categories (loss cells) / mode-flag contexts (axis detection, W-only); continuous X -> Legendre cells, one context
  2. output-equivariant axis (stage A): CR3: MINIMAX over contexts of the relative equivariance residual;  C1-C3: pooled detection (common/equivariant.py)
  3. noise-floor-corrected cell energies  (floor = min_c lambda_min(E_c))
  4. loss = colLog3 on those energies, searched over SO(d) with the stage-A rows fixed (hard), multistart
T is used only to report TEST MCC / leakage.   python final_pipeline.py [--starts 12]
"""
import argparse, os, sys, json, time
from pathlib import Path
import numpy as np, torch, pandas as pd
os.environ.setdefault("GENERIC_CTX", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import harness as H, floor as FL, minimaxA as MA
from common.rotlib import bank_categories
torch.set_num_threads(4)
ap = argparse.ArgumentParser(); ap.add_argument("--starts", type=int, default=12); ap.add_argument("--loss", default="colLog3"); a = ap.parse_args()
OUT = H.OUT / "final"; OUT.mkdir(parents=True, exist_ok=True)
rows = []
for model in ["CR3", "C1", "C2", "C3"]:
    for rep in [1, 2]:
        for ch in ["logit", "prob"]:
            cond = FL.CONDS[model] + ("_rep02" if rep == 2 else "")
            c = H.load(model, cond, "decoder", ch); d = c["g"].shape[-1]
            R = np.load(H.io.ROOT / "outputs" / "three_gaps" / "exp00" / f"{model}_{cond}_pca0" / "R_oracle_raw.npy")
            t0 = time.time()
            if model == "CR3":
                mats = MA.ctx_matrices(c, ch, "gain")            # contexts = W-only mode-flag partition (GENERIC_CTX)
                rng = np.random.default_rng(0)
                e1, w1 = MA.minimax_axis(mats, d, rng); e2, _ = MA.minimax_axis(mats, d, rng, exclude=w1)
                F = w1[None] if (e1 < MA.ABS_T and e1 < MA.REL_T * e2) else np.zeros((0, d))
                c["cat"] = bank_categories(c["X"])              # loss cells = design categories
            else:
                F, _ = H.stageA(c, ch)
            f, lam_min, sig2, E = FL.make_loss(c, a.loss)
            fin = H.optimize(c, f, "SO", F=F, starts=a.starts, steps=250, seed=7)
            M = fin[0][1]; mcc, lk, lkm = H.evaluate(c, M); omcc, olk, olkm = H.evaluate(c, R)
            np.save(OUT / f"Q_{model}_rep{rep}_{ch}.npy", M)
            rows.append(dict(model=model, rep=rep, channel=ch, n_axes_fixed=len(F), mcc=mcc, leak=lk, leak_max=lkm,
                             oracle_mcc=omcc, oracle_leak=olk, oracle_leak_max=olkm, secs=time.time() - t0))
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
print("done")
