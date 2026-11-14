"""EXP13 screen: pure simplicity losses (no stage A) on cached NN-decoder second-order fields, SO(d) multistart, TEST eval.
python screen.py MODEL CHANNEL REPEAT [LOSS,...] [STARTS]"""
import sys, time, os, numpy as np, torch, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "exp11_cr3_search"))
import harness as H, losses2 as L2
from common import io
from common.rotlib import effective_A
torch.set_num_threads(4)
CONDS = {"CR3": "beta5em05_lambda0_ctx0p25to0p8_L4", "C1": "beta0p0005_lambda0_ctx0p25to0p8_L4",
         "C2": "beta0p0005_lambda0_ctx0p25to0p8_L4", "C3": "beta0p0005_lambda0_ctx0p25to0p8_L6"}
OUT = io.ROOT / "outputs" / "three_gaps" / "exp13"


def load(model, rep, ch):
    cond = CONDS[model] + ("_rep02" if rep == 2 else "")
    c = torch.load(OUT / f"cache2_{model}_{cond}_{ch}.pt", weights_only=False)
    c["cat"] = None
    return c, cond


def run(c, name, starts, F=None):
    SX, T4 = L2.moments(c)
    fn = L2.LOSSES[name]
    f = lambda b, A, M, cc: fn(*L2.energies(A, SX, T4))
    return H.optimize(c, f, "SO", F=F, starts=starts, steps=250, seed=7)


if __name__ == "__main__":
    model, ch, rep = sys.argv[1], sys.argv[2], int(sys.argv[3])
    names = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] != "all" else list(L2.LOSSES)
    starts = int(sys.argv[5]) if len(sys.argv) > 5 else 24
    c, cond = load(model, rep, ch); d = c["g"].shape[-1]
    R = np.load(io.ROOT / "outputs" / "three_gaps" / "exp00" / f"{model}_{cond}_pca0" / "R_oracle_raw.npy")
    print(f"== {model} {ch} rep{rep}: oracle TEST mcc/leak/max %.3f %.3f %.3f ; identity %.3f %.3f %.3f" % (H.evaluate(c, R) + H.evaluate(c, np.eye(d))), flush=True)
    rows = []
    for nm in names:
        t = time.time(); fin = run(c, nm, starts)
        m = H.evaluate(c, fin[0][1])
        rows.append(dict(model=model, channel=ch, rep=rep, loss=nm, mcc=m[0], leak=m[1], leak_max=m[2]))
        print(f"{nm:14s} TEST mcc/leak/max %.3f %.3f %.3f  ({time.time()-t:.0f}s)" % m, flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"screen_{model}_{ch}_rep{rep}.csv", index=False)
