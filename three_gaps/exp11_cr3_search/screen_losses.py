"""EXP11 screen: W-Y simplicity losses (view x loss) on NN decoder gradient fields, SO(d) + stage A.
T only in post-hoc TEST evaluation.  python screen_losses.py MODEL CHANNEL [REPEAT]"""
import sys, time, itertools, numpy as np, torch, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import harness as H
from common.losses import LOSSES
from common.rotlib import apply_view, effective_A
torch.set_num_threads(4)
CONDS = {"CR3": "beta5em05_lambda0_ctx0p25to0p8_L4", "C1": "beta0p0005_lambda0_ctx0p25to0p8_L4",
         "C2": "beta0p0005_lambda0_ctx0p25to0p8_L4", "C3": "beta0p0005_lambda0_ctx0p25to0p8_L6"}
model, ch = sys.argv[1], sys.argv[2]
rep = int(sys.argv[3]) if len(sys.argv) > 3 else 1
cond = CONDS[model] + ("_rep02" if rep == 2 else "")
c = H.load(model, cond, "decoder", ch)
F, _ = H.stageA(c, ch)
R = np.load(H.io.ROOT / "outputs" / "three_gaps" / "exp00" / f"{model}_{cond}_pca0" / "R_oracle_raw.npy")
base = H.base_of(c); t_dev = torch.tensor(c["V"][c["ids"]], dtype=torch.float32)
Mo = torch.tensor(R, dtype=torch.float32)
views = ["GCat", "GP"] if model == "CR3" else ["GCat"]
rows = []
for view, loss in itertools.product(views, ["absL1", "rmsL1", "colL1L2", "rowNeff", "cellNeff", "softL0", "quartimax",
                                            "softL0_03", "softL0_10", "softL0_30", "colLog1", "colLog2", "colLog3", "logSp"]):
    f = H.L_view(loss, view); t = time.time()
    fin = H.optimize(c, f, "SO", F=F, starts=6, steps=200)
    mcc, lk, lkm = H.evaluate(c, fin[0][1])
    with torch.no_grad():
        Lo = float(f(base, effective_A(Mo, t_dev), Mo, c))
    sc = max(abs(fin[0][0]), 1e-12)
    rows.append(dict(model=model, channel=ch, rep=rep, view=view, loss=loss, mcc=mcc, leak=lk, leak_max=lkm,
                     L_found=fin[0][0], L_oracle=Lo, oracle_gap=(Lo - fin[0][0]) / sc))
    print(f"{view:5s} {loss:10s} mcc {mcc:.3f} leak {lk:.3f}/{lkm:.2f} L_or-L_found {(Lo-fin[0][0])/sc:+.4f} ({time.time()-t:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(H.OUT / f"screen_{model}_{ch}_rep{rep}.csv", index=False)
