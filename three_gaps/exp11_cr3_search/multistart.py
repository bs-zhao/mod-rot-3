"""EXP11: does a better optimiser (many starts) fix losses whose value at the oracle is below the found optimum?
python multistart.py MODEL CHANNEL REPEAT STARTS LOSS[,LOSS...] [VIEW]"""
import sys, time, numpy as np, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import harness as H
from common.rotlib import effective_A
torch.set_num_threads(4)
CONDS = {"CR3": "beta5em05_lambda0_ctx0p25to0p8_L4", "C1": "beta0p0005_lambda0_ctx0p25to0p8_L4",
         "C2": "beta0p0005_lambda0_ctx0p25to0p8_L4", "C3": "beta0p0005_lambda0_ctx0p25to0p8_L6"}
model, ch, rep, starts, losses = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5].split(",")
view = sys.argv[6] if len(sys.argv) > 6 else "GCat"
cond = CONDS[model] + ("_rep02" if rep == 2 else "")
c = H.load(model, cond, "decoder", ch); F, _ = H.stageA(c, ch)
R = np.load(H.io.ROOT / "outputs" / "three_gaps" / "exp00" / f"{model}_{cond}_pca0" / "R_oracle_raw.npy")
base = H.base_of(c); t_dev = torch.tensor(c["V"][c["ids"]], dtype=torch.float32); Mo = torch.tensor(R, dtype=torch.float32)
for loss in losses:
    f = H.L_view(loss, view); t = time.time()
    fin = H.optimize(c, f, "SO", F=F, starts=starts, steps=300, seed=7)
    with torch.no_grad():
        Lo = float(f(base, effective_A(Mo, t_dev), Mo, c))
    sc = max(abs(fin[0][0]), 1e-12)
    m = [H.evaluate(c, q)[0] for _, q in fin]
    print(f"{model} {ch} rep{rep} {view} {loss:10s} starts {starts}: best-L TEST mcc/leak/max %.3f %.3f %.3f | L_oracle-L_best {(Lo-fin[0][0])/sc:+.4f} | "
          f"top5 mcc {np.round(m[:5],3)} | frac starts within 1e-3 of best {np.mean([abs(L-fin[0][0])<1e-3*sc for L,_ in fin]):.2f} ({time.time()-t:.0f}s)"
          % H.evaluate(c, fin[0][1]), flush=True)
