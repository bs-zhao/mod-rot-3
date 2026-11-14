"""EXP11: one objective = equal-weight sum of Haar-normalised W-Y simplicity losses over channels (logit, prob) and losses.
python ensemble.py MODEL REPEAT LOSS[,LOSS] [STARTS]   (hard stage A from the logit field)"""
import sys, time, numpy as np, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import harness as H
from common.losses import LOSSES
from common.rotlib import apply_view, effective_A, haar
torch.set_num_threads(4)
CONDS = {"CR3": "beta5em05_lambda0_ctx0p25to0p8_L4", "C1": "beta0p0005_lambda0_ctx0p25to0p8_L4",
         "C2": "beta0p0005_lambda0_ctx0p25to0p8_L4", "C3": "beta0p0005_lambda0_ctx0p25to0p8_L6"}
model, rep, losses = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
starts = int(sys.argv[4]) if len(sys.argv) > 4 else 12
cond = CONDS[model] + ("_rep02" if rep == 2 else "")
cs = {ch: H.load(model, cond, "decoder", ch) for ch in ["logit", "prob"]}
assert np.array_equal(cs["logit"]["ids"], cs["prob"]["ids"])
c = cs["logit"]; d = c["g"].shape[-1]
F, _ = H.stageA(c, "logit")
bases = {ch: H.base_of(cs[ch]) for ch in cs}
t_dev = torch.tensor(c["V"][c["ids"]], dtype=torch.float32)
rng = np.random.default_rng(1); Ms = [torch.tensor(haar(d, rng), dtype=torch.float32) for _ in range(200)]
sc = {}
with torch.no_grad():
    for ch in bases:
        for l in losses:
            sc[(ch, l)] = np.std([float(LOSSES[l](apply_view(bases[ch], "GCat", effective_A(M, t_dev)))) for M in Ms])
def f(_b, A, M, _c):
    return sum(LOSSES[l](apply_view(bases[ch], "GCat", A)) / sc[(ch, l)] for ch in bases for l in losses)
R = np.load(H.io.ROOT / "outputs" / "three_gaps" / "exp00" / f"{model}_{cond}_pca0" / "R_oracle_raw.npy")
t = time.time(); fin = H.optimize(c, f, "SO", F=F, starts=starts, steps=300, seed=7)
print(f"{model} rep{rep} ensemble {losses} x [logit,prob]: oracle %.3f/%.3f/%.3f | found TEST mcc/leak/max %.3f %.3f %.3f (%.0fs)" %
      (H.evaluate(c, R) + H.evaluate(c, fin[0][1]) + (time.time() - t,)), flush=True)
