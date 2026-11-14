"""EXP13 combos: equal-weight sums of Haar-spread-normalised generic simplicity terms (no role detection, weights fixed a priori).
python combos.py MODEL CHANNEL REPEAT"""
import sys, os, time, numpy as np, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "exp11_cr3_search"))
import harness as H, floor as FL, losses2 as L2, screen as SC
from common.rotlib import effective_A, haar
torch.set_num_threads(4)
model, ch, rep = sys.argv[1], sys.argv[2], int(sys.argv[3])
c, cond = SC.load(model, rep, ch); d = c["g"].shape[-1]
SX, T4 = L2.moments(c)
if model == "CR3":
    c["cat"] = torch.tensor(H.pure_cells(c["X"].numpy()), dtype=torch.long)     # W-only cells
Efl, _, sig2, _ = None, None, None, None
Ec, lam_min, sig2 = FL.correct(FL.cell_energies(c))[0], None, None
c["cat"] = None if model != "CR3" else c["cat"]
def term_cells(A):                                    # first-order U x (X-context) sparsity, floor-corrected colLog3
    e2 = torch.einsum("id,cde,ie->ic", A, Ec, A).clamp_min(0)
    return FL.LOSSES["colLog3"](e2.sqrt().unsqueeze(0))
TERMS = {"cells": term_cells,
         "UX_neff": lambda A: L2.LOSSES["UX_neff"](*L2.energies(A, SX, T4)),
         "UUd_share": lambda A: L2.LOSSES["UUd_share"](*L2.energies(A, SX, T4)),
         "UUd_sum": lambda A: L2.LOSSES["UUd_sum"](*L2.energies(A, SX, T4)),
         "UUd_l1l2": lambda A: L2.LOSSES["UUd_l1l2"](*L2.energies(A, SX, T4)),
         "UUd_max": lambda A: L2.LOSSES["UUd_max"](*L2.energies(A, SX, T4))}
t_dev = torch.tensor(c["V"][c["ids"]], dtype=torch.float32)
rng = np.random.default_rng(1); Ms = [torch.tensor(haar(d, rng), dtype=torch.float32) for _ in range(200)]
sc = {}
with torch.no_grad():
    for k, f in TERMS.items():
        v = np.array([float(f(effective_A(M, t_dev))) for M in Ms]); sc[k] = v.std() + 1e-9
R = np.load(SC.io.ROOT / "outputs" / "three_gaps" / "exp00" / f"{model}_{cond}_pca0" / "R_oracle_raw.npy")
print(f"== {model} {ch} rep{rep}: oracle %.3f/%.3f/%.3f ; floor sigma2 {float(FL.correct(FL.cell_energies(c))[0].abs().max()):.4f}" % H.evaluate(c, R), flush=True)
combos = [tuple(x.split("+")) for x in (sys.argv[4].split(";") if len(sys.argv) > 4 else ["UUd_share","UUd_sum","UUd_l1l2","UUd_max","cells","cells+UUd_share","cells+UUd_sum","UX_neff+UUd_share","UX_neff+UUd_sum","cells+UX_neff+UUd_share"])]
for names in combos:
    f = lambda b, A, M, cc, names=names: sum(TERMS[k](A) / sc[k] for k in names)
    t = time.time(); fin = H.optimize(c, f, "SO", F=None, starts=24, steps=250, seed=7)
    print(f"{'+'.join(names):28s} TEST mcc/leak/max %.3f %.3f %.3f ({time.time()-t:.0f}s)" % H.evaluate(c, fin[0][1]), flush=True)
