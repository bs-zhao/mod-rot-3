"""Phase 2: third-order UUX simplicity.  E_UUX[(i,k),j] = E[(d3 p / du_i du_k dx_j)^2] from per-point third derivatives of the frozen decoder (autograd).

stage cache : moments M3[j,a,b,c,d] = E[T_abj T_cdj] on the historical 300-subject x X-bank grid   (GPU)
stage run   : catalogue measures on the rotated third-order energy matrix, alone and combined with the frozen terms
python run.py cache MODEL REP | python run.py run SHARD NSHARDS REP [STARTS STEPS TAG]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import cldlib as L
from common.derivs import base_derivatives

RES = HERE / "results"
RES.mkdir(exist_ok=True)
mode = sys.argv[1]
if mode == "cache":
    model, rep = sys.argv[2], int(sys.argv[3])
    st = L.X.Setting(model, rep)
    c = torch.load(st.cache_path, weights_only=False)
    ids = c["ids"]
    h = st.h("prob")
    d, p = st.d, st.p
    M3 = np.zeros((p, d, d, d, d))
    t0 = time.time()
    n = 0
    for i in range(0, len(ids), 8):
        grid = torch.tensor(st.V[ids[i:i + 8]], dtype=torch.float64, device=L.X.DEV)
        T = base_derivatives(h, grid, st.X.to(L.X.DEV), which=("Tttx",), chunk=1024)["Tttx"].double().cpu()   # (B,K,d,d,p)
        B, K = T.shape[:2]
        Tf = T.reshape(B * K, d * d, p)
        M3 += torch.einsum("nmj,nqj->jmq", Tf, Tf).reshape(p, d, d, d, d).numpy()
        n += B * K
    np.savez(L.OUT / f"M3_{model}_rep{rep}.npz", M3=M3 / n)
    print(f"cached third-order moments {model} rep{rep}: {n} points in {time.time() - t0:.0f}s")
    sys.exit(0)

shard, nshards, rep = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
STARTS, STEPS = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (8, 150)
TAG = sys.argv[7] if len(sys.argv) > 7 else "dev"
torch.set_num_threads(2)


class Dev3(L.Dev):
    def __init__(self, model, rep):
        super().__init__(model, rep)
        self.M3 = torch.tensor(np.load(L.OUT / f"M3_{model}_rep{rep}.npz")["M3"], dtype=torch.float32)

    def energies(self, M, K=None, floor=0.0):
        A = L.X.eff_A(M, self.ml.C)
        ex, eu = L.L2.energies(A, self.ml.SX, self.ml.T4)
        e3 = torch.einsum("ia,kb,jabcd,ic,kd->ikj", A, A, self.M3, A, A).clamp_min(0)      # (d,d,p)
        return ex, eu, e3.reshape(self.d * self.d, -1)


MEAS = ["neff", "l1l2", "gini", "renyi0.5", "lp0.5", "logsum0.1", "soft0.1", "support90", "rowneff", "colneff", "cf0.5", "varimax", "tail2"]
cands = [("uux", m) for m in MEAS] + [("uux+uu", m) for m in MEAS] + [("frozen+uux", m) for m in MEAS] + [("frozen", "-")]
mine = [c for i, c in enumerate(cands) if i % nshards == shard]
rows = []
for model in L.MODELS:
    dev = Dev3(model, rep)
    for kind, m in mine:
        ts = {"uux": [L.T_ctx(m)], "uux+uu": [L.T_ctx(m), L.T_uu("diagshare")], "frozen+uux": L.FROZEN + [L.T_ctx(m)], "frozen": L.FROZEN}[kind]
        t0 = time.time()
        loss, Q = L.search_terms(dev, ts, None if kind == "frozen" else 1, 0.0, STARTS, STEPS)
        if Q is None:
            rows.append(dict(model=model, rep=rep, cand=f"{kind}|{m}", status="no_signal"))
            continue
        mcc, leak, lmax = L.X.evaluate(dev.st, Q)
        rows.append(dict(model=model, rep=rep, cand=f"{kind}|{m}", status="ok", mcc=mcc, leak=leak, leak_max=lmax, oracle_angle_max=L.X.signed_perm_angle(Q, dev.st.oracle),
                         loss=loss, secs=time.time() - t0))
    pd.DataFrame(rows).to_csv(RES / f"{TAG}_rep{rep}_shard{shard}of{nshards}.csv", index=False)
    print(f"shard {shard}: finished {model}", flush=True)
open(RES / f"{TAG}_rep{rep}_shard{shard}of{nshards}.COMPLETE", "w").write("ok")
