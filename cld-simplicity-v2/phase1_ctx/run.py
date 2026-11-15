"""Phase 1 (+ phase 4 third term): first-order axes x k-means-context sparsity, alone and combined with the frozen terms.

Candidates (all simplicity losses on rotated derivative energies, no roles):
  ctx(m,K,floor)                       the catalogue measure m on the (axes x contexts) first-order energy matrix
  ctx + UUd_share
  ctx + UX_neff + UUd_share            three-part loss
  UX_neff + UUd_share + UUoff(m)       phase 4: extra pure-structure term penalising off-diagonal UU interaction
python run.py SHARD NSHARDS REP [STARTS=8] [STEPS=150] [TAG=dev]  (candidate list = family below; confirmation: env CLD_CANDS="label;label")
"""
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import cldlib as L

torch.set_num_threads(2)
shard, nshards, rep = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
STARTS, STEPS = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (8, 150)
TAG = sys.argv[6] if len(sys.argv) > 6 else "dev"
RES = HERE / "results"
RES.mkdir(exist_ok=True)
CTX_MEASURES = ["neff", "l1l2", "gini", "renyi0.5", "lp0.5", "logsum0.1", "soft0.1", "support90", "rowneff", "colneff", "cf0.5", "cf1", "varimax", "axisneff", "axismaxshare", "tail2"]
cands = []
for m, K, fl in itertools.product(CTX_MEASURES, (4, 8, 16), (0.0, 2.0)):
    cands += [("ctx", m, K, fl), ("ctx+uu", m, K, fl), ("ctx+ux+uu", m, K, fl)]
for m in ("off_neff", "off_l1l2", "off_gini", "off_lp0.5", "off_soft0.1", "offshare"):
    cands.append(("ux+uu+off", m, 0, 0.0))
cands.append(("frozen", "-", 0, 0.0))


def label(c):
    return f"{c[0]}|{c[1]}|K{c[2]}|f{c[3]}"


if os.environ.get("CLD_CANDS"):
    keep = set(os.environ["CLD_CANDS"].split(";"))
    cands = [c for c in cands if label(c) in keep]
mine = [c for i, c in enumerate(cands) if i % nshards == shard]


def terms(c):
    kind, m, K, fl = c
    if kind == "frozen":
        return L.FROZEN, None, 0.0
    if kind == "ctx":
        return [L.T_ctx(m)], K, fl
    if kind == "ctx+uu":
        return [L.T_ctx(m), L.T_uu("diagshare")], K, fl
    if kind == "ctx+ux+uu":
        return [L.T_ctx(m), L.T_ux("neff"), L.T_uu("diagshare")], K, fl
    return L.FROZEN + [L.T_uu(m)], None, 0.0


rows = []
for model in L.MODELS:
    dev = L.Dev(model, rep)
    for c in mine:
        ts, K, fl = terms(c)
        t0 = time.time()
        loss, Q = L.search_terms(dev, ts, K, fl, STARTS, STEPS)
        if Q is None:
            rows.append(dict(model=model, rep=rep, cand=label(c), status="no_signal"))
            continue
        mcc, leak, lmax = L.X.evaluate(dev.st, Q)
        rows.append(dict(model=model, rep=rep, cand=label(c), kind=c[0], measure=c[1], K=c[2], floor=c[3], status="ok", mcc=mcc, leak=leak, leak_max=lmax,
                         oracle_angle_max=L.X.signed_perm_angle(Q, dev.st.oracle), loss=loss, secs=time.time() - t0))
    pd.DataFrame(rows).to_csv(RES / f"{TAG}_rep{rep}_shard{shard}of{nshards}.csv", index=False)
    print(f"shard {shard} rep{rep}: finished {model}", flush=True)
open(RES / f"{TAG}_rep{rep}_shard{shard}of{nshards}.COMPLETE", "w").write("ok")
