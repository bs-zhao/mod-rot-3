"""EXP23 runner: A/B/C multi-scale aggregation over smoothing scales (historical 300-subject outer measure, CR3).

python run.py MODEL REPEAT CHANNEL [N_IN=16]
Scale sets (preset, equal weights, not chosen with T): 'log' = {.05,.1,.25,.5,1}, 'mid' = {.1,.25,.5}.  B and C use the cached EXP22 per-scale moments; A computes the mixture-smoothed field.
Single-scale degeneracy checks and the sigma=0 baseline are reported.  T / Oracle only in the post-hoc scores.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import combine as CB
import x7core as X
import smooth_fields as SF

model, rep, ch = sys.argv[1], int(sys.argv[2]), sys.argv[3]
n_in = int(sys.argv[4]) if len(sys.argv) > 4 else 16
RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)
st = X.Setting(model, rep)
ids = st.cache_ids
Cfull = X.population_cov(st.V[st.tr])
CD = X.OUT / f"smooth_{model}_{st.cond}_{ch}"
SETS = {"log": [0.05, 0.1, 0.25, 0.5, 1.0], "mid": [0.1, 0.25, 0.5]}
rows = []


def load(s):
    z = np.load(CD / f"a0_cov_s{s}_n{n_in}.npz")
    return z["SX_s"].mean(0), z["T4_s"].mean(0)


def rec(name, agg, sset, obj, secs):
    fin = obj.search()
    Q = fin[0][1]
    mcc, leak, lmax = X.evaluate(st, Q)
    ang = X.signed_perm_angle(Q, st.oracle)
    rows.append(dict(model=model, rep=rep, channel=ch, name=name, agg=agg, scales=sset, mcc=mcc, leak=leak, leak_max=lmax, oracle_angle_max=ang, loss=fin[0][0], secs=secs))
    print(f"{name:14s} {agg:3s} {sset:4s} mcc {mcc:.4f} leak {leak:.4f} max {lmax:.3f} oracle-angle {ang:.1f}", flush=True)
    pd.DataFrame(rows).to_csv(RES / f"exp23_{model}_rep{rep}_{ch}.csv", index=False)


base = X.MomentLoss(*load(0.0), Cfull, st.d)
rec("J3_sigma0", "-", "-", base, 0)
for s in (0.25, 0.5):                                              # single-scale degeneracy: A = B = C = single-scale loss
    ml = X.MomentLoss(*load(s), Cfull, st.d)
    rec(f"single_{s}", "-", str(s), ml, 0)
    c1 = CB.LossAveraged([ml], [1.0])
    b1 = CB.energy_averaged([load(s)], [1.0])
    assert np.allclose(b1[0], load(s)[0]) and np.allclose(b1[1], load(s)[1])
for key, sg in SETS.items():
    moms = [load(s) for s in sg]
    rec(f"B_{key}", "B", key, X.MomentLoss(*CB.energy_averaged(moms, [1] * len(sg)), Cfull, st.d), 0)
    rec(f"C_{key}", "C", key, CB.LossAveraged([X.MomentLoss(*m, Cfull, st.d) for m in moms], [1] * len(sg)), 0)
    t0 = time.time()
    fA = CD / f"mix_a0_{key}_n{n_in}.npz"
    if not fA.exists():
        SXa, T4a = CB.mixture_subject_moments(st, ch, ids, sg, [1] * len(sg), "cov", Cfull, n_in)
        np.savez(fA, SX=SXa, T4=T4a)
    z = np.load(fA)
    rec(f"A_{key}", "A", key, X.MomentLoss(z["SX"].mean(0), z["T4"].mean(0), Cfull, st.d), time.time() - t0)
