"""EXP20A - empirical-distribution sample size: N -> loss landscape / recovery, with the FROZEN EXP13 loss.

Subsets of TRAIN subjects (nested, 5 preset seeds; full TRAIN once), A0 = the historical 300-subject cache grid.
Two axis-scale controls: A-native (C_ref of the subset, EXP13 semantics) and A-fixed (C_ref of the full TRAIN).
Reports TEST mcc / leakage / worst axis, relative moment error vs full TRAIN, landscape correlation on a common Haar set (1000 rotations,
seed 11), distance of the minimiser to the full-TRAIN minimiser.  T is used only for the post-hoc scores.
python run_20a.py MODEL REPEAT CHANNEL [NS=100,300,500,1000,2000] [SEEDS=5]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x7core as X
from common.rotlib import haar

RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)

model, rep, ch = sys.argv[1], int(sys.argv[2]), sys.argv[3]
NS = [int(v) for v in (sys.argv[4] if len(sys.argv) > 4 else "100,300,500,1000,2000").split(",")]
NSEED = int(sys.argv[5]) if len(sys.argv) > 5 else 5
st = X.Setting(model, rep)
mom = np.load(X.OUT / f"moments_{model}_{st.cond}_{ch}.npz")
SXs, T4s, V = mom["SX"], mom["T4"], mom["V"]
tr = st.tr
Ntr = len(tr)
rng_h = np.random.default_rng(11)
HAAR = [torch.tensor(haar(st.d, rng_h), dtype=torch.float32) for _ in range(1000)]


def agg(ids):
    return SXs[ids].mean(0), T4s[ids].mean(0)


SXf, T4f = agg(tr)
Cfull = X.population_cov(V[tr])
fullA = X.MomentLoss(SXf, T4f, Cfull, st.d)
with torch.no_grad():
    Lfull = np.array([float(fullA.total(M)) for M in HAAR])
fin_full = fullA.search()
Qfull = fin_full[0][1]
rows = []


def record(tag, N, seed, scale_mode, ids, ml, fin):
    Q = fin[0][1]
    mcc, leak, lmax = X.evaluate(st, Q)
    SXn, T4n = agg(ids)
    with torch.no_grad():
        L = np.array([float(ml.total(M)) for M in HAAR])
    rows.append(dict(model=model, rep=rep, channel=ch, tag=tag, N=N, seed=seed, scale=scale_mode, mcc=mcc, leak=leak, leak_max=lmax,
                     loss=fin[0][0], err_SX=float(np.linalg.norm(SXn - SXf) / np.linalg.norm(SXf)), err_T4=float(np.linalg.norm(T4n - T4f) / np.linalg.norm(T4f)),
                     landscape_pearson=float(pearsonr(L, Lfull)[0]), landscape_spearman=float(spearmanr(L, Lfull)[0]),
                     angle_to_full_deg=X.signed_perm_angle(Q, Qfull), sUX=ml.s["UX_neff"], sUU=ml.s["UUd_share"],
                     oracle_angle_deg=X.signed_perm_angle(Q, st.oracle)))
    r = rows[-1]
    print(f"{tag:6s} N={N:5d} seed={seed} {scale_mode:6s} mcc {mcc:.4f} leak {leak:.4f} | errSX {r['err_SX']:.3f} spearman {r['landscape_spearman']:.3f} "
          f"dist-to-full {r['angle_to_full_deg']:.1f}deg", flush=True)
    pd.DataFrame(rows).to_csv(RES / f"exp20a_{model}_rep{rep}_{ch}.csv", index=False)


t0 = time.time()
# full TRAIN (once, native == fixed)
record("full", Ntr, 0, "native", tr, fullA, fin_full)
# historical 300 grid (A0), both controls, and fixed-s sensitivity
ids0 = st.cache_ids
ml0 = X.MomentLoss(*agg(ids0), X.population_cov(V[ids0]), st.d)
record("A0", len(ids0), 0, "native", ids0, ml0, ml0.search())
ml0f = X.MomentLoss(*agg(ids0), Cfull, st.d)
record("A0", len(ids0), 0, "fixed", ids0, ml0f, ml0f.search())
mlh = X.MomentLoss(*agg(tr), Cfull, st.d, scales=ml0.s)
record("full_hs", Ntr, 0, "hist_s", tr, mlh, mlh.search())          # full TRAIN, historical (A0) Haar scales
# transductive all-N (explicitly labelled: TEST subjects take part in the measure)
allids = np.arange(len(V))
mla = X.MomentLoss(*agg(allids), X.population_cov(V), st.d)
record("all_trans", len(V), 0, "native", allids, mla, mla.search())
for N in NS:
    for seed in range(NSEED):
        ids = np.random.default_rng(1000 + seed).permutation(tr)[:N]
        for mode, C in (("native", X.population_cov(V[ids])), ("fixed", Cfull)):
            ml = X.MomentLoss(*agg(ids), C, st.d)
            record("sub", N, seed, mode, ids, ml, ml.search())
json.dump(dict(model=model, rep=rep, channel=ch, Ntrain=int(Ntr), secs=time.time() - t0), open(RES / f"exp20a_{model}_rep{rep}_{ch}.COMPLETE.json", "w"))
print("done", time.time() - t0)
