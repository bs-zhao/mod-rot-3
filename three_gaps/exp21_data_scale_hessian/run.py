"""EXP21 runner: data-scale finite-difference UX/UU vs a MATCHED autograd baseline (same fixed 64-subject x 64-stimulus centres, same budget).

python run.py MODEL REPEAT [STARTS=8] [STEPS=120]
Configs: autograd (matched) | hybrid (U finite, X autograd) h_U in {0.1,0.25,0.5} | ambient (U and X finite) h_U=.25 with c = h_X/SD(X_j) in {.1,.25,.5}.
Searches are blind (no T / Oracle); scores are post hoc.  Reports MCC, leakage (mean/max), axis angle to Oracle (diagnostic), landscape Spearman vs the matched
autograd loss on 100 common Haar rotations, support coverage of the ambient X stencil, wall time.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stencils as S
import x7core as X
from common.rotlib import haar

model, rep = sys.argv[1], int(sys.argv[2])
starts = int(sys.argv[3]) if len(sys.argv) > 3 else 8
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 120
RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)
st = X.Setting(model, rep)
fd = S.FD(st, "prob")
Cfull = X.population_cov(st.V[st.tr])
rows = []
rng = np.random.default_rng(21)
HAAR = [torch.tensor(haar(st.d, rng), dtype=torch.float32) for _ in range(100)]


def mean_axis_angle(Q, R):
    from scipy.optimize import linear_sum_assignment
    C = np.abs(Q @ R.T); r, c = linear_sum_assignment(-C)
    a = np.degrees(np.arccos(np.clip(C[r, c], 0, 1)))
    return float(a.max()), float(a.mean())


# matched autograd baseline on the SAME centres
SXs, T4s = X.centre_moments(st, "prob", st.V[fd.sub], X=st.X[fd.xi])
ml = X.MomentLoss(SXs.mean(0), T4s.mean(0), Cfull, st.d)
t0 = time.time()
fin = ml.search(starts=starts, steps=steps)
t_auto = time.time() - t0
with torch.no_grad():
    Lauto = np.array([float(ml.total(M)) for M in HAAR])


def add(name, Q, secs, loss, extra):
    mcc, leak, lmax = X.evaluate(st, Q)
    amax, amean = mean_axis_angle(Q, st.oracle)
    r = dict(model=model, rep=rep, config=name, mcc=mcc, leak=leak, leak_max=lmax, oracle_angle_max=amax, oracle_angle_mean=amean, loss=loss, secs=secs, starts=starts, steps=steps, **extra)
    rows.append(r)
    print(f"{name:28s} mcc {mcc:.4f} leak {leak:.4f} max {lmax:.3f} | oracle angle max/mean {amax:.1f}/{amean:.1f} | spearman-vs-autograd {extra.get('spearman_vs_autograd', float('nan')):.3f} | {secs:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(RES / f"exp21_{model}_rep{rep}.csv", index=False)


add("autograd_matched", fin[0][1], t_auto, fin[0][0], dict(spearman_vs_autograd=1.0, coverage_mean=np.nan, coverage_min=np.nan))
for mode, hU, c in [("hybrid", 0.1, None), ("hybrid", 0.25, None), ("hybrid", 0.5, None), ("ambient", 0.25, 0.1), ("ambient", 0.25, 0.25), ("ambient", 0.25, 0.5)]:
    fl = S.FDLoss(fd, hU, mode, c)
    t0 = time.time()
    f = fl.search(starts=starts, steps=steps)
    secs = time.time() - t0
    with torch.no_grad():
        L = np.array([float(fl.total(M.to(X.DEV))) for M in HAAR])
    cov = fd.support_audit(c) if mode == "ambient" else (np.nan, np.nan)
    name = f"{mode}_hU{hU}" + (f"_c{c}" if c else "")
    add(name, f[0][1], secs, f[0][0], dict(spearman_vs_autograd=float(spearmanr(L, Lauto)[0]), coverage_mean=cov[0], coverage_min=cov[1], hU=hU, hXc=c if c else np.nan))
json.dump(dict(model=model, rep=rep), open(RES / f"exp21_{model}_rep{rep}.COMPLETE.json", "w"))
