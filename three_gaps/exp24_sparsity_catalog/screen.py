"""EXP24 screening: replace the UX part and/or the UU part of the frozen EXP13 loss by catalogue measures; singles and equal-weight combinations.

Frozen semantics kept: probability channel, historical 300-subject moments (A0, same C_ref), per-term Haar-200 (seed 1) standardisation, equal-weight sum,
Haar-initialised SO(d) Adam search.  Stage 'dev' uses a reduced budget (8 starts x 150 steps) on NP repeat 1; stage 'confirm' (confirm.py) re-runs the best
candidates with the frozen budget (24 x 250) on repeat 1 and on the held-out repeat 2.  T / Oracle are used only for post-hoc scores.

python screen.py SHARD NSHARDS [REP=1] [MODELS=CR3,C1,C2,C3]
"""
import json
import sys
import time
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "exp20_latent_integration"))
import catalog as C
import x7core as X
from common.rotlib import haar

torch.set_num_threads(2)
shard, nshards = int(sys.argv[1]), int(sys.argv[2])
rep = int(sys.argv[3]) if len(sys.argv) > 3 else 1
models = (sys.argv[4] if len(sys.argv) > 4 else "CR3,C1,C2,C3").split(",")
STARTS, STEPS = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (8, 150)
RES = HERE / "results"
RES.mkdir(exist_ok=True)
import os
TAG = os.environ.get("EXP24_TAG", "screen")
OUTCSV = RES / f"{TAG}_rep{rep}_shard{shard}of{nshards}.csv"

UXN, UUN, JN = list(C.UX), list(C.UU), list(C.JOINT)
CANDS = [("single_UX", u, None) for u in UXN] + [("single_UU", None, v) for v in UUN] + [("joint", j, None) for j in JN]
CANDS += [("combo", u, v) for u in UXN for v in UUN]
def label(c):
    k, a, b = c
    return {"combo": f"UX:{a} + UU:{b}", "single_UX": f"UX:{a} only", "single_UU": f"UU:{b} only", "joint": f"JOINT:{a}"}[k]


if os.environ.get("EXP24_CANDS"):                       # confirmation stage: an explicit, pre-listed candidate set
    keep = set(os.environ["EXP24_CANDS"].split(";"))
    CANDS = [c for c in CANDS if label(c) in keep]
MINE = [c for i, c in enumerate(CANDS) if i % nshards == shard]


def terms_of(kind, a, b):
    """list of (term_name) used by a candidate"""
    if kind == "single_UX": return [("UX", a)]
    if kind == "single_UU": return [("UU", b)]
    if kind == "joint": return [("J", a)]
    return [("UX", a), ("UU", b)]


def term_fn(t):
    k, n = t
    return {"UX": lambda ex, eu: C.UX[n](ex), "UU": lambda ex, eu: C.UU[n](eu), "J": lambda ex, eu: C.JOINT[n](ex, eu)}[k]


def run_setting(model):
    st = X.Setting(model, rep)
    mom = np.load(X.OUT / f"moments_{model}_{st.cond}_prob.npz")
    ids = st.cache_ids
    ml = X.MomentLoss(mom["SX"][ids].mean(0), mom["T4"][ids].mean(0), X.population_cov(st.V[ids]), st.d, scales={"UX_neff": 1.0, "UUd_share": 1.0})
    d = st.d
    rng = np.random.default_rng(1)
    HM = [torch.tensor(haar(d, rng), dtype=torch.float32) for _ in range(200)]
    needed = sorted({t for c in MINE for t in terms_of(*c)})
    vals = {t: [] for t in needed}
    with torch.no_grad():
        for M in HM:
            ex, eu = ml.energies(M)
            for t in needed:
                vals[t].append(float(term_fn(t)(ex, eu)))
    scale = {t: float(np.std(v)) for t, v in vals.items()}
    ident = X.evaluate(st, np.eye(d)); orac = X.evaluate(st, st.oracle)
    rows = []
    for kind, a, b in MINE:
        ts = terms_of(kind, a, b)
        if any(scale[t] < 1e-9 for t in ts):
            rows.append(dict(model=model, rep=rep, kind=kind, ux=a if kind != "single_UU" else "", uu=b if b else "", status="no_signal"))
            continue
        fns = [(term_fn(t), 1.0 / scale[t]) for t in ts]
        rngs = np.random.default_rng(7)
        best = (1e30, None)
        t0 = time.time()
        for _ in range(STARTS):
            Q0 = torch.tensor(haar(d, rngs), dtype=torch.float32)
            W = torch.zeros(d, d, requires_grad=True)
            opt = torch.optim.Adam([W], lr=0.05)
            ok = True
            for _ in range(STEPS):
                Q = torch.matrix_exp(W - W.T) @ Q0
                ex, eu = ml.energies(Q)
                L = sum(f(ex, eu) * s for f, s in fns)
                opt.zero_grad()
                L.backward()
                if not torch.isfinite(W.grad).all():
                    ok = False
                    break
                opt.step()
            with torch.no_grad():
                Q = torch.matrix_exp(W - W.T) @ Q0
                ex, eu = ml.energies(Q)
                L = float(sum(f(ex, eu) * s for f, s in fns))
            if np.isfinite(L) and L < best[0]:
                best = (L, Q.numpy())
        if best[1] is None:
            rows.append(dict(model=model, rep=rep, kind=kind, ux=a if kind != "single_UU" else "", uu=b if b else "", status="nan"))
            continue
        mcc, leak, lmax = X.evaluate(st, best[1])
        rows.append(dict(model=model, rep=rep, kind=kind, ux=a if kind != "single_UU" else "", uu=b if b else "", status="ok", mcc=mcc, leak=leak, leak_max=lmax,
                         oracle_angle_max=X.signed_perm_angle(best[1], st.oracle), loss=best[0], secs=time.time() - t0,
                         identity_mcc=ident[0], identity_leak=ident[1], oracle_mcc=orac[0], oracle_leak=orac[1]))
        if len(rows) % 20 == 0:
            pd.DataFrame(rows).to_csv(OUTCSV.with_suffix(f".{model}.partial.csv"), index=False)
    return rows


all_rows = []
for m in models:
    all_rows += run_setting(m)
    pd.DataFrame(all_rows).to_csv(OUTCSV, index=False)
    print(f"shard {shard}: finished {m} ({len(MINE)} candidates)", flush=True)
json.dump(dict(shard=shard, nshards=nshards, rep=rep, starts=STARTS, steps=STEPS, n_candidates=len(CANDS)), open(OUTCSV.with_suffix(".COMPLETE.json"), "w"))
