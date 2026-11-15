"""EXP22 runner: smoothed Hessian sigma grid for the frozen EXP13 loss.

python run.py MODEL REPEAT CHANNEL cache SUBJ [N_IN=16] [SIGMAS=0,0.05,0.1,0.25,0.5,1.0] [KERNEL=cov]   # GPU: per-subject smoothed moments
python run.py MODEL REPEAT CHANNEL eval  SUBJ [N_IN=16] [KERNEL=cov]                                      # CPU: searches + report
SUBJ = a0 (historical 300 grid subjects) | train (all TRAIN subjects).  C_ref is the FULL-TRAIN covariance (fixed kernel / fixed axis scale).
smooth-then-square (primary) vs square-then-average (enlarged measure) vs sigma = 0 (autograd).  T only in the post-hoc scores.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp20_latent_integration"))
import smooth_fields as SF
import x7core as X
from common.rotlib import haar

model, rep, ch, stage, subj = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
n_in = int(sys.argv[6]) if len(sys.argv) > 6 else 16
SIG = [float(v) for v in (sys.argv[7] if len(sys.argv) > 7 else "0,0.05,0.1,0.25,0.5,1.0").split(",")]
kernel = sys.argv[8] if len(sys.argv) > 8 else "cov"
RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)
st = X.Setting(model, rep)
ids = st.cache_ids if subj == "a0" else st.tr
Cfull = X.population_cov(st.V[st.tr])
CDIR = X.OUT / f"smooth_{model}_{st.cond}_{ch}"
CDIR.mkdir(exist_ok=True)


def f_cache(sigma):
    return CDIR / f"{subj}_{kernel}_s{sigma}_n{n_in}.npz"


def stage_cache():
    for s in SIG:
        f = f_cache(s)
        if f.exists():
            continue
        t0 = time.time()
        o = SF.smoothed_subject_moments(st, ch, ids, s, kernel, Cfull, n_in)
        np.savez(f.with_suffix(".tmp.npz"), **o)
        f.with_suffix(".tmp.npz").replace(f)
        print(f"{subj} {kernel} sigma={s} n_in={n_in}: {len(ids)} subjects in {time.time() - t0:.0f}s", flush=True)


def stage_eval():
    rngh = np.random.default_rng(11)
    HAAR = [torch.tensor(haar(st.d, rngh), dtype=torch.float32) for _ in range(1000)]
    z0 = np.load(f_cache(0.0))
    ml0 = X.MomentLoss(z0["SX_s"].mean(0), z0["T4_s"].mean(0), Cfull, st.d)
    with torch.no_grad():
        L0 = np.array([float(ml0.total(M)) for M in HAAR])
    fin0 = ml0.search()
    Q0 = fin0[0][1]
    rows = []
    p0 = z0["p0"]
    for s in SIG:
        z = np.load(f_cache(s))
        for agg in ("smooth_then_square", "square_then_average"):
            if s == 0 and agg == "square_then_average":
                continue
            SX, T4 = (z["SX_s"], z["T4_s"]) if agg == "smooth_then_square" else (z["SX_e"], z["T4_e"])
            ml = X.MomentLoss(SX.mean(0), T4.mean(0), Cfull, st.d)
            fin = ml.search()
            Q = fin[0][1]
            mcc, leak, lmax = X.evaluate(st, Q)
            with torch.no_grad():
                L = np.array([float(ml.total(M)) for M in HAAR])
            rows.append(dict(model=model, rep=rep, channel=ch, subjects=subj, kernel=kernel, n_in=n_in, sigma=s, agg=agg, mcc=mcc, leak=leak, leak_max=lmax,
                             loss=fin[0][0], spearman_vs_sigma0=float(spearmanr(L, L0)[0]), angle_to_sigma0_deg=X.signed_perm_angle(Q, Q0),
                             oracle_angle_deg=X.signed_perm_angle(Q, st.oracle), sUX=ml.s["UX_neff"], sUU=ml.s["UUd_share"],
                             mean_abs_dp=float(np.abs(z["p0"] - p0).mean()), rel_SX_change=float(np.linalg.norm(SX.mean(0) - z0["SX_s"].mean(0)) / np.linalg.norm(z0["SX_s"].mean(0))),
                             rel_T4_change=float(np.linalg.norm(T4.mean(0) - z0["T4_s"].mean(0)) / np.linalg.norm(z0["T4_s"].mean(0)))))
            r = rows[-1]
            print(f"{subj} {kernel} n_in={n_in} sigma={s:<5} {agg:19s} mcc {mcc:.4f} leak {leak:.4f} max {lmax:.3f} | spearman {r['spearman_vs_sigma0']:.3f} dp {r['mean_abs_dp']:.4f} "
                  f"dSX {r['rel_SX_change']:.2f} dT4 {r['rel_T4_change']:.2f}", flush=True)
            pd.DataFrame(rows).to_csv(RES / f"exp22_{model}_rep{rep}_{ch}_{subj}_{kernel}_n{n_in}.csv", index=False)
    json.dump(dict(model=model, rep=rep, channel=ch, subjects=subj, kernel=kernel, n_in=n_in), open(RES / f"exp22_{model}_rep{rep}_{ch}_{subj}_{kernel}_n{n_in}.COMPLETE.json", "w"))


{"cache": stage_cache, "eval": stage_eval}[stage]()
