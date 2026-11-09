"""EXP00 - supervised oracle ceilings in the learned Z space (uses T; diagnostic only).

For the chosen coordinates V (raw centred Z if --pca 0, TRAIN PCA scores if 1),
and additionally for TRAIN-whitened V, reports on TEST:
  * affine OLS ceiling: per-parameter R^2 of Tstd ~ A V + b (fit TRAIN)
  * orthogonal oracle B: R in SO(d) maximising sum of matched r^2 (fit TRAIN, 32 starts)
  * identity (no rotation) matched statistics
Saves R_oracle for later experiments (post-hoc reference only).

python three_gaps/exp00_oracle/run.py --model CR3 --beta 0.00005 --zdim 4 --pca 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import io  # noqa: E402
from common.rotlib import haar, match_report  # noqa: E402


def oracle_R(V, T, starts=32, steps=400, seed=0):
    from scipy.optimize import linear_sum_assignment
    rng = np.random.default_rng(seed)
    d = V.shape[1]
    Vt = torch.tensor(V - V.mean(0), dtype=torch.float64)
    Tt = torch.tensor((T - T.mean(0)) / T.std(0), dtype=torch.float64)
    best = (-np.inf, None)
    for s in range(starts):
        Q0 = torch.tensor(haar(d, rng))
        W = torch.zeros(d, d, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([W], lr=0.03)
        for it in range(steps):
            Q = torch.matrix_exp(W - W.T) @ Q0
            U = Vt @ Q.T
            U = U / U.std(0)
            C = (U.T @ Tt) / len(U)
            C2 = C ** 2
            r, c = linear_sum_assignment(-C2.detach().numpy())
            loss = -C2[r, c].sum()
            opt.zero_grad(); loss.backward(); opt.step()
        if -float(loss) > best[0]:
            best = (-float(loss), (torch.matrix_exp(W - W.T) @ Q0).detach().numpy())
    return best[1]


def main():
    ap = io.add_common_args(argparse.ArgumentParser())
    a = ap.parse_args()
    data, truth = io.load_all(a.model, a.beta, a.zdim, a.pca, a.repeat, need_checkpoint=False, whiten=a.whiten)
    tr, te = data["train_ids"], data["test_ids"]
    T = truth.std()
    out = io.ROOT / "outputs" / "three_gaps" / "exp00" / f"{a.model}_{data['condition']}_pca{a.pca}"
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    V = data["V"]
    cov = np.cov(V[tr].T)
    ev, evec = np.linalg.eigh(cov)
    Wh = evec @ np.diag(ev ** -0.5) @ evec.T  # symmetric (ZCA) whitening, TRAIN only
    for name, VV in (("raw", V), ("zca_white", V @ Wh.T)):
        X1 = np.c_[VV, np.ones(len(VV))]
        A, *_ = np.linalg.lstsq(X1[tr], T[tr], rcond=None)
        pred = X1[te] @ A
        r2 = 1 - ((pred - T[te]) ** 2).sum(0) / ((T[te] - T[te].mean(0)) ** 2).sum(0)
        R = oracle_R(VV[tr], T[tr], seed=a.seed)
        rep_or = match_report(VV[te] @ R.T, T[te])
        rep_id = match_report(VV[te], T[te])
        res[name] = dict(ols_r2=r2.round(4).tolist(), ols_r2_mean=float(r2.mean()),
                         oracle=dict({k: v for k, v in rep_or.items() if k != "C"}, C=np.round(rep_or["C"], 3).tolist()),
                         identity=dict({k: v for k, v in rep_id.items() if k != "C"}, C=np.round(rep_id["C"], 3).tolist()),
                         cond_cov=float(ev.max() / ev.min()) if name == "raw" else 1.0)
        np.save(out / f"R_oracle_{name}.npy", R)
        print(name, "OLS r2", r2.round(3), "| oracle MCC %.3f worst %.3f offmax %.3f | identity MCC %.3f" % (
            rep_or["mcc"], rep_or["worst"], rep_or["off_max"], rep_id["mcc"]))
    np.save(out / "zca_W.npy", Wh)
    json.dump(res, open(out / "oracle.json", "w"), indent=1)


if __name__ == "__main__":
    main()
