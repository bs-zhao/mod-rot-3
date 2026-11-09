"""EXP02 - Gap 2: the Gap-1 survivors, recomputed from numeric (W, Y) only.

W = [t, X] with t = Tstd of every TRAIN subject and that subject's REAL
trials X (C1-C3: own random X; CR3: shared 500 bank).  Y is produced by the
generator formula (logit / prob / fresh Bernoulli binary).  The M2 sieve
(common/sieve.py) only receives the arrays.  Derivatives of the fitted h_hat
are evaluated on the SAME S x K grid as EXP01 (same seed) and pushed through
the SAME views/losses/landscape code.  Identity = truth class.

python three_gaps/exp02_gap2_wy/run.py --model C1 --beta 0.0005 --zdim 4 --pca 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import io  # noqa: E402
from common.analytic import X_DIM, ETA, make_h  # noqa: E402
from common.derivs import base_derivatives  # noqa: E402
from common.rotlib import VIEWS, add_centered, haar  # noqa: E402
from common.screen import landscape  # noqa: E402
from common.sieve import Sieve  # noqa: E402

NEEDS = {"g": "g", "gc": "g", "Htt": "Htt", "Htx": "Htx", "Tttx": "Tttx", "Ttxx": "Ttxx", "Tttt": "Tttt"}


def candidates(model, cond, pca, tag, explicit):
    if explicit:
        return [tuple(c.split("|")) for c in explicit.split(";")]
    f = io.ROOT / "outputs" / "three_gaps" / "exp01" / tag / f"{model}_{cond}_pca{pca}" / "screen.csv"
    df = pd.read_csv(f)
    df = df[df.gap1_pass == True]  # noqa: E712
    return [(r.channel, r.view, r.loss) for r in df.itertuples()]


def main():
    ap = io.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--subjects", type=int, default=0)
    ap.add_argument("--nx", type=int, default=150)
    ap.add_argument("--du", type=int, default=0, help="U degree (0 = default 3 for C, 4 for CR3)")
    ap.add_argument("--dx", type=int, default=3)
    ap.add_argument("--channels", default="logit,prob,binary")
    ap.add_argument("--candidates", default="", help="'ch|view|loss;...' (default: EXP01 passes)")
    ap.add_argument("--exp01-tag", default="v1")
    ap.add_argument("--nrand", type=int, default=200)
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    dev = io.pick_device(a.device)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    data, truth = io.load_all(a.model, a.beta, a.zdim, a.pca, a.repeat)
    d, p = io.D_T[a.model], X_DIM[a.model]
    bank = a.model == "CR3"
    # ---- identical evaluation grid to EXP01 (same rng call order) ----
    S = a.subjects or (300 if bank else 400)
    ids = np.sort(rng.choice(data["train_ids"], S, replace=False))
    t_grid = torch.tensor(truth.std(ids), dtype=torch.float64)
    m0 = data["subject_index"] == 0
    X_grid = torch.tensor(data["x"][m0] if bank else rng.uniform(-1, 1, (a.nx, p)), dtype=torch.float64)
    t_dev = t_grid.to(dev, torch.float32)
    du = a.du or (4 if bank else 3)
    out = io.ROOT / "outputs" / "three_gaps" / "exp02" / a.tag / f"{a.model}_{data['condition']}_pca{a.pca}"
    out.mkdir(parents=True, exist_ok=True)

    # ---- numeric WY table (formula Y) on all TRAIN subjects ----
    tr = data["train_ids"]
    sel = np.isin(data["subject_index"], tr)
    si = data["subject_index"][sel]
    xw = data["x"][sel]
    tw = truth.std()[si]
    eta = ETA[a.model](torch.tensor(truth.theta[si]), torch.tensor(xw)).numpy()
    pw = 1 / (1 + np.exp(-eta))
    yb = (np.random.default_rng(a.seed + 1).uniform(size=len(pw)) < pw).astype(float)
    Ys = dict(logit=eta, prob=pw, binary=yb)
    cands = candidates(a.model, data["condition"], a.pca, a.exp01_tag, a.candidates)
    print(f"[{a.model}] WY rows={len(si)}  candidates={len(cands)}", flush=True)
    randQ = [haar(d, rng) for _ in range(a.nrand)]
    rows, curves = [], {}
    for ch in a.channels.split(","):
        ach = "prob" if ch == "binary" else ch  # binary sieve estimates p
        cc = [c for c in cands if c[0] == ach]
        if not cc:
            continue
        need = sorted({NEEDS[VIEWS[v][0]] for _, v, _ in cc})
        if bank and any(n in ("Htx", "Tttx", "Ttxx") for n in need):
            print("  [skip] X-derivative views are not estimable from a fixed bank:",
                  [v for _, v, _ in cc if NEEDS[VIEWS[v][0]] in ("Htx", "Tttx", "Ttxx")])
            cc = [c for c in cc if NEEDS[VIEWS[c[1]][0]] not in ("Htx", "Tttx", "Ttxx")]
            need = [n for n in need if n not in ("Htx", "Tttx", "Ttxx")]
        if not cc:
            continue
        t0 = time.time()
        sv = Sieve(d, p, du, a.dx, "onehot" if bank else "legendre", device=dev)
        if bank:
            K = int(m0.sum())
            subj = si[::K]  # dataset is subject-major, each subject = the bank in order
            if not (np.all(si.reshape(-1, K) == subj[:, None]) and
                    np.allclose(xw.reshape(len(subj), K, p), data["x"][m0][None])):
                raise RuntimeError("CR3 rows are not subject-major copies of the bank")
            Yst = Ys[ch].reshape(len(subj), K)
            sv.fit_bank(truth.std(subj), Yst)
            h_hat = sv.h_bank()
            Xeval = torch.eye(K, dtype=torch.float64)
        else:
            sv.fit(tw, xw, Ys[ch])
            h_hat = sv.h()
            Xeval = X_grid
        print(f"  [{ch}] sieve fitted du={du} dx={a.dx} ({time.time()-t0:.0f}s)", flush=True)
        est = add_centered(base_derivatives(h_hat, t_grid.to(dev), Xeval.to(dev), which=need, chunk=4096))
        ana = add_centered(base_derivatives(make_h(a.model, ach, truth.mu, truth.sd), t_grid.to(dev),
                                            X_grid.to(dev), which=need))
        # value / derivative fidelity on the grid
        fid = {k: float(((est[k] - ana[k]) ** 2).mean().sqrt() / (ana[k] ** 2).mean().sqrt().clamp_min(1e-12))
               for k in need}
        fid["h"] = float(((est["h"] - ana["h"]) ** 2).mean().sqrt() / (ana["h"] ** 2).mean().sqrt())
        print(f"  [{ch}] relative RMSE vs analytic: " + ", ".join(f"{k}={v:.3f}" for k, v in fid.items()), flush=True)
        for (_, view, loss) in cc:
            ra, ca = landscape(ana, t_dev, view, loss, d, rng, randQ=randQ, do_opt=False)
            re, ce = landscape(est, t_dev, view, loss, d, rng, randQ=randQ, starts=a.starts, steps=a.steps)
            r = dict(model=a.model, channel=ch, view=view, loss=loss,
                     fid_h=fid["h"], fid_block=fid[NEEDS[VIEWS[view][0]]],
                     curve_corr=float(np.corrcoef(ca.ravel(), ce.ravel())[0, 1]),
                     ana_L=ra["L_ref"], ana_depth=ra["plane_depth"])
            r.update({k: v for k, v in re.items() if k != "opt_best_Q"})
            r["gap2_pass"] = bool(re["planes_ok"] == re["n_planes"] and re["rand_below"] == 0
                                  and re["opt_best_rel"] > -1e-4 and re["opt_near_best_max_angle"] < 5)
            rows.append(r)
            curves[f"{ch}|{view}|{loss}"] = np.stack([ca, ce])
            print(f"    {view:12s} {loss:10s} pass={r['gap2_pass']} corr={r['curve_corr']:.3f} "
                  f"depth={re['plane_depth']:.4f} best_ang={re['opt_best_angle']:.1f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "gap2.csv", index=False)
    np.savez_compressed(out / "curves.npz", **{k.replace("/", "_"): v for k, v in curves.items()})
    cols = ["channel", "view", "loss", "fid_h", "fid_block", "curve_corr", "plane_depth", "rand_margin",
            "opt_best_angle", "opt_hit", "gap2_pass"]
    with open(out / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# EXP02 Gap2 (formula WY -> sieve) - {a.model}\n\ndu={du} dx={a.dx} rows={len(si)}\n\n")
        f.write(f"passes: {int(df.gap2_pass.sum()) if len(df) else 0} / {len(df)}\n\n")
        if len(df):
            f.write(df.sort_values(["gap2_pass", "plane_depth"], ascending=False)[cols]
                    .to_markdown(index=False, floatfmt=".4f"))
    json.dump(dict(vars(a), S=S, du=du), open(out / "manifest.json", "w"), indent=1)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
