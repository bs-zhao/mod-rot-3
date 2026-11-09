"""EXP06 - two-stage method evaluated through all three gaps in one script.

Stage A: salient output-equivariant axes (shift / gain), common/equivariant.py
Stage B: rotation of the orthogonal complement minimising a complexity loss of
         the stimulus-basis projected gradient view GP (Legendre basis for
         continuous X, trial indicators for the CR3 bank).
Variants: 'AB' (both stages) and 'B' (stage B on the full SO(d)).

sources (same views / losses / optimiser everywhere):
  analytic        exact formula, coordinates Tstd             -> Gap 1  (truth = I)
  formula_sieve   formula Y on real TRAIN trials, M2 sieve     -> Gap 2  (truth = I)
  decoder_ad      frozen decoder autodiff, coordinates V       -> diagnostic
  decoder_sieve   decoder Y on real TRAIN trials, M2 sieve     -> Gap 3  (TEST |corr| vs T)

python three_gaps/exp06_two_stage/run.py --model CR3 --beta 0.00005 --zdim 4 --pca 0
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
from common.analytic import ETA, X_DIM, make_h  # noqa: E402
from common.decoder import decoder_eta, decoder_h  # noqa: E402
from common.derivs import base_derivatives  # noqa: E402
from common.equivariant import detect  # noqa: E402
from common.rotlib import add_projected, axis_error, match_report  # noqa: E402
from common.screen import optimize_constrained  # noqa: E402
from common.sieve import Sieve  # noqa: E402


def main():
    ap = io.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--sources", default="analytic,formula_sieve,decoder_ad,decoder_sieve")
    ap.add_argument("--channels", default="logit,prob,binary")
    ap.add_argument("--losses", default="rmsL1,colLog1,colLog2,colLog3,softL0_03")
    ap.add_argument("--variants", default="AB,B")
    ap.add_argument("--view", default="GP")
    ap.add_argument("--subjects", type=int, default=0)
    ap.add_argument("--nx", type=int, default=200)
    ap.add_argument("--deg", type=int, default=3)
    ap.add_argument("--du", type=int, default=0)
    ap.add_argument("--dx", type=int, default=3)
    ap.add_argument("--starts", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    dev = io.pick_device(a.device)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    data, truth = io.load_all(a.model, a.beta, a.zdim, a.pca, a.repeat, whiten=a.whiten)
    net, _ = io.load_decoder(data["paths"]["checkpoint"], dev)
    d, p = io.D_T[a.model], X_DIM[a.model]
    bank = a.model == "CR3"
    tr, te = data["train_ids"], data["test_ids"]
    V, Tst = data["V"], truth.std()
    S = a.subjects or (300 if bank else 400)
    ids = np.sort(rng.choice(tr, S, replace=False))
    m0 = data["subject_index"] == 0
    if bank:
        X = torch.tensor(data["x"][m0], dtype=torch.float64)
    else:
        X0 = rng.uniform(-1, 1, (a.nx, p))
        X = torch.tensor(np.concatenate([X0, -X0]), dtype=torch.float64)
    K = len(X)
    du = a.du or (4 if bank else 3)
    out = io.ROOT / "outputs" / "three_gaps" / "exp06" / a.tag / f"{a.model}_{data['condition']}_pca{a.pca}"
    out.mkdir(parents=True, exist_ok=True)
    od = io.ROOT / "outputs" / "three_gaps" / "exp00" / f"{a.model}_{data['condition']}_pca{a.pca}"
    R_or = np.load(od / "R_oracle_raw.npy")
    oracle_mcc = match_report(V[te] @ R_or.T, Tst[te])["mcc"]
    ident_mcc = match_report(V[te], Tst[te])["mcc"]
    sel = np.isin(data["subject_index"], tr)
    si, xw = data["subject_index"][sel], data["x"][sel]
    eta_f = ETA[a.model](torch.tensor(truth.theta[si]), torch.tensor(xw)).numpy()
    eta_d = decoder_eta(net, data["z"], xw, si, dev)
    rows = []

    def ys(eta, seed):
        pr = 1 / (1 + np.exp(-eta))
        return dict(logit=eta, prob=pr, binary=(np.random.default_rng(seed).uniform(size=len(pr)) < pr).astype(float))
    Yf, Yd = ys(eta_f, a.seed + 1), ys(eta_d, a.seed + 2)

    for src in a.sources.split(","):
        controlled = src in ("analytic", "formula_sieve")
        coords = Tst if controlled else V
        grid = torch.tensor(coords[ids], dtype=torch.float64, device=dev)
        gdev = grid.float()
        for ch in a.channels.split(","):
            if src in ("analytic", "decoder_ad") and ch == "binary":
                continue
            t0 = time.time()
            if src == "analytic":
                h = make_h(a.model, ch, truth.mu, truth.sd); Xe = X
            elif src == "decoder_ad":
                h = decoder_h(net, data["mu_z"], data["Minv"], ch); Xe = X
            else:
                Y = (Yf if src == "formula_sieve" else Yd)[ch]
                sv = Sieve(d, p, du, a.dx, "onehot" if bank else "legendre", device=dev)
                if bank:
                    subj = si[::K]
                    sv.fit_bank(coords[subj], Y.reshape(len(subj), K)); h = sv.h_bank()
                    Xe = torch.eye(K, dtype=torch.float64)
                else:
                    sv.fit(coords[si], xw, Y); h = sv.h(); Xe = X
            base = base_derivatives(h, grid, Xe.to(dev), which=("g",), chunk=4096)
            base = add_projected(base, X.to(dev), bank, a.deg)
            ach = "prob" if ch == "binary" else ch
            axes, rep = detect(base["g"].double().cpu().numpy(), base["h"].double().cpu().numpy(), ach)
            F = np.array([ax["w"] for ax in axes]).reshape(-1, d)
            stageA = ";".join(f"{ax['type']}({ax['e1']:.3f}/{ax['e2']:.3f})" for ax in axes) or "none"
            print(f"[{src}/{ch}] stage A: {stageA}  report={ {k: tuple(round(x, 3) for x in v) for k, v in rep.items()} } "
                  f"({time.time()-t0:.0f}s)", flush=True)
            for var in a.variants.split(","):
                Fv = F if var == "AB" else np.zeros((0, d))
                for loss in a.losses.split(","):
                    fin = optimize_constrained(base, gdev, a.view, loss, Fv, rng, a.starts, a.steps)
                    Lb, Q = fin[0]
                    sc = max(abs(Lb), 1e-12)
                    near = [axis_error(Qi @ Q.T)[0] for Li, Qi in fin if Li <= Lb + 1e-4 * sc]
                    r = dict(model=a.model, source=src, channel=ch, variant=var, loss=loss, stageA=stageA,
                             L_best=Lb, near_best_spread=max(near), n_starts=len(fin))
                    if controlled:
                        ang, mang, _ = axis_error(Q)
                        # loss at truth: identity (with stage-A rows replaced by the detected ones in AB)
                        r.update(angle_to_truth=ang, mean_angle=mang)
                        U = Tst[te] @ Q.T
                    else:
                        r.update(angle_to_oracle=axis_error(Q @ R_or.T)[0])
                        U = V[te] @ Q.T
                    rep_t = match_report(U, Tst[te])  # post-hoc only
                    r.update(test_mcc=rep_t["mcc"], test_worst=rep_t["worst"], test_off_max=rep_t["off_max"],
                             oracle_mcc=oracle_mcc if not controlled else 1.0, identity_mcc=ident_mcc)
                    rows.append(r)
                    print(f"   {var:2s} {loss:10s} mcc={rep_t['mcc']:.3f} worst={rep_t['worst']:.3f} "
                          f"offmax={rep_t['off_max']:.3f} " +
                          (f"angle={r['angle_to_truth']:.1f}" if controlled else f"angle_or={r['angle_to_oracle']:.1f}") +
                          f" spread={r['near_best_spread']:.1f}", flush=True)
            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    df = pd.DataFrame(rows)
    with open(out / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# EXP06 two-stage - {a.model}  (oracle TEST mcc {oracle_mcc:.3f}, identity {ident_mcc:.3f})\n\n")
        piv = df.pivot_table(index=["variant", "loss"], columns=["source", "channel"], values="test_mcc")
        f.write(piv.to_markdown(floatfmt=".3f"))
    json.dump(dict(vars(a), S=S, K=K), open(out / "manifest.json", "w"), indent=1)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
