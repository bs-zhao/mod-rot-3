"""EXP03 - Gap 1, round 2: stimulus-basis projected derivative views + support losses.

Changes vs EXP01 (see EXPERIMENT_LOG.md for the reasons):
  * C1-C3 X grid is antithetic (X and -X) to remove odd sample cross-moments
    that shifted shallow quadratic minima by ~2.5 deg in EXP01.
  * new views GP/HUXP/HUUP/HUUdP/HUUoP: derivative fields projected onto an
    orthonormal stimulus basis Psi (Legendre for continuous X, trial indicators
    for the CR3 bank); cells = basis functions, aggregated over subjects.
  * support-type losses softL0_{03,10,30}, logSp.
  * 16 optimizer starts x 300 steps.

python three_gaps/exp03_gap1_projected/run.py --model C3 --beta 0.0005 --zdim 6 --pca 0
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
from common.analytic import X_DIM, make_h  # noqa: E402
from common.derivs import base_derivatives  # noqa: E402
from common.rotlib import add_centered, add_projected, axis_error, haar  # noqa: E402
from common.screen import landscape, optimize  # noqa: E402

VIEWS = "GP,HUXP,HUUP,HUUdP,HUUoP,HUX/pool,HUUd/pool,G/trial"
LOSSES = "rmsL1,absL1,colL1L2,cellNeff,quartimax,logSp,softL0_03,softL0_10,softL0_30,localL1"


def grid(a, data, truth, rng):
    d, p = io.D_T[a.model], X_DIM[a.model]
    bank = a.model == "CR3"
    S = a.subjects or (300 if bank else 400)
    ids = np.sort(rng.choice(data["train_ids"], S, replace=False))
    t = torch.tensor(truth.std(ids), dtype=torch.float64)
    if bank:
        X = torch.tensor(data["x"][data["subject_index"] == 0], dtype=torch.float64)
    else:
        X0 = rng.uniform(-1, 1, (a.nx, p))
        X = torch.tensor(np.concatenate([X0, -X0]), dtype=torch.float64)
    return d, p, bank, S, ids, t, X


def main():
    ap = io.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--subjects", type=int, default=0)
    ap.add_argument("--nx", type=int, default=200, help="antithetic pairs (C1-C3)")
    ap.add_argument("--deg", type=int, default=3, help="Legendre total degree of Psi")
    ap.add_argument("--nrand", type=int, default=200)
    ap.add_argument("--starts", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--channels", default="logit,prob")
    ap.add_argument("--views", default=VIEWS)
    ap.add_argument("--losses", default=LOSSES)
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    dev = io.pick_device(a.device)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    data, truth = io.load_all(a.model, a.beta, a.zdim, a.pca, a.repeat)
    d, p, bank, S, ids, t, X = grid(a, data, truth, rng)
    out = io.ROOT / "outputs" / "three_gaps" / "exp03" / a.tag / f"{a.model}_{data['condition']}_pca{a.pca}"
    out.mkdir(parents=True, exist_ok=True)
    randQ = [haar(d, rng) for _ in range(a.nrand)]
    t_dev = t.to(dev, torch.float32)
    rows, curves = [], {}
    for ch in a.channels.split(","):
        t0 = time.time()
        base = base_derivatives(make_h(a.model, ch, truth.mu, truth.sd), t.to(dev), X.to(dev),
                                which=("g", "Htt", "Htx"))
        base = add_projected(add_centered(base), X.to(dev), bank, a.deg)
        g_energy = float((base["g"] ** 2).mean())
        for view in a.views.split(","):
            for loss in a.losses.split(","):
                from common.rotlib import VIEWS as V, apply_view, effective_A
                B0 = apply_view(base, view, effective_A(torch.eye(d, device=dev), t_dev))
                e = float((B0 ** 2).mean())
                r = dict(model=a.model, channel=ch, view=view, loss=loss)
                if e < 1e-10 * g_energy * max(1, B0.shape[-1]) ** -1:
                    r["status"] = "zero_block"; rows.append(r); continue
                res, cv = landscape(base, t_dev, view, loss, d, rng, randQ=randQ, starts=0, do_opt=False)
                r.update(status="ok", **res)
                if res["planes_ok"] == res["n_planes"] and res["rand_below"] == 0:
                    fin = optimize(base, t_dev, view, loss, d, rng, a.starts, a.steps)
                    sc = max(abs(res["L_ref"]), 1e-12)
                    angs = [axis_error(Q)[0] for _, Q in fin]
                    best = fin[0][0]
                    near = [ag for (Lf, _), ag in zip(fin, angs) if Lf <= best + 1e-4 * sc]
                    r.update(opt_best_rel=(best - res["L_ref"]) / sc, opt_best_angle=angs[0],
                             opt_hit=float(np.mean(np.array(angs) < 5)), opt_near_best_max_angle=max(near))
                    r["gap1_pass"] = bool(r["opt_best_rel"] > -1e-4 and max(near) < 5)
                else:
                    r["gap1_pass"] = False
                rows.append(r)
                curves[f"{ch}|{view}|{loss}"] = cv
            print(f"[{a.model}/{ch}] {view:10s} ({time.time()-t0:.0f}s) passes="
                  f"{sum(1 for x in rows if x.get('gap1_pass'))}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "screen.csv", index=False)
    np.savez_compressed(out / "plane_curves.npz", **{k.replace("/", "_"): v for k, v in curves.items()})
    ok = df[df.status == "ok"]
    cols = ["channel", "view", "loss", "planes_ok", "plane_depth", "rand_margin", "opt_best_rel", "opt_hit",
            "opt_near_best_max_angle"]
    with open(out / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# EXP03 Gap1 projected - {a.model}\n\nS={S} K={len(X)} deg={a.deg}\n\n")
        f.write(f"## passes {int(ok.gap1_pass.sum())}/{len(ok)}\n\n")
        f.write(ok[ok.gap1_pass == True].sort_values("plane_depth", ascending=False)[cols]  # noqa: E712
                .to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## all-planes-ok but not passed\n\n")
        f.write(ok[(ok.planes_ok == ok.n_planes) & (ok.gap1_pass != True)][cols + ["rand_below"]]  # noqa: E712
                .to_markdown(index=False, floatfmt=".4f"))
    json.dump(dict(vars(a), S=S, K=len(X), subject_ids=ids.tolist()), open(out / "manifest.json", "w"), indent=1)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
