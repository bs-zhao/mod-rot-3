"""EXP01 - Gap 1 screen: which (derivative view x complexity loss) prefers the
true standardised parameter axes of the EXACT analytic generator?

Coordinates: t = Tstd (TRAIN mean/sd), u = Q t, Q in SO(d).  Q = I is the
truth class (with signed permutations).  For every channel x view x loss:
  * G0: invariance  L(P D Q) == L(Q)  for random Q, signed permutation P,
        and wild diagonal scale D (relative error reported)
  * all d(d-1)/2 Givens plane scans around I, 0..90 deg
  * Haar-random rotations: fraction with L < L(I), margin
  * multistart Adam on SO(d) (expm param) for screen survivors
No Z, no decoder, no WY estimation is used here (pure analytic oracle).

python three_gaps/exp01_gap1_analytic/run.py --model C1 --beta 0.0005 --zdim 4 --pca 0
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import io  # noqa: E402
from common.analytic import X_DIM, make_h, check_against_dataset  # noqa: E402
from common.derivs import base_derivatives  # noqa: E402
from common.losses import LOSSES  # noqa: E402
from common.rotlib import (VIEWS, add_centered, apply_view, axis_error, effective_A,  # noqa: E402
                           givens, haar)


def main():
    ap = io.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--subjects", type=int, default=0, help="0 = model default")
    ap.add_argument("--nx", type=int, default=150, help="common X points (C1-C3)")
    ap.add_argument("--nrand", type=int, default=200)
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--channels", default="logit,prob")
    ap.add_argument("--views", default="all")
    ap.add_argument("--losses", default="all")
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    dev = io.pick_device(a.device)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    data, truth = io.load_all(a.model, a.beta, a.zdim, a.pca, a.repeat, need_checkpoint=True)
    err = check_against_dataset(a.model, truth.theta, data["x"], data["subject_index"], data["predictive_true"])
    print(f"[sanity] analytic vs stored predictive_true max |dp| = {err:.2e}", flush=True)
    if err > 1e-6:
        raise RuntimeError("analytic formula does not reproduce the generator")

    d = io.D_T[a.model]
    p = X_DIM[a.model]
    S = a.subjects or (300 if a.model == "CR3" else 400)
    ids = np.sort(rng.choice(data["train_ids"], S, replace=False))
    t = torch.tensor(truth.std(ids), dtype=torch.float64)
    if a.model == "CR3":
        m0 = data["subject_index"] == 0
        X = torch.tensor(data["x"][m0], dtype=torch.float64)  # the shared 500x6 bank
    else:
        X = torch.tensor(rng.uniform(-1, 1, (a.nx, p)), dtype=torch.float64)
    out = io.ROOT / "outputs" / "three_gaps" / "exp01" / a.tag / f"{a.model}_{data['condition']}_pca{a.pca}"
    out.mkdir(parents=True, exist_ok=True)

    views = list(VIEWS) if a.views == "all" else a.views.split(",")
    losses = list(LOSSES) if a.losses == "all" else a.losses.split(",")
    planes = list(itertools.combinations(range(d), 2))
    angles = np.radians(np.linspace(0, 90, 37))
    randQ = [haar(d, rng) for _ in range(a.nrand)]
    t_dev = t.to(dev, torch.float32)
    rows, curves = [], {}

    for ch in a.channels.split(","):
        t0 = time.time()
        h = make_h(a.model, ch, truth.mu, truth.sd)
        base = base_derivatives(h, t.to(dev), X.to(dev))
        base = add_centered(base)
        print(f"[{a.model}/{ch}] derivatives {time.time()-t0:.1f}s", flush=True)
        g_energy = float((base["g"] ** 2).mean())

        def L_of(M, view, loss):
            M = torch.as_tensor(M, dtype=torch.float32, device=dev)
            B = apply_view(base, view, effective_A(M, t_dev))
            return LOSSES[loss](B)

        for view in views:
            B0 = apply_view(base, view, effective_A(torch.eye(d, device=dev), t_dev))
            energy = float((B0 ** 2).mean())
            for loss in losses:
                key = f"{ch}|{view}|{loss}"
                r = dict(model=a.model, channel=ch, view=view, loss=loss, energy=energy / g_energy)
                if energy < 1e-10 * g_energy:
                    r["status"] = "zero_block"
                    rows.append(r)
                    continue
                with torch.no_grad():
                    LI = float(L_of(np.eye(d), view, loss))
                    scale = max(abs(LI), 1e-12)
                    # G0 invariance audit
                    inv = 0.0
                    for _ in range(3):
                        Q = haar(d, rng)
                        P = np.eye(d)[rng.permutation(d)] * rng.choice([-1, 1], d)[:, None]
                        D = np.diag(np.exp(rng.normal(0, 1.5, d)))
                        inv = max(inv, abs(float(L_of(P @ D @ Q, view, loss)) - float(L_of(Q, view, loss))) / scale)
                    curv, plane_ok, depth = [], 0, np.inf
                    for (i, j) in planes:
                        c = np.array([float(L_of(givens(d, i, j, th), view, loss)) for th in angles])
                        curv.append(c)
                        k = int(np.argmin(c))
                        rel_range = (c.max() - c.min()) / scale
                        if k in (0, len(c) - 1) and rel_range > 1e-4:
                            plane_ok += 1
                        inner = c[(angles > np.radians(10)) & (angles < np.radians(80))]
                        depth = min(depth, (inner.min() - LI) / scale)
                    curves[key] = np.array(curv)
                    Lr = np.array([float(L_of(Q, view, loss)) for Q in randQ])
                r.update(status="ok", L_I=LI, invariance_err=inv, planes_ok=plane_ok, n_planes=len(planes),
                         plane_depth=depth, rand_below=float(np.mean(Lr < LI - 1e-6 * scale)),
                         rand_margin=float((Lr.min() - LI) / scale),
                         rand_median_rel=float((np.median(Lr) - LI) / scale))
                screen = plane_ok == len(planes) and r["rand_below"] == 0
                if screen and a.starts > 0:
                    best, finals = np.inf, []
                    for s in range(a.starts):
                        Q0 = torch.tensor(haar(d, rng), dtype=torch.float32, device=dev)
                        W = torch.zeros(d, d, device=dev, requires_grad=True)
                        opt = torch.optim.Adam([W], lr=0.05)
                        for _ in range(a.steps):
                            Q = torch.matrix_exp(W - W.T) @ Q0
                            Lq = LOSSES[loss](apply_view(base, view, effective_A(Q, t_dev)))
                            opt.zero_grad(); Lq.backward(); opt.step()
                        with torch.no_grad():
                            Q = (torch.matrix_exp(W - W.T) @ Q0).cpu().numpy()
                            Lf = float(L_of(Q, view, loss))
                        finals.append((Lf, axis_error(Q)[0]))
                    finals.sort()
                    best = finals[0][0]
                    near = [ang for (Lf, ang) in finals if Lf <= best + 1e-4 * scale]
                    r.update(opt_best_rel=(best - LI) / scale, opt_best_angle=finals[0][1],
                             opt_hit=float(np.mean([ang < 5 for _, ang in finals])),
                             opt_near_best_max_angle=float(max(near)))
                    r["gap1_pass"] = bool(r["opt_best_rel"] > -1e-4 and r["opt_near_best_max_angle"] < 5)
                else:
                    r["gap1_pass"] = False
                rows.append(r)
            print(f"  {ch} {view:12s} done ({time.time()-t0:.0f}s) passes so far="
                  f"{sum(1 for x in rows if x.get('gap1_pass'))}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "screen.csv", index=False)
    np.savez_compressed(out / "plane_curves.npz", **{k.replace("/", "_"): v for k, v in curves.items()})
    ok = df[df.status == "ok"].copy()
    ok["score"] = ok["plane_depth"].clip(upper=1) + ok["rand_margin"].clip(upper=1)
    passed = ok[ok.gap1_pass == True].sort_values("rand_margin", ascending=False)  # noqa: E712
    with open(out / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# EXP01 Gap1 analytic screen - {a.model}\n\nS={S}, K={len(X)}, planes={len(planes)}, "
                f"nrand={a.nrand}, starts={a.starts}x{a.steps}\n\n")
        f.write(f"max invariance error: {ok.invariance_err.max():.2e}\n\n")
        f.write(f"## Gap-1 passes: {len(passed)} / {len(ok)}\n\n")
        cols = ["channel", "view", "loss", "plane_depth", "rand_margin", "rand_median_rel", "opt_best_rel", "opt_hit"]
        f.write(passed[cols].to_markdown(index=False, floatfmt=".4f") if len(passed) else "none")
        f.write("\n\n## Near misses (all planes ok, top by rand_margin)\n\n")
        near = ok[(ok.planes_ok == ok.n_planes) & (ok.gap1_pass != True)].sort_values("rand_margin", ascending=False).head(25)  # noqa: E712
        f.write(near[cols + ["rand_below"]].to_markdown(index=False, floatfmt=".4f") if len(near) else "none")
    json.dump(dict(vars(a), S=S, K=len(X), subject_ids=ids.tolist()), open(out / "manifest.json", "w"), indent=1)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
