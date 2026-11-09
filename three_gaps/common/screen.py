"""Shared rotation-landscape evaluation of one (view, loss) on a derivative base.

`base` holds derivative tensors in BASE coordinates v (analytic: Tstd;
WY-estimated: the coordinates the estimator was fitted in).  `t_dev` (S, d)
are the subjects' base coordinates used for the per-axis z-scoring.
`ref` (d x d) is the reference rotation that should be optimal (identity for
controlled Tstd coordinates).
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from .losses import LOSSES
from .rotlib import apply_view, axis_error, effective_A, givens, haar

ANGLES = np.radians(np.linspace(0, 90, 37))


def make_L(base, t_dev):
    dev = t_dev.device

    def L_of(M, view, loss):
        M = torch.as_tensor(M, dtype=torch.float32, device=dev)
        return LOSSES[loss](apply_view(base, view, effective_A(M, t_dev)))
    return L_of


def optimize(base, t_dev, view, loss, d, rng, starts=8, steps=200, lr=0.05, Q0s=None):
    dev = t_dev.device
    finals = []
    for s in range(starts):
        Q0 = torch.tensor(Q0s[s] if Q0s is not None else haar(d, rng), dtype=torch.float32, device=dev)
        W = torch.zeros(d, d, device=dev, requires_grad=True)
        opt = torch.optim.Adam([W], lr=lr)
        for _ in range(steps):
            Q = torch.matrix_exp(W - W.T) @ Q0
            Lq = LOSSES[loss](apply_view(base, view, effective_A(Q, t_dev)))
            opt.zero_grad(); Lq.backward(); opt.step()
        with torch.no_grad():
            Q = torch.matrix_exp(W - W.T) @ Q0
            Lf = float(LOSSES[loss](apply_view(base, view, effective_A(Q, t_dev))))
        finals.append((Lf, Q.cpu().numpy()))
    finals.sort(key=lambda z: z[0])
    return finals


def landscape(base, t_dev, view, loss, d, rng, ref=None, nrand=200, starts=8, steps=200,
              randQ=None, do_opt=True):
    ref = np.eye(d) if ref is None else ref
    L_of = make_L(base, t_dev)
    planes = list(itertools.combinations(range(d), 2))
    with torch.no_grad():
        LI = float(L_of(ref, view, loss))
        scale = max(abs(LI), 1e-12)
        curves, ok, depth = [], 0, np.inf
        for (i, j) in planes:
            c = np.array([float(L_of(givens(d, i, j, th) @ ref, view, loss)) for th in ANGLES])
            curves.append(c)
            k = int(np.argmin(c))
            if k in (0, len(c) - 1) and (c.max() - c.min()) / scale > 1e-4:
                ok += 1
            inner = c[(ANGLES > np.radians(10)) & (ANGLES < np.radians(80))]
            depth = min(depth, (inner.min() - LI) / scale)
        randQ = randQ if randQ is not None else [haar(d, rng) for _ in range(nrand)]
        Lr = np.array([float(L_of(Q, view, loss)) for Q in randQ])
    r = dict(L_ref=LI, planes_ok=ok, n_planes=len(planes), plane_depth=depth,
             rand_below=float(np.mean(Lr < LI - 1e-6 * scale)), rand_margin=float((Lr.min() - LI) / scale))
    if do_opt:
        fin = optimize(base, t_dev, view, loss, d, rng, starts, steps)
        best = fin[0][0]
        angs = [axis_error(Q @ ref.T)[0] for _, Q in fin]
        near = [a for (Lf, _), a in zip(fin, angs) if Lf <= best + 1e-4 * scale]
        r.update(opt_best_rel=(best - LI) / scale, opt_best_angle=angs[0],
                 opt_hit=float(np.mean(np.array(angs) < 5)), opt_near_best_max_angle=float(max(near)),
                 opt_best_Q=fin[0][1])
    return r, np.array(curves)
