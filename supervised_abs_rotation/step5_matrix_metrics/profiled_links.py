"""Local output-link profiling of Hessian structure; no link formula supplied.

H transforms as a H + b gg' under a smooth scalar output transformation.
Profiling a multiple of gg' out of off-diagonal H and normalizing the
residual removes this nuisance locally. It does NOT establish that the
profiled scalar is a function of Y or defines a global integrable link.
"""
from __future__ import annotations

import numpy as np


def residual(g, h, method="l2"):
    d = g.shape[-1]
    ii, jj = np.triu_indices(d, 1)
    outer = g[..., :, None]*g[..., None, :]
    a, b = h[..., ii, jj], outer[..., ii, jj]
    if method == "l2":
        lam = np.sum(a*b, axis=-1)/np.maximum(np.sum(b*b, axis=-1), 1e-300)
    elif method == "l1":
        ratios = np.divide(a, b, out=np.zeros_like(a), where=np.abs(b) > 1e-150)
        order = np.argsort(ratios, axis=-1)
        weights = np.take_along_axis(np.abs(b), order, axis=-1)
        ratios = np.take_along_axis(ratios, order, axis=-1)
        ids = np.argmax(np.cumsum(weights, axis=-1) >= .5*weights.sum(-1, keepdims=True), axis=-1)
        lam = np.take_along_axis(ratios, ids[..., None], axis=-1)[..., 0]
    else:
        raise ValueError(method)
    r = h-lam[..., None, None]*outer
    norm = np.sqrt(np.sum(r*r, axis=(-1, -2)))
    scale = np.sqrt(np.sum(h*h, axis=(-1, -2)))
    active = norm > 1e-10*np.maximum(scale, 1e-100)
    unit = np.divide(r, norm[..., None, None], out=np.zeros_like(r), where=active[..., None, None])
    return unit, lam, active


def normalized_tails(energy):
    total = energy.sum(-1)
    order = np.sort(energy, axis=-1)[..., ::-1]
    active = total > 1e-24
    out = {}
    for k in (1, 2, 4):
        tails = np.sqrt(order[..., k:].sum(-1)/np.maximum(total, 1e-24))
        out[f"tail{k}_l2"] = float(tails[active].mean()) if np.any(active) else 0.
    p = energy/np.maximum(total[..., None], 1e-24)
    entropy = -np.sum(p*np.log(np.maximum(p, 1e-300)), axis=-1)/max(np.log(energy.shape[-1]), 1.)
    out["entropy"] = float(entropy[active].mean()) if np.any(active) else 0.
    return out


def scores(g, h, method):
    r, lam, active = residual(g, h, method)
    d = g.shape[-1]
    ii, jj = np.triu_indices(d, 1)
    energy = r*r
    edges = 2*energy[..., ii, jj]
    diagonal = np.diagonal(energy, axis1=-2, axis2=-1)
    all_entries = np.concatenate([diagonal, edges], axis=-1)
    out = {"profiled_edge_fraction": float(edges.sum(-1).mean()),
           "profiled_axis_curvature_l1": float(np.abs(np.diagonal(r, axis1=-2, axis2=-1)).sum(-1).mean())}
    for prefix, values in (("point_all", all_entries), ("point_edges", edges),
                           ("pooled_all", all_entries.mean(axis=0, keepdims=True)),
                           ("pooled_edges", edges.mean(axis=0, keepdims=True))):
        for metric, value in normalized_tails(values).items():
            out[prefix+"_"+metric] = value
    audit = dict(active_fraction=float(active.mean()),
                 lambda_absolute_median=float(np.median(np.abs(lam))),
                 local_nuisance_parameters=int(lam.size),
                 global_link_integrability_verified=False)
    return out, audit
