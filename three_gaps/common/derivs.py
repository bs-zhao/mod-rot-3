"""Exact derivative fields of a scalar function h(t, x) via torch.func.

`base_derivatives` returns tensors laid out as (S, K, ...) for an S x K grid of
(parameter point, stimulus) pairs.  Leading index groups after (S, K):
  g    : (d,)        dh/dt
  Htt  : (d, d)      d2h/dt dt
  Htx  : (d, p)      d2h/dt dx
  Tttx : (d, d, p)   d3h/dt dt dx
  Ttxx : (d, p, p)   d3h/dt dx dx
  Tttt : (d, d, d)   d3h/dt dt dt
All U-indices come first in every block; this is what `views.transform`
assumes.
"""
from __future__ import annotations

import torch
from torch.func import jacfwd, jacrev, vmap

ORDERS = ("g", "Htt", "Htx", "Tttx", "Ttxx", "Tttt")


def _fns(h):
    g = jacrev(h, 0)
    return {
        "g": g,
        "Htt": jacfwd(g, 0),
        "Htx": jacfwd(g, 1),
        "Tttx": jacfwd(jacfwd(g, 0), 1),
        "Ttxx": jacfwd(jacfwd(g, 1), 1),
        "Tttt": jacfwd(jacfwd(g, 0), 0),
    }


def base_derivatives(h, t_grid, x_grid, which=ORDERS, chunk=8192, out_dtype=torch.float32):
    """t_grid: (S, d); x_grid: (S, K, p) or (K, p) shared."""
    S, d = t_grid.shape
    if x_grid.ndim == 2:
        x_grid = x_grid.unsqueeze(0).expand(S, -1, -1)
    K, p = x_grid.shape[1], x_grid.shape[2]
    tt = t_grid.unsqueeze(1).expand(S, K, d).reshape(-1, d)
    xx = x_grid.reshape(-1, p)
    fns = _fns(h)
    out = {}
    for name in which:
        f = vmap(fns[name])
        parts = [f(tt[i:i + chunk], xx[i:i + chunk]).to(out_dtype) for i in range(0, len(tt), chunk)]
        out[name] = torch.cat(parts).reshape(S, K, *parts[0].shape[1:])
    vals = torch.cat([vmap(h)(tt[i:i + chunk], xx[i:i + chunk]) for i in range(0, len(tt), chunk)])
    out["h"] = vals.reshape(S, K).to(out_dtype)
    return out
