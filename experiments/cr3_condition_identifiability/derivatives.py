"""Exact analytic CR3 response derivatives, with explicit domain masks.

The derivative target is the original generator, NOT any frozen NN. U is
true theta standardized by the original dataset population SD. X is raw
physical gain/loss/probability; results are additionally rescaled to
standardized X coordinates when requested. Zero-amount X derivatives
returned by torch.where are masked OUT, never treated as mathematical zero.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.func import jacfwd, jacrev, vmap
from scipy.special import expit

from three_gaps.common import analytic
from .design import Design, physical_finite_probes


def functions(design: Design, device="cpu"):
    mu = torch.as_tensor(design.mu, dtype=torch.float64, device=device)
    sd = torch.as_tensor(design.sd, dtype=torch.float64, device=device)
    def eta(t, x):
        theta = mu + sd * t
        return analytic.eta_cr3(theta, x)
    g = jacrev(eta, argnums=0)
    H = jacfwd(g, argnums=0)
    ux = jacfwd(g, argnums=1)
    gx = jacfwd(eta, argnums=1)
    return eta, g, H, ux, gx


def evaluate_many(fn, t, x, device="cpu", batch=128):
    """Vmap a scalar-valued or tuple-valued function over paired t,x rows."""
    if len(t) != len(x):
        raise ValueError("paired t,x rows required")
    result = None
    for a in range(0, len(t), batch):
        tv = torch.as_tensor(t[a:a+batch], dtype=torch.float64, device=device)
        xv = torch.as_tensor(x[a:a+batch], dtype=torch.float64, device=device)
        with torch.enable_grad():
            got = vmap(fn)(tv, xv)
        got = tuple(np.asarray(y.detach().cpu(), dtype=np.float64) for y in got) if isinstance(got, tuple) else (np.asarray(got.detach().cpu(), dtype=np.float64),)
        if result is None:
            result = [[] for _ in got]
        for group, y in zip(result, got):
            group.append(y)
    if result is None:
        raise ValueError("no evaluation points")
    return tuple(np.concatenate(parts, axis=0) for parts in result)


def true_generator_selfcheck(design: Design):
    """Check analytic reimplementation against the legacy CR3 generator formula."""
    import sys
    from pathlib import Path
    refdir = Path(__file__).resolve().parents[2] / "np/reference"
    if str(refdir) not in sys.path:
        sys.path.insert(0, str(refdir))
    from s0_cr2_design_recovery import logits
    theta = design.theta[:min(3,len(design.theta))]
    x = design.x[:min(8,len(design.x))]
    expected = logits(x[None],theta[:,None])
    actual = []
    with torch.no_grad():
        for tt in theta:
            xt=torch.tensor(x, dtype=torch.float64)
            th=torch.as_tensor(tt,dtype=torch.float64).expand(len(x),-1)
            actual.append(analytic.eta_cr3(th,xt).numpy())
    maximum = float(np.max(np.abs(np.asarray(actual)-expected)))
    if not np.isfinite(maximum) or maximum > 2e-10:
        raise RuntimeError(f"analytic CR3 differs from original legacy formula: {maximum}")
    return maximum


def compute_fields(design: Design, channel="logit", device="cpu",
                   batch=128, amount_step=5., probability_step=.05,
                   x_scale="standardized", finite=True):
    """Return complete derivative fields AND explicit validity/probe masks.

    Finite contrast in X is computed from the exact analytic U gradient:
       [grad_U f(x_plus) - grad_U f(x_minus)] / physical_width
    Its estimand is finite-scale and NEVER a claimed derivative at zero.
    """
    if channel not in ("logit", "prob"):
        raise ValueError("only logit/prob channels")
    if x_scale not in ("standardized", "raw"):
        raise ValueError("x_scale must be standardized or raw")
    S, K = len(design.theta), len(design.x)
    t = np.repeat(design.t, K, axis=0)
    x = np.tile(design.x, (S, 1))
    eta, gfn, Hfn, uxfn, gxfn = functions(design, device)
    def bundle(a, b):
        return (eta(a,b),gfn(a,b),Hfn(a,b),uxfn(a,b),gxfn(a,b))
    value, g, H, UX, gx = (
        z.reshape(S, K, *z.shape[1:]) for z in
        evaluate_many(bundle,t,x,device=device,batch=batch)
    )
    logit = value.copy()
    logit_gradient = g.copy()
    p = expit(logit)
    if channel == "prob":
        g0 = g.copy()
        UX0 = UX.copy()
        H0 = H.copy()
        gx0 = gx.copy()
        s1 = p*(1-p)
        s2 = s1*(1-2*p)
        g = s1[...,None]*g0
        H = s1[...,None,None]*H0 + s2[...,None,None]*np.einsum("...i,...j->...ij",g0,g0)
        UX = s1[...,None,None]*UX0 + s2[...,None,None]*np.einsum("...i,...j->...ij",g0,gx0)
        gx = s1[...,None]*gx0
        value = p
    xp, xm, width, flags, interior, tangent, one_sided = physical_finite_probes(
        design.x, design.codes, amount_step, probability_step)
    if x_scale == "standardized":
        UX = UX * design.x_sd[None,None,None,:]
        gx = gx * design.x_sd[None,None,:]
    finite_UX = np.zeros((S,K,4,6),dtype=np.float64)
    if finite:
        for j in range(6):
            plus_x = np.tile(xp[:,j,:],(S,1))
            minus_x = np.tile(xm[:,j,:],(S,1))
            eta_plus,g_plus = evaluate_many(lambda a,b:(eta(a,b),gfn(a,b)),t,plus_x,
                                            device=device,batch=batch)
            eta_minus,g_minus = evaluate_many(lambda a,b:(eta(a,b),gfn(a,b)),t,minus_x,
                                               device=device,batch=batch)
            if channel == "prob":
                pp = expit(eta_plus)
                pm = expit(eta_minus)
                g_plus = g_plus* (pp*(1-pp))[:,None]
                g_minus = g_minus* (pm*(1-pm))[:,None]
            numerator = (g_plus-g_minus).reshape(S,K,4)
            finite_UX[...,j] = numerator / width[None,:,j,None]
        if x_scale == "standardized":
            finite_UX = finite_UX * design.x_sd[None,None,None,:]
    if not all(np.isfinite(a).all() for a in (value,g,H,gx,UX,finite_UX)):
        raise FloatingPointError("non-finite CR3 analytic fields; inspect formula/input")
    return dict(value=value, logit=logit, p=p, G=g, G_logit=logit_gradient, H=H,
                X_gradient=gx, UX=UX, finite_UX=finite_UX,
                ambient_valid=interior, tangent_valid=tangent,
                finite_kind=flags, finite_onesided=one_sided,
                finite_width_physical=width,
                probability_weight=p*(1-p),
                x_scale=x_scale, channel=channel)


def save_fields(path, design, fields, save_dense=True):
    """Always save protocol and masks. Dense pointwise fields are optional."""
    path.parent.mkdir(parents=True,exist_ok=True)
    record = dict(X_raw=design.x, condition_codes=design.codes,
                  theta_raw=design.theta, T_standardized=design.t,
                  theta_mu=design.mu, theta_sd=design.sd, x_sd=design.x_sd,
                  subject_ids=design.subject_ids, trial_ids=design.trial_ids)
    for k,v in fields.items():
        if k in ("channel","x_scale"):
            continue
        if save_dense or k in ("ambient_valid","tangent_valid","finite_kind",
                              "finite_onesided","finite_width_physical"):
            record[k] = v
    np.savez_compressed(path,**record)
