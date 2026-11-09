"""Standalone exact T,T and T,X Hessians for C1/C2/C3/CR3.

Physical parameters T; X is in ORIGINAL raw stimulus coordinates.  C1-3
are linear in T; probability derivatives use the sigmoid chain rule.
CR3 uses the original separate gain/loss probability weighting calls.
Never differentiate a discrete Bernoulli observation.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit, logit

DIMENSIONS = {"C1": (4, 3), "C2": (4, 4), "C3": (6, 4), "CR3": (4, 6)}
CR3_EPS = 1e-12


def _inputs(t, x, model):
    model = str(model).upper()
    if model not in DIMENSIONS:
        raise ValueError(f"Unknown model {model}; valid: {list(DIMENSIONS)}")
    t, x = np.asarray(t, float), np.asarray(x, float)
    d, p = DIMENSIONS[model]
    if t.ndim != 2 or x.ndim != 2 or t.shape != (len(x), d) or x.shape[1] != p:
        raise ValueError(f"{model}: paired T (N,{d}), X (N,{p}) required; got {t.shape},{x.shape}")
    return t, x, model


def c_features(x, model):
    """Return phi(X) and Jacobian d phi_j/dX_k."""
    n, p = x.shape
    d = DIMENSIONS[model][0]
    phi = np.zeros((n, d), float)
    j = np.zeros((n, d, p), float)
    c = 3.0 * np.sqrt(5.0) / 2.0
    s = np.sqrt(3.0)
    phi[:, 0] = 1.0
    if model == "C1":
        phi[:, 1:] = s * x
        for k in range(3):
            j[:, k + 1, k] = s
        return phi, j
    a, b, u, v = x.T
    phi[:, 1] = s * (a - b)
    phi[:, 2] = c * (a*a - b*b)
    phi[:, 3] = s * (u - v)
    j[:, 1, 0], j[:, 1, 1] = s, -s
    j[:, 2, 0], j[:, 2, 1] = 2*c*a, -2*c*b
    j[:, 3, 2], j[:, 3, 3] = s, -s
    if model == "C3":
        phi[:, 4] = c * (u*u - v*v)
        phi[:, 5] = 3.0 * (a*u - b*v)
        j[:, 4, 2], j[:, 4, 3] = 2*c*u, -2*c*v
        j[:, 5, 0], j[:, 5, 1] = 3*u, -3*v
        j[:, 5, 2], j[:, 5, 3] = 3*a, -3*b
    return phi, j


def cr3_outputs(t, x):
    """Actual CPT-like logit and probability, including zero-amount stimuli."""
    a, lam, gam, tau = t.T
    delta = np.zeros(len(t))
    for opt, sign in ((0, 1.0), (1, -1.0)):
        gain, loss, prob = x[:, 3*opt:3*opt+3].T
        wp = expit(gam * logit(np.clip(prob, CR3_EPS, 1-CR3_EPS)))
        wl = expit(gam * logit(np.clip(1-prob, CR3_EPS, 1-CR3_EPS)))
        delta += sign * (wp*(gain/100.0)**a - lam*wl*(loss/100.0)**a)
    eta = tau*delta
    return eta, expit(eta)


def values(t, x, model):
    t, x, model = _inputs(t, x, model)
    if model == "CR3":
        return cr3_outputs(t, x)
    phi, _ = c_features(x, model)
    eta = np.einsum("ni,ni->n", t, phi)
    return eta, expit(eta)


def valid_x_mask(x, model):
    x = np.asarray(x, float)
    if model != "CR3":
        return np.isfinite(x).all(axis=1)
    # X amount derivatives are singular at zero for alpha in (0,1).
    # Never silently report conditional means as unconditional 500-trial means.
    return (np.isfinite(x).all(axis=1)
            & np.all(x[:, [0, 1, 3, 4]] > 0, axis=1)
            & np.all(x[:, [2, 5]] > 1e-9, axis=1)
            & np.all(x[:, [2, 5]] < 1-1e-9, axis=1))


def derivatives(t, x, model):
    """Return dict eta,p and local TT,TX Hessians for both channels.

    H arrays have leading N; requires valid CR3 X (positive amounts).
    """
    t, x, model = _inputs(t, x, model)
    n = len(t)
    d, p = DIMENSIONS[model]
    if not np.all(valid_x_mask(x, model)):
        raise ValueError("Analytic TX Hessian undefined for some X; filter CR3 zero-amount rows")
    if model != "CR3":
        phi, j = c_features(x, model)
        eta = np.einsum("ni,ni->n", t, phi)
        gt = phi
        gx = np.einsum("ni,nip->np", t, j)
        htt = np.zeros((n, d, d))
        htx = j
    else:
        a, lam, gam, tau = t.T
        eta = np.zeros(n)
        gt = np.zeros((n, 4))
        gx = np.zeros((n, 6))
        htt = np.zeros((n, 4, 4))
        htx = np.zeros((n, 4, 6))
        def put(i, j, v):
            htt[:, i, j] += v
            if i != j:
                htt[:, j, i] += v
        for opt, sign in ((0, 1.0), (1, -1.0)):
            k = 3*opt
            gain, loss, prob = x[:, k:k+3].T
            rg = logit(np.clip(prob, CR3_EPS, 1-CR3_EPS))
            rl = logit(np.clip(1-prob, CR3_EPS, 1-CR3_EPS))
            wg, wl = expit(gam*rg), expit(gam*rl)
            wg_g = wg*(1-wg)*rg
            wl_g = wl*(1-wl)*rl
            wg_gg = wg*(1-wg)*(1-2*wg)*rg**2
            wl_gg = wl*(1-wl)*(1-2*wl)*rl**2
            wg_p = gam*wg*(1-wg)/(prob*(1-prob))
            wl_p = -gam*wl*(1-wl)/(prob*(1-prob))
            wg_gp = wg*(1-wg)/(prob*(1-prob))*(1+gam*rg*(1-2*wg))
            wl_gp = -wl*(1-wl)/(prob*(1-prob))*(1+gam*rl*(1-2*wl))
            ga = (gain/100.0)**a
            lo = (loss/100.0)**a
            lg, ll = np.log(gain/100.0), np.log(loss/100.0)
            val = wg*ga - lam*wl*lo
            va = wg*ga*lg - lam*wl*lo*ll
            vl = -wl*lo
            vg = wg_g*ga - lam*wl_g*lo
            eta += sign*tau*val
            gt[:, 0] += sign*tau*va
            gt[:, 1] += sign*tau*vl
            gt[:, 2] += sign*tau*vg
            gt[:, 3] += sign*val
            gx[:, k] += sign*tau*wg*a*ga/gain
            gx[:, k+1] += sign*(-tau*lam*wl*a*lo/loss)
            gx[:, k+2] += sign*tau*(wg_p*ga-lam*wl_p*lo)
            put(0, 0, sign*tau*(wg*ga*lg**2-lam*wl*lo*ll**2))
            put(0, 1, sign*(-tau*wl*lo*ll))
            put(0, 2, sign*tau*(wg_g*ga*lg-lam*wl_g*lo*ll))
            put(0, 3, sign*va)
            put(1, 2, sign*(-tau*wl_g*lo))
            put(1, 3, sign*vl)
            put(2, 2, sign*tau*(wg_gg*ga-lam*wl_gg*lo))
            put(2, 3, sign*vg)
            htx[:, 0, k] += sign*tau*wg*ga/gain*(1+a*lg)
            htx[:, 0, k+1] += sign*(-tau*lam*wl*lo/loss*(1+a*ll))
            htx[:, 0, k+2] += sign*tau*(wg_p*ga*lg-lam*wl_p*lo*ll)
            htx[:, 1, k+1] += sign*(-tau*wl*a*lo/loss)
            htx[:, 1, k+2] += sign*(-tau*wl_p*lo)
            htx[:, 2, k] += sign*tau*wg_g*a*ga/gain
            htx[:, 2, k+1] += sign*(-tau*lam*wl_g*a*lo/loss)
            htx[:, 2, k+2] += sign*tau*(wg_gp*ga-lam*wl_gp*lo)
            htx[:, 3, k] += sign*wg*a*ga/gain
            htx[:, 3, k+1] += sign*(-lam*wl*a*lo/loss)
            htx[:, 3, k+2] += sign*(wg_p*ga-lam*wl_p*lo)
    prob = expit(eta)
    dp = prob*(1-prob)
    d2p = dp*(1-2*prob)
    ptt = dp[:, None, None]*htt + d2p[:, None, None]*gt[:, :, None]*gt[:, None, :]
    ptx = dp[:, None, None]*htx + d2p[:, None, None]*gt[:, :, None]*gx[:, None, :]
    return dict(eta=eta, probability=prob, tt_logit=htt, tx_logit=htx,
                tt_probability=ptt, tx_probability=ptx)
