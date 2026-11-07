"""Standalone CR3/CPT-like analytic oracle. No imports from sibling math experiments.

Inputs T=(alpha,lambda,gamma,tau), X=(gainA,lossA,pA,gainB,lossB,pB).
The exact generator uses two SEPARATE clipped weighting calls for p and 1-p.
Analytic derivatives require strictly positive amounts and interior probabilities.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit, logit
from scipy.stats import beta, gamma as gamma_dist, truncnorm

PARAM_NAMES = ('alpha', 'lambda', 'gamma', 'tau')
X_NAMES = ('gain_A', 'loss_A', 'prob_A', 'gain_B', 'loss_B', 'prob_B')
EPS = 1e-12


def sample_prior(n, rng):
    alpha = beta.rvs(12, 3, size=n, random_state=rng)
    lam = gamma_dist.rvs(9, scale=.22, size=n, random_state=rng)
    bad = (lam < .3) | (lam > 5.)
    while np.any(bad):
        lam[bad] = gamma_dist.rvs(9, scale=.22, size=int(np.sum(bad)), random_state=rng)
        bad = (lam < .3) | (lam > 5.)
    gamm = beta.rvs(9, 4, size=n, random_state=rng)
    tau = truncnorm.rvs((.5-2.2)/.55, (4.-2.2)/.55,
                        loc=2.2, scale=.55, size=n, random_state=rng)
    return np.column_stack((alpha, lam, gamm, tau))


def outputs(T, X):
    """Paired leading shapes, returns (eta, p), not Bernoulli draws."""
    T, X = np.asarray(T, float), np.asarray(X, float)
    if T.shape[:-1] != X.shape[:-1] or T.shape[-1] != 4 or X.shape[-1] != 6:
        raise ValueError('T and X must have matching leading shapes and widths 4,6')
    alpha, lam, gamma, tau = np.moveaxis(T, -1, 0)
    dv = np.zeros(T.shape[:-1], dtype=float)
    for opt, sign in ((0, 1.), (1, -1.)):
        g, loss, prob = np.moveaxis(X[..., 3*opt:3*opt+3], -1, 0)
        wg = expit(gamma * logit(np.clip(prob, EPS, 1.-EPS)))
        wl = expit(gamma * logit(np.clip(1.-prob, EPS, 1.-EPS)))
        dv += sign * (wg*(g/100.)**alpha-lam*wl*(loss/100.)**alpha)
    eta = tau*dv
    return np.stack((eta, expit(eta)), axis=-1)


def derivatives(T, X):
    """Paired arrays; return H_ZZ,H_ZX for eta and probability, shape (n,2,4,4),(n,2,4,6).

    Coordinates are physical T,X. Probability Hessian uses exact sigmoid chain rule.
    No X second derivatives are needed by this experiment.
    """
    T, X = np.asarray(T,float), np.asarray(X,float)
    if T.ndim != 2 or X.ndim != 2 or T.shape[0] != X.shape[0] or T.shape[1:] != (4,) or X.shape[1:] != (6,):
        raise ValueError('Expected paired T (n,4), X (n,6)')
    if np.any(X[:, [0,1,3,4]] <= 0) or np.any(X[:,[2,5]] <= EPS) or np.any(X[:,[2,5]] >= 1-EPS):
        raise ValueError('Analytic X derivatives require positive amounts and interior probabilities')
    n = len(T)
    a, lam, gamma, tau = T.T
    eta = np.zeros(n)
    gt = np.zeros((n,4)); gx = np.zeros((n,6))
    zz = np.zeros((n,4,4)); zx = np.zeros((n,4,6))
    def put(i,j,v):
        zz[:,i,j] += v
        if i!=j: zz[:,j,i] += v
    for opt,sign in ((0,1.),(1,-1.)):
        k=opt*3
        g, loss, prob = X[:, k:k+3].T
        rg,rl = logit(np.clip(prob,EPS,1-EPS)),logit(np.clip(1-prob,EPS,1-EPS))
        wg,wl = expit(gamma*rg),expit(gamma*rl)
        wg_gamma=wg*(1-wg)*rg; wl_gamma=wl*(1-wl)*rl
        wg_gg=wg*(1-wg)*(1-2*wg)*rg**2
        wl_gg=wl*(1-wl)*(1-2*wl)*rl**2
        wg_p=gamma*wg*(1-wg)/(prob*(1-prob))
        wl_p=-gamma*wl*(1-wl)/(prob*(1-prob))
        wg_gp=wg*(1-wg)/(prob*(1-prob))*(1+gamma*rg*(1-2*wg))
        wl_gp=-wl*(1-wl)/(prob*(1-prob))*(1+gamma*rl*(1-2*wl))
        ga=(g/100.)**a; lo=(loss/100.)**a
        lg,ll=np.log(g/100.),np.log(loss/100.)
        val=wg*ga-lam*wl*lo
        va=wg*ga*lg-lam*wl*lo*ll
        vl=-wl*lo
        vg=wg_gamma*ga-lam*wl_gamma*lo
        eta += sign*tau*val
        gt[:,0]+=sign*tau*va; gt[:,1]+=sign*tau*vl
        gt[:,2]+=sign*tau*vg; gt[:,3]+=sign*val
        gx[:,k]+=sign*tau*wg*a*ga/g
        gx[:,k+1]+=sign*(-tau*lam*wl*a*lo/loss)
        gx[:,k+2]+=sign*tau*(wg_p*ga-lam*wl_p*lo)
        put(0,0,sign*tau*(wg*ga*lg**2-lam*wl*lo*ll**2))
        put(0,1,sign*(-tau*wl*lo*ll))
        put(0,2,sign*tau*(wg_gamma*ga*lg-lam*wl_gamma*lo*ll))
        put(0,3,sign*va)
        put(1,2,sign*(-tau*wl_gamma*lo))
        put(1,3,sign*vl)
        put(2,2,sign*tau*(wg_gg*ga-lam*wl_gg*lo))
        put(2,3,sign*vg)
        zx[:,0,k]+=sign*tau*wg*ga/g*(1+a*lg)
        zx[:,0,k+1]+=sign*(-tau*lam*wl*lo/loss*(1+a*ll))
        zx[:,0,k+2]+=sign*tau*(wg_p*ga*lg-lam*wl_p*lo*ll)
        # lambda-gain cross derivative is identically zero.
        zx[:,1,k+1]+=sign*(-tau*wl*a*lo/loss)
        zx[:,1,k+2]+=sign*(-tau*wl_p*lo)
        zx[:,2,k]+=sign*tau*wg_gamma*a*ga/g
        zx[:,2,k+1]+=sign*(-tau*lam*wl_gamma*a*lo/loss)
        zx[:,2,k+2]+=sign*tau*(wg_gp*ga-lam*wl_gp*lo)
        zx[:,3,k]+=sign*wg*a*ga/g
        zx[:,3,k+1]+=sign*(-lam*wl*a*lo/loss)
        zx[:,3,k+2]+=sign*(wg_p*ga-lam*wl_p*lo)
    p=expit(eta); dp=p*(1-p); d2p=dp*(1-2*p)
    zz_p=dp[:,None,None]*zz+d2p[:,None,None]*gt[:,:,None]*gt[:,None,:]
    zx_p=dp[:,None,None]*zx+d2p[:,None,None]*gt[:,:,None]*gx[:,None,:]
    return np.stack((zz,zz_p),axis=1),np.stack((zx,zx_p),axis=1)
