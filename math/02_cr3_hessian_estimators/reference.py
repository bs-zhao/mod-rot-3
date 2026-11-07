"""Independent CR3 truth/IO for benchmarking; estimator modules NEVER import it.

CR3 exact source: np/reference/s0_cr2_design_recovery.py.
Coordinates (alpha,lambda,gamma,tau). Only T-T Hessian is evaluated,
so zero-gain/zero-loss stimulus points present no undefined amount derivatives.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit, logit
from scipy.stats import beta, gamma as gamma_dist, truncnorm
from numpy.polynomial.legendre import leggauss

NAMES = ('alpha', 'lambda_loss', 'gamma', 'tau')
BOUNDS = np.array([[0.,1.],[.3,5.],[0.,1.],[.5,4.]], float)


def load_cr3(path):
    with np.load(path, allow_pickle=False) as d:
        need=('parameters','trial_counts','subject_index','trial_index','x','y','predictive_true')
        if not all(k in d.files for k in need):
            raise ValueError('CR3 dataset missing required fields')
        t = np.asarray(d['parameters'], float)
        count=np.asarray(d['trial_counts'],int)
        if t.ndim != 2 or t.shape[1] != 4 or not np.all(count == 500):
            raise ValueError('CR3 requires four true parameters and 500 trials per subject')
        n=len(t)
        if not np.array_equal(d['subject_index'],np.repeat(np.arange(n),500)) or not np.array_equal(d['trial_index'],np.tile(np.arange(500),n)):
            raise ValueError('CR3 dataset subject/trial ordering differs from generator')
        xx=np.asarray(d['x'],float).reshape(n,500,6)
        x=xx[0].copy()
        if not np.all(xx == x[None,:,:]):
            raise ValueError('All subjects must receive identical fixed X bank')
        y=np.asarray(d['y'],float).reshape(n,500)
        p=np.asarray(d['predictive_true'],float).reshape(n,500)
    if not np.isin(y, (0.,1.)).all() or not ((p>=0)&(p<=1)).all():
        raise ValueError('Invalid choices or probability')
    return t,x,dict(binary=y.mean(axis=1), probability=p.mean(axis=1)),p


def logits_and_hess_tt(T, X):
    """Return trial-averaged (eta, p, H_eta, H_p) at each T.

    Exact CR3 clipping and independent weight(p) and weight(1-p) calls.
    Vectorize over (subjects in current batch, 500 trials). Hessian of
    conditional probability, not realized binary choice.
    """
    T=np.asarray(T,float); X=np.asarray(X,float)
    n=len(T)
    a,lam,gam,tau=(T[:,None,i] for i in range(4))
    eta=np.zeros((n,len(X)))
    grad=np.zeros((n,len(X),4))
    H=np.zeros((n,len(X),4,4))
    def add(i,j,v):
        H[...,i,j]+=v
        if j!=i:H[...,j,i]+=v
    for opt, sign in ((0,1.), (1,-1.)):
        gain,loss,p=(X[None,:,opt*3+k] for k in range(3))
        r_g=logit(np.clip(p,1e-12,1-1e-12))
        r_l=logit(np.clip(1-p,1e-12,1-1e-12))
        wg=expit(gam*r_g); wl=expit(gam*r_l)
        wg1=wg*(1-wg)*r_g; wl1=wl*(1-wl)*r_l
        wg2=wg*(1-wg)*(1-2*wg)*r_g**2
        wl2=wl*(1-wl)*(1-2*wl)*r_l**2
        lg=np.log(np.maximum(gain/100.,1e-300));ll=np.log(np.maximum(loss/100.,1e-300))
        v_g=(gain/100.)**a;v_l=(loss/100.)**a
        g_a=v_g*lg; l_a=v_l*ll
        g_aa=g_a*lg; l_aa=l_a*ll
        v=wg*v_g-lam*wl*v_l
        va=wg*g_a-lam*wl*l_a
        vl=-wl*v_l
        vg=wg1*v_g-lam*wl1*v_l
        eta+=sign*tau*v
        grad[...,0]+=sign*tau*va
        grad[...,1]+=sign*tau*vl
        grad[...,2]+=sign*tau*vg
        grad[...,3]+=sign*v
        add(0,0,sign*tau*(wg*g_aa-lam*wl*l_aa))
        add(0,1,sign*(-tau*wl*l_a))
        add(0,2,sign*tau*(wg1*g_a-lam*wl1*l_a))
        add(0,3,sign*va)
        add(1,2,sign*(-tau*wl1*v_l))
        add(1,3,sign*vl)
        add(2,2,sign*tau*(wg2*v_g-lam*wl2*v_l))
        add(2,3,sign*vg)
    p=expit(eta)
    H_p=p[...,None,None]*(1-p)[...,None,None]*(H+(1-2*p)[...,None,None]*grad[:,:,:,None]*grad[:,:,None,:])
    return eta.mean(1),p.mean(1),H.mean(1),H_p.mean(1)


def truth(T,X,batch_size=48):
    n=len(T); values=[np.zeros(n),np.zeros(n)]; mats=[np.zeros((n,4,4)),np.zeros((n,4,4))]
    for i in range(0,n,batch_size):
        j=min(i+batch_size,n)
        out=logits_and_hess_tt(T[i:j],X)
        for k in range(2):
            values[k][i:j]=out[k];mats[k][i:j]=out[k+2]
    return {'logit':(values[0],mats[0]),'probability':(values[1],mats[1]),'binary':(values[1],mats[1])}


def prior_quadrature(n):
    """Product Gauss-Legendre quadrature for actual CR3 independent priors."""
    points, legw=leggauss(n)
    specs=[(0.,1.,lambda z:beta.pdf(z,12,3)),
           (.3,5.,lambda z:gamma_dist.pdf(z,9,scale=.22)/(gamma_dist.cdf(5,9,scale=.22)-gamma_dist.cdf(.3,9,scale=.22))),
           (0.,1.,lambda z:beta.pdf(z,9,4)),
           (.5,4.,lambda z:truncnorm.pdf(z,(.5-2.2)/.55,(4-2.2)/.55,loc=2.2,scale=.55))]
    nodes=[];weights=[];errors=[]
    for lo,hi,fun in specs:
        z=(points+1)*(hi-lo)/2+lo
        w=legw*(hi-lo)/2*fun(z)
        errors.append(float(w.sum()-1))
        nodes.append(z);weights.append(w/w.sum())
    T=np.stack(np.meshgrid(*nodes,indexing='ij'),axis=-1).reshape(-1,4)
    W=np.prod(np.stack(np.meshgrid(*weights,indexing='ij'),axis=-1),axis=-1).ravel()
    W/=W.sum()
    return T,W,errors


def check_logit_values(T,X,p_saved,atol=1e-12):
    """Compare EVERY saved trial probability to the exact CR3 equation."""
    T=np.asarray(T,float);X=np.asarray(X,float)
    worst=0.
    for i in range(0,len(T),32):
        t=T[i:i+32]
        a,lam,gamma,tau=(t[:,None,j] for j in range(4))
        dv=np.zeros((len(t),len(X)))
        for opt,sign in ((0,1.),(1,-1.)):
            g,loss,p=(X[None,:,opt*3+j] for j in range(3))
            wg=expit(gamma*logit(np.clip(p,1e-12,1-1e-12)))
            wl=expit(gamma*logit(np.clip(1-p,1e-12,1-1e-12)))
            dv+=sign*(wg*(g/100.)**a-lam*wl*(loss/100.)**a)
        predicted=expit(tau*dv)
        worst=max(worst,float(np.max(np.abs(predicted-p_saved[i:i+32]))))
    if worst>atol:
        raise ValueError(f'CR3 oracle p mismatch (max over all trials): {worst}')
    return worst
