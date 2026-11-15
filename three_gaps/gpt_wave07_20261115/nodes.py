"""Physical Gaussian integration, independently implemented."""
import numpy as np
from scipy.stats import qmc,norm,chi2
def gaussian(n,d,seed,method='qmc',order=3):
    if method=='mc': e=np.random.default_rng(seed).normal(size=(n,d));w=np.ones(n)/n;clipped=0
    elif method=='qmc':
        if n<=0 or n&(n-1):raise ValueError('QMC N must be a positive power of two')
        u=qmc.Sobol(d,scramble=True,seed=seed).random_base2(int(np.log2(n)))
        clipped=int(((u<1e-12)|(u>1-1e-12)).sum());e=norm.ppf(np.clip(u,1e-12,1-1e-12));w=np.ones(n)/n
    elif method=='gh':
        x,v=np.polynomial.hermite.hermgauss(order)
        E=np.meshgrid(*([np.sqrt(2)*x]*d),indexing='ij');W=np.meshgrid(*([v/np.sqrt(np.pi)]*d),indexing='ij')
        e=np.stack([a.ravel() for a in E],1);w=np.prod(np.stack([a.ravel() for a in W],1),axis=1);clipped=0
    else:raise ValueError(method)
    return e,w,dict(clipped=clipped,nodes=len(e),method=method,seed=seed,order=order)

def physical(e,C,mu,measure='P1'):
    eig=np.linalg.eigvalsh(C)
    if eig.min()<1e-10*eig.max():raise ValueError(f'near singular covariance {eig}; no silent jitter')
    if measure=='P1':return e@np.linalg.cholesky(C).T
    if measure=='P3':return e-mu  # N(0,I) in original Z, not V.
    raise ValueError(measure)

def antithetic(n,d,seed):
    e,_,_=gaussian(n//2,d,seed)
    # interleaving gives nested antithetic prefixes for every n.
    return np.stack([e,-e],1).reshape(n,d)

def truncated(e,w,q=.95):
    keep=(e*e).sum(1)<=chi2.ppf(q,e.shape[1]);v=w*keep
    if not v.sum():raise ValueError('no retained nodes')
    return v/v.sum(),float(w[~keep].sum())
