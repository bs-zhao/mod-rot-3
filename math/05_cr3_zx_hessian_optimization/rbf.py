"""Differentiable finite-feature Gaussian RBF product kernel for full ZX Hessians.

For each group G in {Z,X} use standardized Gaussian kernel responses
phi_G(g)_j = exp(-||((g-landmark_j)/sd)||²/(2 ell_G²)).
Fit f(Z,X)=phi_Z(Z)^T B phi_X(X) with two-sided SVD ridge as in tensor.py.
First/second derivatives are exact with respect to original Z/X units.

This is a new nonlinear function class, unlike monomial vs Chebyshev
(which share a polynomial function space at fixed degree).
"""
from __future__ import annotations
import numpy as np
from tensor import TensorExperiment


class RBFGroup:
    def __init__(self,points,train_ids,*,n_landmarks,lengthscale,seed):
        p=np.asarray(points,dtype=float)
        tr=np.asarray(train_ids,dtype=int)
        if p.ndim!=2 or not len(tr):raise ValueError("Expected matrix and training IDs")
        if lengthscale<=0 or n_landmarks<4:raise ValueError("Bad kernel configuration")
        self.p,self.dim=p,p.shape[1]
        self.center=p[tr].mean(axis=0)
        self.sd=np.maximum(p[tr].std(axis=0,ddof=1),1e-7)
        rng=np.random.default_rng(seed)
        indices=rng.choice(tr,size=min(int(n_landmarks),len(tr)),replace=False)
        self.landmarks=np.array(p[indices],copy=True)
        self.lengthscale=float(lengthscale)
        self._cache={}

    def matrix(self,derivatives=()):
        key=tuple(sorted(int(i) for i in derivatives))
        if len(key)>2:raise ValueError("Only first/second kernel derivatives supported")
        if key in self._cache:return self._cache[key]
        d=(self.p[:,None,:]-self.landmarks[None,:,:])/self.sd[None,None,:]
        k=np.exp(-.5*np.sum(d*d,axis=2)/self.lengthscale**2)
        if len(key)==0:
            out=k
        elif len(key)==1:
            i=key[0]
            out=k*(-d[:,:,i]/(self.lengthscale**2*self.sd[i]))
        else:
            i,j=key
            out=k*(d[:,:,i]*d[:,:,j]/
                 (self.lengthscale**4*self.sd[i]*self.sd[j]))
            if i==j:
                out-=k/(self.lengthscale**2*self.sd[i]**2)
        self._cache[key]=out
        return out

    def sample_mean(self,derivatives=(),ids=None):
        val=self.matrix(derivatives)
        return val.mean(axis=0) if ids is None else val[ids].mean(axis=0)


def make_rbf_experiment(Z,X,Y,train,val,test,*,landmarks_z=64,
                        landmarks_x=64,ell_z=1.0,ell_x=1.0,seed=20261106):
    zbasis=RBFGroup(Z,train,n_landmarks=landmarks_z,
                    lengthscale=ell_z,seed=seed)
    xbasis=RBFGroup(X,np.arange(len(X)),n_landmarks=landmarks_x,
                    lengthscale=ell_x,seed=seed+1)
    return TensorExperiment(Z,X,Y,train,val,test,
                            zbasis_override=zbasis,xbasis_override=xbasis)
