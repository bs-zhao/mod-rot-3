"""Cached six-response data-only Hessian estimator in rotated U and fixed X.

The estimator is the tested step-05 two-sided tensor polynomial ridge
regression (same basis, feature normalization, and SVD shrinkage).
X basis, X SVD, and Y @ Ux are computed ONCE for all A rotations.

Every rotation refits the U feature geometry, six regression coefficient
matrices, and all 10x10 average Hessians. Never differentiates a neural net.
"""
from __future__ import annotations
import numpy as np
from features import Features,validity_mask


class RotationTensor:
    def __init__(self,X,Y,train_ids,*,degree_u=4,degree_x=4,
                 ridge_u=1e-6,ridge_x=1e-6,basis="monomial",
                 heldout_ids=None,validation_limit=128):
        self.X=np.asarray(X,dtype=np.float64)
        self.Y=np.asarray(Y,dtype=np.float64)
        self.train=np.asarray(train_ids,dtype=np.int64)
        if self.Y.ndim!=3 or self.Y.shape[1]!=len(self.X) or self.X.shape[1]!=6:
            raise ValueError("Y=(subjects,stimuli,6); X=(stimuli,6) required")
        if self.Y.shape[-1]!=6:raise ValueError("Exactly six stored response arrays required")
        if not len(self.train):raise ValueError("No training subjects")
        self.degree_u=int(degree_u)
        self.degree_x=int(degree_x)
        self.ridge_u=float(ridge_u)
        self.ridge_x=float(ridge_x)
        self.basis=basis
        if min(self.ridge_u,self.ridge_x)<=0:raise ValueError("Ridge must be positive")
        self.xbasis=Features(self.X,np.arange(len(self.X)),self.degree_x,basis)
        px=self.xbasis.matrix()
        self.xnorm=np.maximum(np.linalg.norm(px,axis=0),1e-12)
        self.xU,self.xS,self.xV=np.linalg.svd(px/self.xnorm,full_matrices=False)
        self.xshrink=self.xS/(self.xS**2+self.ridge_x)
        self.n_features_x=px.shape[1]
        # Reuse this projected response cube for ALL pair/angle sweeps.
        self.yproj=np.einsum("nmc,mk->nkc",self.Y[self.train],
                             self.xU,optimize=True)
        self.xmoment={}
        for i in range(10):
            for j in range(i,10):
                dx=tuple(k-4 for k in (i,j) if k>=4)
                if dx not in self.xmoment:
                    mask=validity_mask(self.X,dx)
                    if not mask.any():raise ValueError("Empty valid X support")
                    self.xmoment[dx]=(self.xbasis.matrix(dx)[mask].mean(axis=0),
                                      float(mask.mean()))
        self.validation=np.asarray(heldout_ids,dtype=int) if heldout_ids is not None else np.zeros(0,dtype=int)
        self.validation=self.validation[:int(validation_limit)]
        self._last=None

    def fit_at(self,A,Z,b):
        """A,Z,X,Y -> six fitted mean Hessians in U=AZ+b coordinates."""
        A=np.asarray(A,dtype=float);Z=np.asarray(Z,dtype=float)
        b=np.asarray(b,dtype=float)
        if A.shape!=(4,4) or b.shape!=(4,) or Z.shape!=(len(self.Y),4):
            raise ValueError("Mismatched A,Z,b shapes")
        U=Z@A.T+b
        f=Features(U,self.train,self.degree_u,self.basis)
        pu=f.matrix()
        unorm=np.maximum(np.linalg.norm(pu[self.train],axis=0),1e-12)
        uu,us,uv=np.linalg.svd(pu[self.train]/unorm,full_matrices=False)
        projected=np.einsum("na,nkc->akc",uu,self.yproj,optimize=True)
        ushrink=us/(us**2+self.ridge_u)
        scaled=projected*ushrink[:,None,None]*self.xshrink[None,:,None]
        left=np.einsum("ai,ijc->ajc",uv.T,scaled,optimize=True)
        coef=np.einsum("ajc,jb->abc",left,self.xV,optimize=True)
        coef/=unorm[:,None,None]*self.xnorm[None,:,None]
        if not np.isfinite(coef).all():raise FloatingPointError("Nonfinite fitted coefficients")
        umean={}
        for i in range(10):
            for j in range(i,10):
                du=tuple(k for k in (i,j) if k<4)
                if du not in umean:
                    umean[du]=f.matrix(du).mean(axis=0)
        mean=np.empty((6,10,10),dtype=float)
        validity=np.ones((10,10),dtype=float)
        for i in range(10):
            for j in range(i,10):
                du=tuple(k for k in (i,j) if k<4)
                dx=tuple(k-4 for k in (i,j) if k>=4)
                mx,fraction=self.xmoment[dx]
                v=np.einsum("a,abc,b->c",umean[du],coef,mx,optimize=True)
                mean[:,i,j]=mean[:,j,i]=v
                validity[i,j]=validity[j,i]=fraction
        val_rmse=None
        if len(self.validation):
            pval=pu[self.validation]
            pred=np.einsum("na,abc->nbc",pval,coef,optimize=True)
            pred=np.einsum("nbc,mb->nmc",pred,self.xbasis.matrix(),optimize=True)
            residual=pred-self.Y[self.validation]
            val_rmse=np.sqrt(np.mean(residual**2,axis=(0,1)))
        self._last={"A":A.copy(),"b":b.copy(),"coefficients":coef,
                    "n_u_features":pu.shape[1],"n_x_features":self.n_features_x}
        return mean,validity,val_rmse
