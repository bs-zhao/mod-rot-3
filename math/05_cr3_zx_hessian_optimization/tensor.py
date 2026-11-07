"""Fast multi-response tensor polynomial regression and 10x10 mean Hessians.

Uses exactly the old factorized two-sided Tikhonov estimator, but supports
orthogonal Chebyshev features, independent lambda_Z/lambda_X, broad degree
sweeps, and one expensive SVD per (basis,degree_Z,degree_X).

No oracle is used to fit, differentiate or tune coefficients.
"""
from __future__ import annotations
import numpy as np
from features import Features, validity_mask


class TensorExperiment:
    def __init__(self,Z,X,Y,train_ids,val_ids,test_ids,*,degree_z=4,
                 degree_x=3,basis="monomial",zbasis_override=None,xbasis_override=None):
        Z=np.asarray(Z,dtype=float)
        X=np.asarray(X,dtype=float)
        Y=np.asarray(Y,dtype=float)
        self.Z,self.X,self.Y=Z,X,Y
        self.train=np.asarray(train_ids,dtype=int)
        self.val=np.asarray(val_ids,dtype=int)
        self.test=np.asarray(test_ids,dtype=int)
        if Z.shape[1]!=4 or X.shape[1]!=6 or Y.shape[:2]!=(len(Z),len(X)):
            raise ValueError("Expected Z=(N,4), X=(M,6), Y=(N,M,channels)")
        if len(set(self.train.tolist()+self.val.tolist()+self.test.tolist()))!=len(Z):
            raise ValueError("Subject train/validation/test splits must partition N")
        self.zbasis=(zbasis_override if zbasis_override is not None
                     else Features(Z,self.train,degree_z,basis))
        self.xbasis=(xbasis_override if xbasis_override is not None
                     else Features(X,np.arange(len(X)),degree_x,basis))
        Pz=self.zbasis.matrix()
        Px=self.xbasis.matrix()
        self.normz=np.maximum(np.linalg.norm(Pz[self.train],axis=0),1e-12)
        self.normx=np.maximum(np.linalg.norm(Px,axis=0),1e-12)
        ztrain=Pz[self.train]/self.normz
        xtrain=Px/self.normx
        self.Uz,self.Sz,self.Vz=np.linalg.svd(ztrain,full_matrices=False)
        self.Ux,self.Sx,self.Vx=np.linalg.svd(xtrain,full_matrices=False)
        self.projected_y=np.stack(
            [self.Uz.T @ Y[self.train,:,c] @ self.Ux
             for c in range(Y.shape[2])],axis=-1)
        self.n_channels=Y.shape[2]
        self.features_z=Pz
        self.features_x=Px
        self._moment_cache_z={}
        self._moment_cache_x={}

    @property
    def feature_count(self):
        return int(self.features_z.shape[1]*self.features_x.shape[1])

    def fit(self,ridge_z,ridge_x):
        rz,rx=float(ridge_z),float(ridge_x)
        if rz<=0 or rx<=0:raise ValueError("Ridge must be > 0")
        sz=self.Sz/(self.Sz**2+rz)
        sx=self.Sx/(self.Sx**2+rx)
        scale=self.projected_y*sz[:,None,None]*sx[None,:,None]
        left=np.einsum("ai,ijc->ajc",self.Vz.T,scale,optimize=True)
        coefficients=np.einsum("ajc,jb->abc",left,self.Vx,optimize=True)
        coefficients/=(self.normz[:,None,None]*self.normx[None,:,None])
        if not np.isfinite(coefficients).all():
            raise FloatingPointError("Nonfinite tensor coefficients")
        return coefficients

    def predict(self,coefficients,ids):
        ids=np.asarray(ids,dtype=int)
        a=np.einsum("na,abc->nbc",self.features_z[ids],coefficients,optimize=True)
        return np.einsum("nbc,mb->nmc",a,self.features_x,optimize=True)

    def metrics(self,coefficients,ids):
        pred=self.predict(coefficients,ids)
        true=self.Y[np.asarray(ids,dtype=int)]
        err=pred-true
        return {"rmse":np.sqrt(np.mean(err**2,axis=(0,1))),
                "mae":np.mean(np.abs(err),axis=(0,1)),
                "bias":np.mean(err,axis=(0,1))}

    def _zmean(self,derivatives,ids):
        key=(tuple(sorted(derivatives)),tuple(ids.tolist()) if ids is not None else "all")
        if key not in self._moment_cache_z:
            self._moment_cache_z[key]=self.zbasis.sample_mean(derivatives,ids=ids)
        return self._moment_cache_z[key]

    def _xmean(self,derivatives):
        key=tuple(sorted(derivatives))
        if key not in self._moment_cache_x:
            mask=validity_mask(self.X,key)
            if not mask.any():raise ValueError(f"No valid X rows for derivatives {key}")
            self._moment_cache_x[key]=(
                self.xbasis.matrix(key)[mask].mean(axis=0),float(mask.mean()))
        return self._moment_cache_x[key]

    def mean_hessian(self,coefficients,eval_ids=None):
        """Return mean (10,10,channels) plus defined-domain fractions."""
        ids=None if eval_ids is None else np.asarray(eval_ids,dtype=int)
        out=np.empty((10,10,self.n_channels),dtype=float)
        validfrac=np.ones((10,10),dtype=float)
        for i in range(10):
            for j in range(i,10):
                dz=tuple(k for k in (i,j) if k<4)
                dx=tuple(k-4 for k in (i,j) if k>=4)
                mt=self._zmean(dz,ids)
                mx,frac=self._xmean(dx)
                h=np.einsum("a,abc,b->c",mt,coefficients,mx,optimize=True)
                out[i,j]=out[j,i]=h
                validfrac[i,j]=validfrac[j,i]=frac
        return out,validfrac
