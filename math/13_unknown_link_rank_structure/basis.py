"""Predeclared polynomial feature dictionary for an unknown monotone-index model.

Data-dependent standardization is learned ONLY from the training split. Polynomial
terms are not assumed to be the generating formula. For CR3, X is evaluated only
at valid observed lottery designs; no partial derivatives in X are used.
"""
from __future__ import annotations
from itertools import combinations_with_replacement
import numpy as np


class Basis:
    def __init__(self, z_degree=1, x_degree=2):
        if z_degree not in (1, 2, 3) or x_degree not in (1, 2):
            raise ValueError('z_degree must be 1..3; x_degree must be 1..2')
        self.z_degree = z_degree
        self.x_degree = x_degree

    @staticmethod
    def _exponents(d, degree):
        result=[]
        for k in range(1, degree+1):
            for idx in combinations_with_replacement(range(d), k):
                ex=np.bincount(idx,minlength=d).astype(int)
                result.append(ex)
        return np.asarray(result,dtype=int)

    @staticmethod
    def _poly(values, exponents):
        return np.prod(np.power(values[:,None,:],exponents[None,:,:]),axis=2)

    def fit(self, W, zdim):
        W=np.asarray(W,dtype=float)
        if W.ndim!=2 or not 1<=zdim<W.shape[1] or len(W)<10:
            raise ValueError('W must be (n,p), with 1 <= zdim < p and n >= 10')
        self.zdim=int(zdim); self.xdim=int(W.shape[1]-zdim)
        self.zexp=self._exponents(self.zdim,self.z_degree)
        self.xexp=self._exponents(self.xdim,self.x_degree)
        self.zmean=W[:,:zdim].mean(axis=0)
        self.zsd=np.maximum(W[:,:zdim].std(axis=0),1e-5)
        self.xmean=W[:,zdim:].mean(axis=0)
        self.xsd=np.maximum(W[:,zdim:].std(axis=0),1e-5)
        z=self._poly((W[:,:zdim]-self.zmean)/self.zsd,self.zexp)
        x=self._poly((W[:,zdim:]-self.xmean)/self.xsd,self.xexp)
        self.zpolymean=z.mean(axis=0)
        self.xpolymean=x.mean(axis=0)
        # Independent terms centered; product of centered terms carries Z:X.
        raw=self._raw(W)
        self.scale=np.maximum(raw.std(axis=0),1e-5)
        self.n_zterms=len(self.zexp);self.n_xterms=len(self.xexp)
        return self

    def _pieces(self,W):
        W=np.asarray(W,dtype=float)
        if W.ndim!=2 or W.shape[1]!=self.zdim+self.xdim:
            raise ValueError('Wrong W dimension')
        z=(W[:,:self.zdim]-self.zmean)/self.zsd
        x=(W[:,self.zdim:]-self.xmean)/self.xsd
        zp=self._poly(z,self.zexp)-self.zpolymean
        xp=self._poly(x,self.xexp)-self.xpolymean
        return zp,xp

    def _raw(self,W):
        zp,xp=self._pieces(W)
        return np.c_[zp,xp,np.einsum('ni,nj->nij',zp,xp).reshape(len(W),-1)]

    def transform(self,W):
        return (self._raw(W)/self.scale).astype('float32')

    def coefficient_matrix(self,scaled_coeff):
        c=np.asarray(scaled_coeff,dtype=float)/self.scale
        start=len(self.zexp)+len(self.xexp)
        return c[start:].reshape(len(self.zexp),len(self.xexp))

    def jacobian_z(self,Z):
        """Derivative of uncentered polynomial ψ_z in STANDARDIZED z coordinates.

        Shape (n, zdim, n_zterms), no numerical differentiation.
        """
        Z=np.asarray(Z,dtype=float)
        if Z.ndim!=2 or Z.shape[1]!=self.zdim:
            raise ValueError('Z anchor dimensionality mismatch')
        z=(Z-self.zmean)/self.zsd
        out=np.empty((len(z),self.zdim,len(self.zexp)),dtype=float)
        for t,ex in enumerate(self.zexp):
            for i in range(self.zdim):
                if ex[i]==0:out[:,i,t]=0.
                else:
                    ee=ex.copy();ee[i]-=1
                    out[:,i,t]=ex[i]*np.prod(z**ee[None,:],axis=1)
        return out

    @staticmethod
    def _names(ex,prefix):
        return ['*'.join(f'{prefix}{j+1}' if k==1 else f'{prefix}{j+1}^{k}'
                         for j,k in enumerate(row) if k>0) for row in ex]

    def serialize(self):
        return {'z_dim':self.zdim,'x_dim':self.xdim,'z_degree':self.z_degree,
                'x_degree':self.x_degree,'z_terms':self._names(self.zexp,'z'),
                'x_terms':self._names(self.xexp,'x'),
                'zmean':self.zmean.tolist(),'zsd':self.zsd.tolist(),
                'xmean':self.xmean.tolist(),'xsd':self.xsd.tolist(),
                'z_poly_mean':self.zpolymean.tolist(),'x_poly_mean':self.xpolymean.tolist(),
                'feature_scale':self.scale.tolist()}

    @classmethod
    def from_json(cls,d):
        m=cls(d['z_degree'],d['x_degree'])
        m.zdim=d['z_dim'];m.xdim=d['x_dim']
        m.zexp=m._exponents(m.zdim,m.z_degree)
        m.xexp=m._exponents(m.xdim,m.x_degree)
        for key,actual in [('zmean','zmean'),('zsd','zsd'),('xmean','xmean'),('xsd','xsd'),
                           ('z_poly_mean','zpolymean'),('x_poly_mean','xpolymean'),
                           ('feature_scale','scale')]:
            setattr(m,actual,np.asarray(d[key],dtype=float))
        m.n_zterms=len(m.zexp);m.n_xterms=len(m.xexp)
        return m
