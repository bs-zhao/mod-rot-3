"""Exact polynomial/Chebyshev tensor feature values and original-coordinate derivatives.

Each candidate preserves the original separable f(z,x)=phi_z(z)^T B phi_x(x).
Total-degree monomial and Chebyshev bases span the SAME finite polynomial
space at a fixed degree but differ under diagonal ridge regularization.
"""
from __future__ import annotations
from itertools import combinations_with_replacement
from functools import lru_cache
import numpy as np
from numpy.polynomial import Chebyshev


def exponent_table(dim:int, degree:int):
    if dim<1 or degree<0:raise ValueError("Invalid basis dimensionality/degree")
    rows=[np.zeros(dim,dtype=np.int16)]
    for k in range(1,degree+1):
        for comb in combinations_with_replacement(range(dim),k):
            rows.append(np.bincount(comb,minlength=dim).astype(np.int16))
    return np.asarray(rows,dtype=np.int16)


@lru_cache(maxsize=None)
def _cheb_coefs(degree,order):
    # A polynomial vector in the scaled variable, before 1/scale**order.
    return [Chebyshev.basis(k).deriv(order) for k in range(degree+1)]


class Features:
    def __init__(self,points,train_ids,degree,basis="monomial"):
        p=np.asarray(points,dtype=np.float64)
        train_ids=np.asarray(train_ids,dtype=np.int64)
        if p.ndim!=2 or len(train_ids)==0:raise ValueError("Expected nonempty 2D points")
        if basis not in ("monomial","chebyshev"):raise ValueError("Unknown basis")
        self.p,self.dim,self.degree,self.basis=p,p.shape[1],int(degree),basis
        self.powers=exponent_table(self.dim,self.degree)
        self.center=(p[train_ids].max(axis=0)+p[train_ids].min(axis=0))*.5
        self.scale=np.maximum((p[train_ids].max(axis=0)-p[train_ids].min(axis=0))*.5,1e-9)
        self.cache={}
    def matrix(self,derivatives=()):
        key=tuple(sorted(int(k) for k in derivatives))
        if key in self.cache:return self.cache[key]
        count=np.bincount(key,minlength=self.dim)
        norm=(self.p-self.center)/self.scale
        result=np.ones((len(self.p),len(self.powers)),dtype=np.float64)
        for j in range(self.dim):
            d=int(count[j])
            k=self.powers[:,j]
            if self.basis=="monomial":
                # Avoid 0**negative: product of falling factorials gives zero
                # for impossible derivatives; use clipped exponent.
                falling=np.ones(len(k),dtype=np.float64)
                for z in range(d):falling*=np.maximum(k-z,0)
                m=(norm[:,j,None]**np.maximum(k-d,0)[None,:])
                m*= (falling/self.scale[j]**d)[None,:]
            else:
                polynomials=_cheb_coefs(self.degree,d)
                table=np.stack([fun(norm[:,j]) for fun in polynomials],axis=1)
                m=table[:,k] / self.scale[j]**d
            result*=m
        self.cache[key]=result
        return result
    def sample_mean(self,derivatives=(),ids=None):
        v=self.matrix(derivatives)
        return np.mean(v if ids is None else v[ids],axis=0)


def validity_mask(X,derivatives_x):
    x=np.asarray(X)
    valid=np.ones(len(x),dtype=bool)
    # CR3 gain/loss has noninteger powers with undefined derivatives at 0.
    for j in derivatives_x:
        if j in (0,1,3,4):valid&=x[:,j]>0.
    return valid
