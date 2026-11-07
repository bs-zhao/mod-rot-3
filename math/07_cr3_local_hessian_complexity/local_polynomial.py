"""Local Z/X polynomial derivatives through order four, on a SMALL fixed grid.

The SIX data-only models all come from original numeric Y; no NN gradients.
This evaluator reuses step-06 fitted polynomial coefficients and their EXACT
training centering/scaling, but evaluates derivatives locally instead of
averaging them before diagnostics.

All derivative arrays have channel axis FIRST and are in original U and X
units. The X-set must be fixed across angles and all three model sources.
"""
from __future__ import annotations
from itertools import combinations_with_replacement,permutations
import numpy as np
from features import Features


def _feature_on(points,base,derivatives):
    """Evaluate a frozen/trained feature basis at new points."""
    p=np.asarray(points,dtype=float)
    f=Features(p,np.arange(len(p)),base.degree,base.basis)
    f.center=base.center.copy()
    f.scale=base.scale.copy()
    return f.matrix(derivatives)


class LocalPolynomial:
    def __init__(self,engine,U_all,X_eval):
        if engine._last is None:
            raise ValueError("Run the frozen numeric-Y regression first")
        self.coef=np.asarray(engine._last["coefficients"],dtype=float)
        self.U=np.asarray(U_all,dtype=float)
        self.X=np.asarray(X_eval,dtype=float)
        self.zbasis=Features(self.U,engine.train,engine.degree_u,engine.basis)
        self.xbasis=engine.xbasis
        self.zm={}
        self.xm={}
        self.xproject={}
        self.basis_name=engine.basis

    def evaluate(self,U,du=(),dx=()):
        zu=np.asarray(U,dtype=float)
        du=tuple(sorted(int(i) for i in du))
        dx=tuple(sorted(int(i) for i in dx))
        keyu=(du,id(zu))
        # Use a per-call input matrix; repeated all-fields evaluation
        # explicitly caches each derivative at the single fixed grid.
        if not np.array_equal(zu,self.U):
            zu_basis=_feature_on(zu,self.zbasis,du)
        elif du not in self.zm:
            self.zm[du]=self.zbasis.matrix(du)
            zu_basis=self.zm[du]
        else:
            zu_basis=self.zm[du]
        if dx not in self.xm:
            self.xm[dx]=_feature_on(self.X,self.xbasis,dx)
        if dx not in self.xproject:
            self.xproject[dx]=np.einsum("abc,mb->amc",
                         self.coef,self.xm[dx],optimize=True)
        result=np.einsum("na,amc->nmc",
                         zu_basis,self.xproject[dx],optimize=True)
        # Standard result: channels FIRST.
        return result.transpose(2,0,1)

    def fields(self,*,fourth=True):
        """Return value, ZZ, ZX, and local derivatives on fitted grid."""
        N,M=len(self.U),len(self.X)
        C=self.coef.shape[-1]
        zz=np.empty((C,N,M,4,4))
        zx=np.empty((C,N,M,4,6))
        zzz=np.empty((C,N,M,4,4,4))
        zzx=np.empty((C,N,M,4,4,6))
        zxx=np.empty((C,N,M,4,6,6))
        for i in range(4):
            for j in range(i,4):
                t=self.evaluate(self.U,(i,j))
                zz[...,i,j]=zz[...,j,i]=t
                for k in range(6):
                    t=self.evaluate(self.U,(i,j),(k,))
                    zzx[...,i,j,k]=zzx[...,j,i,k]=t
            for k in range(6):
                zx[...,i,k]=self.evaluate(self.U,(i,),(k,))
                for l in range(k,6):
                    t=self.evaluate(self.U,(i,),(k,l))
                    zxx[...,i,k,l]=zxx[...,i,l,k]=t
        for t in combinations_with_replacement(range(4),3):
            val=self.evaluate(self.U,t)
            for inds in set(permutations(t)):
                zzz[(...,)+inds]=val
        out=dict(value=self.evaluate(self.U),zz=zz,zx=zx,
                 zzz=zzz,zzx=zzx,zxx=zxx)
        if fourth:
            zzzz=np.empty((C,N,M,4,4,4,4))
            zzxx=np.empty((C,N,M,4,4,6))
            zxxx=np.empty((C,N,M,4,6))
            for t in combinations_with_replacement(range(4),4):
                val=self.evaluate(self.U,t)
                for inds in set(permutations(t)):
                    zzzz[(...,)+inds]=val
            for i in range(4):
                for j in range(i,4):
                    for k in range(6):
                        zzxx[...,i,j,k]=zzxx[...,j,i,k]=self.evaluate(self.U,(i,j),(k,k))
                for k in range(6):
                    zxxx[...,i,k]=self.evaluate(self.U,(i,),(k,k,k))
            out.update(zzzz=zzzz,zzxx=zzxx,zxxx=zxxx)
        return out

    def predict_at(self,U_grid,X_eval=None):
        """Numeric Y predictions for controlled one-coordinate sweeps."""
        if X_eval is not None and not np.array_equal(X_eval,self.X):
            raise ValueError("Sweep must retain fixed X coordinates")
        return self.evaluate(np.asarray(U_grid,dtype=float))


def polynomial_field_shake(engine,A,Z,b,eval_idx,X_eval,*,fourth=True):
    """Fit six Y in rotated U, then compute *local* (not mean) derivatives."""
    fitted,mask,rmse=engine.fit_at(A,Z,b)
    U=np.asarray(Z,dtype=float)@np.asarray(A,dtype=float).T+b
    local=LocalPolynomial(engine,U[eval_idx],X_eval)
    # Preserve original training centering/scales: do not create
    # Features(U_eval,train_ids) from evaluation subjects alone.
    local.zbasis=Features(U,engine.train,engine.degree_u,engine.basis)
    local.U=U[eval_idx]
    local.zm={}
    return local,local.fields(fourth=fourth),fitted,mask,rmse
