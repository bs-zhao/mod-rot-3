"""Analytic CR3 local Hessian and higher-order derivatives, independent of NN.

Second derivatives: original exact formula from math/04/../oracle.py.
Third/fourth derivatives: CENTRAL FINITE DIFFERENCES OF THOSE
EXACT ANALYTIC HESSIANS (not finite differences of NN predictions).
The finite-difference steps are recorded, and convergence must be
checked at alternative step sizes before interpreting fourth-order values.

All arrays have channel axis: [analytic E[Choice]=p, analytic Logit, p].
Once the original T-coordinate fields are built, any pair rotation is
a cheap EXACT tensor coordinate transformation.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"math"/"04_cr3_zx_hessian_threeway"))
from oracle import oracle_hessians,oracle_outputs  # noqa: E402


def _channels(t,x):
    Heta,Hp=oracle_hessians(t,x)
    return np.stack((Hp,Heta,Hp),axis=0)


def reference_fields(T,X,*,fourth=True,tsteps=None,xsteps=None):
    """Build local analytic tensors on strictly interior X bank.

    X monetary entries must be >0, and probabilities must be interior.
    The caller must sample the SAME X subset for all three fitted paths.
    """
    T=np.asarray(T,dtype=float);X=np.asarray(X,dtype=float)
    if T.ndim!=2 or T.shape[1]!=4 or X.ndim!=2 or X.shape[1]!=6:
        raise ValueError("Expected T=(N,4), X=(M,6)")
    if np.any(X[:,[0,1,3,4]]<=0) or np.any((X[:,[2,5]]<=.002)|(X[:,[2,5]]>=.998)):
        raise ValueError("Analytic local derivatives require positive money/interior p")
    # Use fixed reference-scale steps if supplied, not angle-dependent
    # steps, to make convergence diagnostics meaningful.
    ht=np.asarray(tsteps if tsteps is not None
                  else np.maximum(np.std(T,axis=0)*.002,1e-4),dtype=float)
    hx=np.asarray(xsteps if xsteps is not None
                  else np.maximum(np.std(X,axis=0)*.0007,1e-5),dtype=float)
    if ht.shape!=(4,) or hx.shape!=(6,) or np.any(ht<=0) or np.any(hx<=0):
        raise ValueError("Invalid FD step sizes")
    if np.any(X[:,[0,1,3,4]]<2*hx[[0,1,3,4]]):
        raise ValueError("Money too near zero to safely finite-difference")
    if np.any(X[:,[2,5]]<2*hx[[2,5]]) or np.any(X[:,[2,5]]>1-2*hx[[2,5]]):
        raise ValueError("Probability too near boundary")
    base=_channels(T,X)
    C,N,M=base.shape[:3]
    zzz=np.zeros((C,N,M,4,4,4))
    zzx=np.zeros((C,N,M,4,4,6))
    zxx=np.zeros((C,N,M,4,6,6))
    Hzt=base[..., :4,:4]
    Htx=base[..., :4,4:]
    cached_t={}
    cached_x={}
    def shift_t(k,delta):
        key=(k,delta)
        if key not in cached_t:
            p=T.copy();p[:,k]+=delta*ht[k]
            cached_t[key]=_channels(p,X)
        return cached_t[key]
    def shift_x(k,delta):
        key=(k,delta)
        if key not in cached_x:
            p=X.copy();p[:,k]+=delta*hx[k]
            cached_x[key]=_channels(T,p)
        return cached_x[key]
    for k in range(4):
        zzz[...,k]=(shift_t(k,+1)[...,:4,:4]-
                    shift_t(k,-1)[...,:4,:4])/(2*ht[k])
    for k in range(6):
        pair=(shift_x(k,+1)-shift_x(k,-1))/(2*hx[k])
        zzx[...,k]=pair[...,:4,:4]
        zxx[...,k]=pair[...,:4,4:]
    # H_{zi,xk,xl}: derivatives commute; symmetrize the last 2 axes.
    zxx=.5*(zxx+np.swapaxes(zxx,-1,-2))
    # The analytic higher derivative tensors must be symmetric in
    # repeated Z indices. Finite differences introduce small mismatch.
    # Symmetrize 4x4x4 over all 6 permutations.
    import itertools
    zzz=sum(np.transpose(zzz,(0,1,2)+tuple(3+i for i in perm))
            for perm in itertools.permutations(range(3)))/6.
    eta,p=oracle_outputs(T,X)
    values=np.stack((p,eta,p),axis=0)
    out=dict(value=values,zz=base[...,:4,:4],zx=base[...,:4,4:],
             zzz=zzz,zzx=zzx,zxx=zxx)
    if fourth:
        zzzz=np.zeros((C,N,M,4,4,4,4))
        zzxx=np.zeros((C,N,M,4,4,6))
        zxxx=np.zeros((C,N,M,4,6))
        for j in range(4):
            for k in range(j,4):
                if j==k:
                    sec=(shift_t(j,+1)[...,:4,:4]-2*Hzt+
                         shift_t(j,-1)[...,:4,:4])/(ht[j]**2)
                else:
                    def cross(a,b):
                        p=T.copy()
                        p[:,j]+=a*ht[j];p[:,k]+=b*ht[k]
                        return _channels(p,X)[...,:4,:4]
                    sec=(cross(+1,+1)-cross(+1,-1)-
                         cross(-1,+1)+cross(-1,-1))/(4*ht[j]*ht[k])
                zzzz[...,j,k]=zzzz[...,k,j]=sec
        # For a fourth derivative, symmetry between the first two and
        # last two pairs holds only approximately under finite differencing.
        # Symmetrize all 24 permutations explicitly.
        zzzz=sum(np.transpose(zzzz,(0,1,2)+tuple(3+i for i in perm))
                 for perm in itertools.permutations(range(4)))/24.
        for k in range(6):
            plus=shift_x(k,+1)
            minus=shift_x(k,-1)
            second=(plus-2*base+minus)/(hx[k]**2)
            zzxx[...,k]=second[...,:4,:4]
            zxxx[...,k]=second[...,:4,4+k]
        out.update(zzzz=zzzz,zzxx=zzxx,zxxx=zxxx)
    return out,dict(tstep=ht,xstep=hx,method="analytic_H_central_finite_difference")


def rotate_fields(star,R):
    """Map local derivatives from T_hat into U=R(T_hat-b)+b.

    The inverse relation is T_hat=b+R.T@(U-b), so tensor derivatives
    covariantly transform with R on all Z parameter axes.
    X axes are unchanged.
    """
    R=np.asarray(R,dtype=float)
    if R.shape!=(4,4) or not np.allclose(R.T@R,np.eye(4),atol=1e-9):
        raise ValueError("Expected a 4D orthogonal rotation")
    out={
        "value":star["value"],
        "zz":np.einsum("ia,jb,qn mab->qn mij",R,R,star["zz"],optimize=True),
        "zx":np.einsum("ia,qn mak->qn mik",R,star["zx"],optimize=True),
        "zzz":np.einsum("ia,jb,kc,qnmabc->qnmijk",R,R,R,star["zzz"],optimize=True),
        "zzx":np.einsum("ia,jb,qnmabd->qnmijd",R,R,star["zzx"],optimize=True),
        "zxx":np.einsum("ia,qnmabc->qnmibc",R,star["zxx"],optimize=True),
    }
    if "zzzz" in star:
        out["zzzz"]=np.einsum("ia,jb,kc,ld,qnmabcd->qnmijkl",
                            R,R,R,R,star["zzzz"],optimize=True)
        out["zzxx"]=np.einsum("ia,jb,qnmabk->qnmijk",
                            R,R,star["zzxx"],optimize=True)
        out["zxxx"]=np.einsum("ia,qnm ak->qn mik",R,star["zxxx"],optimize=True)
    return out
