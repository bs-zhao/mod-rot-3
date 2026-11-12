"""Optimistic exact-matrix lower bound for pooled worst-X soft support.

For each PSD H, diag(Q.T H Q) is a doubly stochastic average of its
eigenvalues. Since t/(t+eps^2) is concave, its summed value on any such
diagonal is at least its summed value on the eigenvalues. Taking max over
X yields a lower bound even when no common eigenbasis exists.
"""
import numpy as np


def normalized_matrices(covariance):
    h=np.asarray(covariance,dtype=float)
    h=(h+h.transpose(0,2,1))/2
    trace=np.trace(h,axis1=1,axis2=2)
    return np.divide(h,trace[:,None,None],out=np.zeros_like(h),where=trace[:,None,None]>0)


def score(normalized,q,eps=.03):
    energy=np.maximum(np.einsum('ai,jab,bi->ij',q,normalized,q,optimize=True),0.)
    return float(np.max(np.sum(energy/(energy+eps**2),axis=0)))


def lower_bound(normalized,eps=.03):
    eigen=np.linalg.eigvalsh(normalized)
    if eigen.min() < -1e-9:raise ArithmeticError('The frozen gradient covariance is not PSD')
    eigen=np.maximum(eigen,0.)
    by_x=np.sum(eigen/(eigen+eps**2),axis=1)
    return float(np.max(by_x)),eigen,by_x


def fine_plane_grid(normalized,reference,eps=.03):
    rows=[];d=len(reference)
    for i in range(d):
        for j in range(i+1,d):
            for angle in range(-45,46):
                change=np.eye(d);r=np.deg2rad(angle);c,s=np.cos(r),np.sin(r)
                change[i,i]=change[j,j]=c;change[i,j]=-s;change[j,i]=s
                rows.append(dict(U_i=i+1,U_j=j+1,angle_deg=angle,loss=score(normalized,reference@change,eps)))
    return rows
