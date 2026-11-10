"""Truncated response-polynomial complexity reconstructed from Hessians only."""
import itertools
import math
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def fit_hessian_taylor(fields,cov,d,degree):
    groups=fields['strict_subject_groups'];ids=np.unique(groups);z=[];hu=[];hx=[]
    for sid in ids:
        mask=groups==sid;z.append(fields['strict_z'][mask][0])
        hu.append(fields['h'][mask,:d,:d]);hx.append(fields['h'][mask,:d,d:])
    z=np.asarray(z);hu=np.asarray(hu);hx=np.asarray(hx)
    scale=max(float(np.sqrt(np.trace(cov)/d)),1e-8)
    polynomial=PolynomialFeatures(degree);phi=polynomial.fit_transform(z/scale);powers=polynomial.powers_
    weights=np.array([math.sqrt(math.factorial(int(p.sum()))/math.prod(math.factorial(int(v)) for v in p)) for p in powers])
    projection=np.linalg.pinv(phi*weights,rcond=1e-10)
    raw_factors=weights/scale**powers.sum(axis=1)
    cu=(projection@hu.reshape(len(z),-1))*raw_factors[:,None]
    cx=(projection@hx.reshape(len(z),-1))*raw_factors[:,None]
    cu=cu.reshape(len(powers),*hu.shape[1:]);cx=cx.reshape(len(powers),*hx.shape[1:])
    first=cx[0];second=cu[0];third=np.zeros((len(first),d,d,d))
    for k in range(d):
        target=np.eye(d,dtype=int)[k];index=np.flatnonzero(np.all(powers==target,axis=1))[0]
        third[:,:,:,k]=cu[index]
    third=sum(third.transpose((0,)+tuple(i+1 for i in perm)) for perm in itertools.permutations(range(3)))/6
    reference=max(float(np.linalg.norm(first)+np.linalg.norm(second)),1e-20)
    if np.linalg.norm(third)<1e-12*reference:third*=0
    return first,second,third


def unique_indices(d,order):
    choices=list(itertools.combinations_with_replacement(range(d),order))
    indices=np.asarray(choices).T
    factors=np.array([math.prod(math.factorial(ix.count(i)) for i in set(ix)) for ix in choices])
    return indices,factors


def taylor_features(tensors,a,order):
    first,second,third=tensors;d=len(a)
    linear=np.einsum('ia,xap->xip',a,first,optimize=True)
    quadratic=np.einsum('ia,xab,jb->xij',a,second,a,optimize=True)
    ix,factors=unique_indices(d,2)
    values=[np.sqrt(np.mean(linear*linear,axis=0)).reshape(-1),np.sqrt(np.mean(quadratic[:,ix[0],ix[1]]**2,axis=0))/factors]
    if order==3:
        cubic=np.einsum('ia,jb,kc,xabc->xijk',a,a,a,third,optimize=True)
        ix,factors=unique_indices(d,3)
        values.append(np.sqrt(np.mean(cubic[:,ix[0],ix[1],ix[2]]**2,axis=0))/factors)
    return np.concatenate(values)
