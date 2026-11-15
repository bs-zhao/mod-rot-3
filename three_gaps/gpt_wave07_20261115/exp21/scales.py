"""Observed-bank resolution, direction rank, support; no role/design labels."""
import numpy as np
from scipy.spatial import cKDTree

def resolution(X):
    sd=X.std(0);Z=(X-X.mean(0))/sd;tree=cKDTree(Z)
    gap=[]
    for j in range(X.shape[1]):
        v=np.diff(np.unique(X[:,j]));v=v[v>0]
        if not len(v):raise ValueError('constant X column, ambient derivative unsupported')
        gap.append(np.quantile(v,[.1,.5,.9]))
    gap=np.asarray(gap)
    dist,idx=tree.query(Z,k=min(13,len(X)))
    directions=Z[idx[:,1:]]-Z[:,None,:]
    sv=np.linalg.svd(directions,compute_uv=False)
    rank=(sv>sv[:,0,None]*1e-8).sum(1)
    return dict(sd=sd,gap_quantiles=gap,knn={str(k):np.quantile(dist[:,k],[.1,.5,.9]) for k in [1,3,5,10]},
        direction_rank=rank,direction_condition=sv[:,0]/np.maximum(sv[:,-1],1e-30),neighbours=idx[:,1:])

def support(X,hX):
    sd=X.std(0);Z=(X-X.mean(0))/sd;tree=cKDTree(Z)
    knn=tree.query(Z,k=2)[0][:,1];q95=np.quantile(knn,.95)
    rows=[]
    for j in range(X.shape[1]):
        for sign in [-1,1]:
            q=X.copy();q[:,j]+=sign*hX[j]
            dist=tree.query((q-X.mean(0))/sd)[0]
            rows.append(dict(column=j,sign=sign,halfwidth=float(hX[j]),end_to_end=float(2*hX[j]),
                range_coverage=float(((q[:,j]>=X[:,j].min())&(q[:,j]<=X[:,j].max())).mean()),
                knn_distance_quantiles=np.quantile(dist,[.5,.9,.99]).tolist(),within_observed_nn95=float((dist<=q95).mean())))
    return rows

def chord_design(X,k=12,condition_cap=1e5):
    sd=X.std(0);Z=(X-X.mean(0))/sd;_,idx=cKDTree(Z).query(Z,k=k+1)
    nb=idx[:,1:];D=Z[nb]-Z[:,None,:];length=np.linalg.norm(D,axis=-1)
    unit=D/length[:,:,None];sv=np.linalg.svd(unit,compute_uv=False)
    rank=(sv>sv[:,0,None]*1e-8).sum(-1);cond=sv[:,0]/sv[:,-1]
    if np.any(rank<X.shape[1]) or np.any(cond>condition_cap):
        raise ValueError(f'chord full UX reconstruction rejected: rank min={rank.min()}, condition max={cond.max():.4g}, cap={condition_cap}')
    return nb,length,np.linalg.pinv(unit),rank,cond
