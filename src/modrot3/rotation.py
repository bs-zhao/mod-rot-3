"""Post-hoc orthogonal rotation module for already-frozen NP Z,X,Y.

All rotation optimization uses only frozen z, observed X and observed Y.
Theta is imported ONLY after R/A have been persisted, for held-out audits.
Add future objectives to OBJECTIVES below without changing baseline data/NP.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize,linear_sum_assignment
from sklearn.decomposition import PCA

from .pipeline import ROOT, dataset_path, sha256
from modlearn_rot.generators.dataset import load_dataset


def objective_identity(q:np.ndarray, x, y, mask, R:np.ndarray)->float:
    return float(np.linalg.norm(R-np.eye(R.shape[0]))**2)


def objective_kurtosis(q:np.ndarray, x, y, mask, R:np.ndarray)->float:
    """Fourth-cumulant concentration, an unsupervised distribution-only control."""
    u=q@R.T
    center=u-u.mean(0)
    variance=np.maximum(np.mean(center**2,axis=0),1e-10)
    excess=np.mean(center**4,axis=0)/variance**2-3
    return -float(np.sum(excess**2))


def objective_quartic(q:np.ndarray,x,y,mask,R:np.ndarray)->float:
    """Quartic concentration control on whitened latent axes."""
    u=q@R.T
    return -float(np.sum(np.mean(u**4,axis=0)))


OBJECTIVES={
    'identity': objective_identity,
    'kurtosis': objective_kurtosis,
    'quartic': objective_quartic,
}


def skew_matrix(angles:np.ndarray,k:int)->np.ndarray:
    E=np.zeros((k,k));i,j=np.triu_indices(k,1)
    E[i,j]=angles;E[j,i]=-angles
    return E


def optimize(q,x,y,mask,method,starts=4,maxiter=200,seed=31):
    k=q.shape[1]
    if method=='identity':
        return np.eye(k),dict(loss=0.,starts=0,converged=True)
    score=OBJECTIVES[method]
    rng=np.random.default_rng(seed)
    count=k*(k-1)//2
    seeds=[np.zeros(count)]
    seeds.extend(rng.normal(0,.35,size=count) for _ in range(max(0,starts-1)))
    results=[]
    for vec in seeds:
        res=minimize(lambda a:score(q,x,y,mask,expm(skew_matrix(a,k))),
                     vec,method='Powell',options=dict(maxiter=maxiter,xtol=1e-6,ftol=1e-7))
        results.append((float(res.fun),expm(skew_matrix(res.x,k)),bool(res.success)))
    val,R,ok=min(results,key=lambda item:item[0])
    if not np.all(np.isfinite(R)) or not np.allclose(R@R.T,np.eye(k),atol=1e-7):
        raise RuntimeError('Nonorthogonal rotation')
    return R,dict(loss=val,starts=len(results),converged=ok)


def corr(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    a=a-a.mean(0);b=b-b.mean(0)
    den=np.sqrt(np.sum(a*a,axis=0)[:,None]*np.sum(b*b,axis=0)[None,:])
    return np.divide(a.T@b,den,out=np.zeros((a.shape[1],b.shape[1])),where=den>1e-12)


def evaluate_after_freeze(theta,z,u,test,folder,method,labels):
    C=corr(theta[test],u[test])
    # Rows are true parameters; columns are latent axes.
    rows,cols=linear_sum_assignment(-C*C)
    match={int(r):int(c) for r,c in zip(rows,cols)}
    chosen=np.array([C[i,match[i]]**2 for i in range(len(labels))])
    mask=np.ones_like(C,bool)
    for i,j in match.items():mask[i,j]=False
    report=dict(method=method,matched_mean_r2=float(np.mean(chosen)),
                matched_min_r2=float(np.min(chosen)),
                parameter_r2={labels[i]:float(chosen[i]) for i in range(len(labels))},
                max_offtarget_abs_r=float(np.max(np.abs(C[mask]))) if mask.any() else 0.,
                mean_offtarget_r2=float(np.mean(C[mask]**2)) if mask.any() else 0.,
                matching=match,theta_axis_corr=C.tolist())
    (folder/'rotation__test_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(max(6,C.shape[1]),4.4))
    m=ax.imshow(C*C,vmin=0,vmax=1,cmap='Blues')
    ax.set_yticks(range(len(labels)),labels)
    ax.set_xticks(range(C.shape[1]),[f'u{j+1}' for j in range(C.shape[1])])
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            ax.text(j,i,f'{C[i,j]**2:.2f}',ha='center',va='center',fontsize=9,
                    color='white' if C[i,j]**2>.55 else 'black')
    for i,j in match.items():
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,ec='red',lw=2))
    fig.colorbar(m,ax=ax,label='TEST Pearson r²')
    fig.tight_layout();fig.savefig(folder/'rotation__test_corr2.png',dpi=170);plt.close(fig)
    return report


def aligned_observations(data):
    """Return (N,T,D), (N,T), (N,T) arrays aligned to each subject's Z.

    Original observations are unchanged; shorter C-series trial histories are
    zero padded with a False mask and must never contribute to objective loss.
    """
    counts=np.asarray(data.trial_counts,dtype=int)
    n=len(counts);t=int(counts.max());d=int(data.x.shape[1])
    x=np.zeros((n,t,d),dtype=np.float32)
    y=np.zeros((n,t),dtype=np.int64)
    mask=np.zeros((n,t),dtype=bool)
    offsets=np.r_[0,np.cumsum(counts)]
    if offsets[-1]!=len(data.x):
        raise ValueError('Subject trial counts disagree with data')
    if not np.array_equal(np.asarray(data.subject_index),np.repeat(np.arange(n),counts)):
        raise ValueError('Rotation expects preserved contiguous subject trial order')
    for sid in range(n):
        a,b=int(offsets[sid]),int(offsets[sid+1])
        x[sid,:counts[sid]]=data.x[a:b]
        y[sid,:counts[sid]]=np.asarray(data.y[a:b]).reshape(-1)
        mask[sid,:counts[sid]]=True
    return x,y,mask


def main(argv=None):
    p=argparse.ArgumentParser(description='Independent frozen-ZXY rotation objectives')
    p.add_argument('--model',choices=['C1','C2','C3','CR3'],required=True)
    p.add_argument('--condition',required=True,help='S3/S4 condition tag, e.g. beta0_lambda0_ctx0p25to0p8_L8')
    p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--representation',choices=['pca','full'],default='pca')
    p.add_argument('--rotation-dim',type=int,default=None)
    p.add_argument('--method',choices=tuple(OBJECTIVES),default='identity')
    p.add_argument('--starts',type=int,default=4)
    p.add_argument('--maxiter',type=int,default=200)
    p.add_argument('--seed',type=int,default=31)
    p.add_argument('--fit-max-subjects',type=int,default=2500)
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args(argv)
    root=ROOT/'outputs'/'choice'/a.model
    src=root/'s4_evaluate'/a.condition/f'repeat{a.repeat:02d}'/'s4_evaluate__latent.npz'
    if not src.exists():raise FileNotFoundError(f'Run baseline extract first: {src}')
    with np.load(src,allow_pickle=False) as f:
        z=np.array(f['z'],float)
        tr=np.array(f['train_ids'],int);te=np.array(f['test_ids'],int)
        if str(f['context_mode']) != 'all_observed_trials':
            raise RuntimeError('Rotation requires Z encoded using all observed trials')
    data=load_dataset(dataset_path(a.model))
    # X,Y are grouped by subject and aligned to the same IDs as rows of Z.
    x,y,mask=aligned_observations(data)
    k = a.rotation_dim or {'C1':4,'C2':4,'C3':6,'CR3':4}[a.model]
    if k>z.shape[1] or k<2:raise ValueError('rotation-dim must be 2..latent_dim')
    pca=PCA(n_components=k,svd_solver='full').fit(z[tr])
    if a.representation=='full' and k!=z.shape[1]:
        raise ValueError('Full representation requires rotation-dim == native latent_dim')
    floor=1e-5
    W=pca.components_/np.sqrt(np.maximum(pca.explained_variance_,floor))[:,None]
    q=(z-pca.mean_)@W.T
    chosen=tr if len(tr)<=a.fit_max_subjects else np.random.default_rng(a.seed).choice(tr,a.fit_max_subjects,replace=False)
    if a.fit_max_subjects<1:raise ValueError('--fit-max-subjects must be >=1')
    # Every metric receives subject-aligned (U,X,Y,mask) with the SAME rows.
    # True theta is deliberately unavailable during optimize().
    R,stats=optimize(q[chosen],x[chosen],y[chosen],mask[chosen],
                     a.method,a.starts,a.maxiter,a.seed)
    A=R@W
    offset=-A@pca.mean_
    u=(z@A.T)+offset
    folder=root/'rotations'/a.method/a.condition/f'repeat{a.repeat:02d}'
    if folder.exists() and (folder/'rotation__matrix.npz').exists() and not a.overwrite:
        raise FileExistsError('Rotation already exists; use a different method/seed or --overwrite')
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'rotation__matrix.npz',R=R,A=A,offset=offset,
                        pca_components=pca.components_,pca_eigenvalues=pca.explained_variance_,
                        pca_mean=pca.mean_,fit_ids=chosen,source_z_sha256=np.array(sha256(src)),
                        optimization_method=np.array(a.method))
    np.savez_compressed(folder/'rotation__U.npz',U=u,train_ids=tr,test_ids=te)
    (folder/'rotation__fit.json').write_text(json.dumps(dict(model=a.model,method=a.method,
        representation=a.representation,rotation_dim=k,fit_subjects=len(chosen),
        seed=a.seed,source_z_sha256=sha256(src),**stats),indent=2),encoding='utf-8')
    # Only postfit accesses true parameters.
    labels=(['alpha','lambda_loss','gamma','tau'] if a.model=='CR3'
            else list(__import__('modlearn_rot.generators.registry',fromlist=['get_model']).get_model(a.model).parameter_names))
    report=evaluate_after_freeze(np.asarray(data.parameters),z,u,te,folder,a.method,labels)
    print(f'{a.model}: {a.method} TEST mean matched r²={report["matched_mean_r2"]:.4f};',folder)


if __name__=='__main__':
    main()