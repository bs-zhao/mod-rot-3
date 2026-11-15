"""Fixed physical Gaussian kernel, nested antithetic RQMC; both averaging orders."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as c
import nodes
import numpy as np
import torch

def offset_fields(st,V,X,delta,query_batch=1):
    """Batch identical physical nodes for throughput; never mix their Hessians."""
    if query_batch<1:raise ValueError('query_batch must be positive')
    for start in range(0,len(delta),query_batch):
        shifts=delta[start:start+query_batch]
        if len(shifts)==1:
            yield c.fields(st,V+shifts[0],X)
        else:
            B,G,K=len(V),len(shifts),len(X)
            grid=(V[:,None,:]+shifts[None,:,:]).reshape(B*G,st.d)
            f=c.fields(st,grid,X)
            tt=f['Htt'].reshape(B,G,K,st.d,st.d)
            tx=f['Htx'].reshape(B,G,K,st.d,st.p)
            hh=f['h'].reshape(B,G,K)
            for j in range(G):yield dict(Htt=tt[:,j],Htx=tx[:,j],h=hh[:,j])

def smooth(st,V,X,sigmas,inner,dest,seed=20261115,kernel='cov',weights=None,query_batch=1):
    """Compute all scale fields on each block before release, including cross terms.

    Snapshots at nested inner budgets permit quadrature convergence checks.
    No field can be rotated as a finite-difference tensor: these are true Hessians
    of fixed convolutions. A signed multiscale mixture is accumulated BEFORE squaring.
    """
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    if weights is None:
        mixture=[s for s in [.1,.25,.5] if s in sigmas]
        weights={s:1/len(mixture) for s in mixture} if mixture else {s:1/len([v for v in sigmas if v>0]) for s in sigmas if s>0}
    cfg=dict(V=c.arrsha(V),X=c.arrsha(X),sigmas=list(sigmas),inner=list(inner),seed=seed,kernel=kernel,source=st.source,
        mixture_weights={str(k):v for k,v in weights.items()})
    if query_batch!=1:cfg['query_batch']=query_batch
    cp=dest/'config.json'
    if cp.exists():
        import json
        if json.loads(cp.read_text())!=cfg:raise ValueError('smoothing configuration mismatch')
    c.dump(cp,cfg)
    eps=nodes.antithetic(max(inner),st.d,seed)
    L=np.linalg.cholesky(st.C) if kernel=='cov' else np.eye(st.d)*np.sqrt(np.trace(st.C)/st.d)
    delta=eps@L.T;save=dict(offsets=delta,inner=np.asarray(inner),sigmas=np.asarray(sigmas))
    c.save_npz(dest/'nodes.npz',**save)
    acc={}; k=len(X);batch=8
    for bidx in range(0,len(V),batch):
        bp=dest/f'block_{bidx:06d}.npz'
        if not bp.exists():
            v=V[bidx:bidx+batch];zero=c.fields(st,v,X)
            ztt,ztx=zero['Htt'].double(),zero['Htx'].double()
            zsx,zt4=c.moments(ztt,ztx);block={}
            signed={n:[torch.zeros_like(ztt),torch.zeros_like(ztx)] for n in inner}
            for sigma in sigmas:
                ts=torch.zeros_like(ztt);xs=torch.zeros_like(ztx)
                se=np.zeros_like(zsx);te=np.zeros_like(zt4);ps=torch.zeros_like(zero['h'],dtype=torch.float64)
                iterator=offset_fields(st,v,X,sigma*delta,query_batch) if sigma else iter([zero])
                for j,f in enumerate(iterator):
                    tt,tx=f['Htt'].double(),f['Htx'].double();ts+=tt;xs+=tx;ps+=f['h'].double()
                    s,t=c.moments(tt,tx);se+=s;te+=t
                    n=j+1
                    if n in inner or sigma==0:
                        ss,tt4=c.moments(ts/n,xs/n);prefix=f's{sigma:g}_n{n}'
                        block[prefix+'_SX_s']=ss;block[prefix+'_T4_s']=tt4
                        block[prefix+'_SX_e']=se/n;block[prefix+'_T4_e']=te/n
                        block[prefix+'_p']=ps.cpu().numpy()/n
                        if sigma in weights:
                            signed[n][0]+=weights[sigma]*ts/n
                            signed[n][1]+=weights[sigma]*xs/n
                print('smooth block',bidx,'sigma',sigma,'inner',max(inner),flush=True)
            for n,(tt,tx) in signed.items():
                s,t=c.moments(tt,tx);block[f'mix_n{n}_SX']=s;block[f'mix_n{n}_T4']=t
            block['p0']=zero['h'].cpu().numpy()
            c.save_npz(bp,**block)
        with np.load(bp) as z:
            for key in z.files:acc.setdefault(key,[]).append(z[key])
    res={k:np.concatenate(v) for k,v in acc.items()}
    c.save_npz(dest/'moments.npz',**res)
    c.dump(dest/'COMPLETE.json',dict(sha256=c.sha(dest/'moments.npz'),outer_nodes=len(V)*len(X),inner=max(inner)))
    return res
