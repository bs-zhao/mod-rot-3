"""EXP16: local entropy, fixed X contexts, rotation-invariant V neighborhoods."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as C
import fields as F
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def main(a):
    c=C.load(a.model,a.repeat,a.channel,a.subjects,a.trials); c,b=F.get(c,a.source)
    cs=F.components(b); tx,tt=cs['p' if a.channel=='prob' else 'eta']
    out=C.result_dir('exp16_local_context',a.tag,a.model,a.repeat,a.channel+'_'+a.source)
    x=c['X'].numpy(); xs=(x-x.mean(0))/(x.std(0)+1e-9)
    clusters=KMeans(4,random_state=11,n_init=10).fit_predict(xs)
    _,_,vt=np.linalg.svd(xs,full_matrices=False); pc=xs@vt[0]
    quantiles=np.digitize(pc,np.quantile(pc,[.25,.5,.75]))
    vi=KMeans(4,random_state=11,n_init=10).fit_predict(c['Vtrain'].numpy())
    knn=NearestNeighbors(n_neighbors=max(4,len(xs)//3)).fit(xs)
    anchors=KMeans(4,random_state=11,n_init=10).fit(xs).cluster_centers_
    neighbors=knn.kneighbors(anchors,return_distance=False)
    moments=C.moments(tx,tt)
    groups={'cluster':[C.moments(tx[:,clusters==j],tt[:,clusters==j]) for j in range(4)],
            'quantile':[C.moments(tx[:,quantiles==j],tt[:,quantiles==j]) for j in np.unique(quantiles)],
            'knn':[C.moments(tx[:,ii],tt[:,ii]) for ii in neighbors],
            'Vregion':[C.moments(tx[vi==j],tt[vi==j]) for j in np.unique(vi)]}
    rng=np.random.default_rng(1601); n=tx.shape[0]*tx.shape[1]
    idx=np.sort(rng.choice(n,min(a.local_points,n),replace=False))
    pointx=tx.reshape(n,*tx.shape[-2:])[idx]; pointu=tt.reshape(n,*tt.shape[-2:])[idx]
    active=pointx.square().sum((-2,-1))>1e-20
    pointx=pointx[active]; pointu=pointu[active]
    def stats(A):
        ux,uu=C.energies(A,moments)
        px=torch.einsum('...ia,naj->...nij',A,pointx).square()
        en=px.sum((-2,-1)); nn=C.neff(px)
        local=nn.mean(-1); weighted=(en*nn).sum(-1)/(en.sum(-1)+C.EPS)
        off=uu*(1-torch.eye(uu.shape[-1],dtype=uu.dtype))
        values=[C.neff(ux),C.share(uu),local,weighted,C.neff(off)]
        for name in ('cluster','quantile','knn','Vregion'):
            values.append(torch.stack([C.neff(C.energies(A,m)[0]) for m in groups[name]],-1).mean(-1))
        # First-order gradient context energy, kept separate from Hessian local sparsity.
        g=b['g'] if a.channel=='logit' else b['h'].sigmoid()[...,None]*(1-b['h'].sigmoid()[...,None])*b['g']
        gm=torch.stack([g[:,clusters==j].mean(1) for j in range(4)],1)
        ge=torch.einsum('...ia,sca->...sci',A,gm).square().mean(-3)
        values.append(C.neff(ge))
        return torch.stack(values,-1)
    definitions={'global_baseline':{0:1,1:1},'pointwise_UX':{2:1},'energy_weighted_UX':{3:1},
                 'pointwise_plus_share':{2:1,1:1},'weighted_plus_share':{3:1,1:1},
                 'cluster_plus_share':{5:1,1:1},'quantile_plus_share':{6:1,1:1},
                 'knn_plus_share':{7:1,1:1},'Vregion_plus_share':{8:1,1:1},
                 'cluster_plus_offdiag':{5:1,4:1},'gradient_context':{9:1}}
    methods={}
    for name,weights in definitions.items():
        w=torch.zeros(10,dtype=torch.float64)
        for k,v in weights.items(): w[k]=v
        methods[name]=C.Objective(c['Vtrain'],stats,w)
    if a.only: methods={k:methods[k] for k in a.only.split(',')}
    C.run_methods(c,out,methods,a,dict(source=a.source,cluster=clusters,quantile=quantiles,knn=neighbors,
                   Vregion=vi,local_flat_indices=idx,local_skipped_fraction=float(1-active.double().mean()),
                   zero_energy='fixed raw-field threshold 1e-20, independent of R',
                   local_cost='pointwise terms use fixed local_points; pooled/context terms use all selected points',
                   limitation='Hessian and gradient representations implemented; finite-difference local tensors remain a separate follow-up'))

if __name__=='__main__':
    p=C.parser(__doc__); p.add_argument('--source',choices=['decoder','analytic'],default='decoder')
    p.add_argument('--local-points',type=int,default=128); p.add_argument('--only',default='')
    main(p.parse_args())
