"""Matched 8x8 autograd limit: separates pairing/sampling/search from finite h."""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as C
from run import search

class Adapter:
    def __init__(self,obj,d): self.obj=obj; self.d=d; self.calls=0
    def raw(self,R): return self.obj.raw(R)

def main(a):
    for model in C.COND:
        for rep in (1,2):
            for ch in ('prob','logit'):
                c=C.load(model,rep,ch,a.subjects,a.trials); d=c['Vtrain'].shape[1]
                out=C.result_dir('exp15_finite_scale',a.tag,model,rep,ch)
                if (out/'COMPLETE.json').exists(): continue
                x=c['X'].numpy(); xs=(x-x.mean(0))/(x.std(0)+1e-9)
                dist=cdist(xs,xs); dist[dist<1e-10]=np.inf; pair=dist.argmin(1)
                norm=dist[np.arange(len(x)),pair]; direction=C.tensor((xs[pair]-xs)/norm[:,None])
                g=C.tensor(c['g']); chord=(g[:,pair]-g)/C.tensor(norm)[None,:,None]
                def fn(A):
                    cg=torch.einsum('...ia,ska->...ski',A,chord)
                    # A[..., S,K,i] -> [..., S,K,i,j], mean over S,K.
                    ux=(cg[..., :,None]*direction[:,None,:]).square().mean((-4,-3))
                    uu=C.energies(A,mm)[1]
                    return torch.stack([C.neff(ux),C.share(uu)],-1)
                mm=C.moments(c['Htx'],c['Htt'])
                pair_obj=C.Objective(c['Vtrain'],fn)
                ambient=C.Objective(c['Vtrain'],lambda A:C.terms(A,mm))
                rows=[]; saved={}; diag={}; pcs=[]
                for label,obj,algo in [('pair_Givens',pair_obj,'Givens'),('pair_Adam',pair_obj,'Adam'),
                                       ('ambient_Givens',ambient,'Givens'),('ambient_Adam',ambient,'Adam')]:
                    t=time.perf_counter()
                    if algo=='Givens':
                        ll,rr,hist=search(Adapter(obj,d),obj.scales,torch.ones(2,dtype=torch.float64),a.starts,a.steps,a.seed)
                    else: ll,rr=C.optimize(obj,d,a.starts,a.steps,a.seed)
                    saved[label+'__R']=rr; saved[label+'__loss']=ll
                    saved[label+'__axis_sd']=np.std(c['V'][c['ids']]@rr[0].T,axis=0)
                    np.savez_compressed(out/'rotations.npz',**saved)
                    met=C.evaluate(c,rr[0]); dd,pp,rv=C.diagnose(obj,c,rr[0],a.random_n)
                    diag[label]={**dd,'correlation':met['correlation'],'matched_correlations':met['matched_correlations']}
                    pcs.extend({**p,'method':label} for p in pp)
                    rows.append(dict(model=model,repeat=rep,channel=ch,method=label,loss=ll[0],starts=a.starts,
                                     steps=a.steps,centers=len(c['ids'])*len(c['X']),seconds=time.perf_counter()-t,
                                     **{k:met[k] for k in ('mcc','leakage','max_leakage')}))
                pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
                pd.DataFrame(pcs).to_csv(out/'plane_curves.csv',index=False)
                C.dump(out/'diagnostics.json',diag)
                C.dump(out/'config.json',dict(**vars(a),environment=C.environment(),ids=c['ids'],X=c['X'],
                                             pair=pair,paired_direction=direction,evidence='decoder-autograd matched limit',
                                             note='same 8x8 sample and bank chords as finite-scale run; separates search/pairing effects'))
                C.dump(out/'COMPLETE.json',dict(rows=len(rows))); print(model,rep,ch,rows,flush=True)

if __name__=='__main__':
    p=C.parser(__doc__); p.set_defaults(subjects=8,trials=8,tag='matched_limit'); main(p.parse_args())
