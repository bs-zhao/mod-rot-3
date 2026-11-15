"""EXP17A/C: grouped structure splits, T-free landscape reliability weights."""
from __future__ import annotations
import itertools
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as C

def distance(a,b):
    cost=np.abs(a@b.T); i,j=linear_sum_assignment(-cost)
    return float(np.sqrt(max(0,2*len(a)-2*cost[i,j].sum())))

def main(a):
    c=C.load(a.model,a.repeat,a.channel,a.subjects,a.trials)
    out=C.result_dir('exp17_reliability',a.tag,a.model,a.repeat,a.channel)
    S,K=c['Htt'].shape[:2]; d=c['Vtrain'].shape[1]
    pooled=C.moments(c['Htx'],c['Htt']); cv=C.cov(c['Vtrain'])
    # Separate common Haar sets: calibration and reliability-check diagnostic rotations.
    cal=C.rotations(d,200,1); check=C.rotations(d,a.random_n,91)
    Ac=C.effective(cal,cv); Ae=C.effective(check,cv)
    methods={'equal':C.Objective(c['Vtrain'],lambda A:C.terms(A,pooled))}; diagnostics={}; saved={}
    for axis,count in itertools.product(('subjects','trials'),(2,4,8)):
        n=S if axis=='subjects' else K
        if n<count*2: continue
        ids=np.array_split(np.random.default_rng(1701).permutation(n),count)
        ms=[C.moments(c['Htx'][ii] if axis=='subjects' else c['Htx'][:,ii],
                      c['Htt'][ii] if axis=='subjects' else c['Htt'][:,ii]) for ii in ids]
        with torch.no_grad():
            values=torch.stack([C.terms(Ac,m) for m in ms],1).numpy() # Haar, split, term
            checks=torch.stack([C.terms(Ae,m) for m in ms],1).numpy()
        rho=[]; detail=[]
        for t in range(2):
            prs=[]; spr=[]; low=[]
            for i,j in itertools.combinations(range(count),2):
                if min(values[:,i,t].std(),values[:,j,t].std())<1e-9:
                    prs.append(0.); spr.append(0.); low.append(0.); continue
                prs.append(float(np.corrcoef(values[:,i,t],values[:,j,t])[0,1]))
                spr.append(float(spearmanr(checks[:,i,t],checks[:,j,t]).statistic))
                keep=checks[:,i,t]<=np.quantile(checks[:,i,t],.2)
                low.append(float(spearmanr(checks[keep,i,t],checks[keep,j,t]).statistic))
            rho.append(max(0,float(np.mean(prs))))
            detail.append(dict(pearson=prs,spearman_independent=spr,low_tail_spearman=low))
        tag=f'{axis}{count}'
        if sum(rho)>0:
            methods[tag+'_rho']=C.Objective(c['Vtrain'],lambda A:C.terms(A,pooled),rho)
        sc=torch.tensor(values.std(0).clip(1e-9),dtype=torch.float64)
        def ens(A,ms=ms,sc=sc):
            v=torch.stack([C.terms(A,m) for m in ms],-2)/sc
            # Mean + fixed 1 SD; no T-based tuning.
            return (v.mean(-2)+v.std(-2,unbiased=False))
        methods[tag+'_meanSD']=C.Objective(c['Vtrain'],ens)
        endpoints=[]; endvalues=[]
        for i,m in enumerate(ms):
            obj=C.Objective(c['Vtrain'],lambda A,m=m:C.terms(A,m))
            ls,rs=C.optimize(obj,d,a.starts,a.steps,a.seed)
            endpoints.append(rs[0]); endvalues.append(ls[0])
        cross=np.array([[float(C.Objective(c['Vtrain'],lambda A,m=m:C.terms(A,m))(C.tensor(r)))
                         for m in ms] for r in endpoints])
        diagnostics[tag]=dict(rho=rho,landscape=detail,split_ids=ids,cross_subset_loss=cross,
                              signed_permutation_distances=[[distance(r,s) for s in endpoints] for r in endpoints],
                              no_signal=bool(sum(rho)==0),
                              caveat='Shared decoder bootstrap cannot exclude systematic decoder error')
        saved[tag+'_rotations']=np.array(endpoints); saved[tag+'_calibration']=values
        saved[tag+'_check']=checks
    np.savez_compressed(out/'split_landscapes.npz',**saved); C.dump(out/'reliability.json',diagnostics)
    if a.only: methods={k:methods[k] for k in a.only.split(',')}
    C.run_methods(c,out,methods,a,dict(weight_rule='max(0,mean pairwise calibration Pearson); SD floor 1e-9',
                   penalty_lambda=1,weight_selection_uses_T=False,source='shared frozen decoder'))

if __name__=='__main__':
    p=C.parser(__doc__); p.add_argument('--only',default=''); main(p.parse_args())
