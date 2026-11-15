"""Geometric support proxy for every saved EXP15 solution, without NN queries."""
import itertools
import json
import numpy as np
import pandas as pd
import torch
import core as C

def coverage(R,V,covariance,inverse,threshold):
    A=C.effective(C.tensor(R),covariance); center=C.tensor(V)
    result={}
    for h in (.1,.25,.5,1.):
        shifts=[torch.zeros_like(A[0])]+[s*h*A[i] for i in range(len(A)) for s in (1,-1)]
        for i,j in itertools.combinations(range(len(A)),2):
            shifts += [h*(si*A[i]+sj*A[j]) for si,sj in ((1,1),(1,-1),(-1,1),(-1,-1))]
        points=center[None]+torch.stack(shifts)[:,None]
        distance=torch.einsum('...i,ij,...j->...',points,inverse,points)
        result[str(h)]=float((distance<=threshold).double().mean())
    return result

def main():
    rows=[]
    for path in (C.HERE/'exp15_finite_scale/results').glob('*/*/rotations.npz'):
        if 'pilot' in path.parts or not (path.parent/'COMPLETE.json').exists(): continue
        model,rep,ch=path.parent.name.split('_'); rep=int(rep[3:]); c=C.load(model,rep,ch)
        raw=torch.load(c['cache_path'],weights_only=False); V=C.tensor(raw['V']); tr=raw['tr']
        covariance=C.cov(V[tr]); inverse=torch.linalg.pinv(covariance)
        dist=torch.einsum('na,ab,nb->n',V[tr],inverse,V[tr]); threshold=float(dist.quantile(.99))
        config=json.loads((path.parent/'config.json').read_text(encoding='utf8'))
        ids=config.get('centers_subjects',config.get('ids'))
        metric=C.tensor(config.get('metric_covariance',config.get('train_cov',C.cov(V[ids]).numpy())))
        data=np.load(path); audit={'proxy':'99% TRAIN Mahalanobis ellipsoid; not proof of joint-density support',
                                   'threshold':threshold,'no_points_removed':True,'solutions':{}}
        for name in data.files:
            if not name.endswith('__R'): continue
            R=data[name][0]; values=coverage(R,V[ids].numpy(),metric,inverse,threshold)
            method=name[:-3]; audit['solutions'][method]=values
            rows.extend(dict(tag=path.parent.parent.name,model=model,repeat=rep,channel=ch,method=method,h=float(h),coverage=v)
                        for h,v in values.items())
        audit['oracle']=coverage(C.oracle(c),V[ids].numpy(),metric,inverse,threshold)
        C.dump(path.parent/'support_audit.json',audit)
    pd.DataFrame(rows).to_csv(C.HERE/'exp15_finite_scale/results/support_summary.csv',index=False)
    print('Geometric support rows',len(rows))

if __name__=='__main__': main()
