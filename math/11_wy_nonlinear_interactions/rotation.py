"""Rotate only Z, keep a trained function fixed via the correct inverse map.
An option compares the ORACLE curve but it is never visible to blind learning.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from interactions import detect


def givens(d,i,j,angle_deg):
    a=np.deg2rad(angle_deg);r=np.eye(d)
    r[i,i]=r[j,j]=np.cos(a);r[i,j]=np.sin(a);r[j,i]=-np.sin(a)
    return r


def evaluate_rotation(predict,W,R,zdim,model,detector,n_eval,seed):
    new=W.copy();new[:,:zdim]=W[:,:zdim]@R.T
    def coordinate_model(u):
        orig=u.copy();orig[:,:zdim]=u[:,:zdim]@R
        return predict(orig)
    m,info=detect(coordinate_model,new,zdim,model,detector,n_eval,seed)
    return m,info


def scan(data,fit,output,pair=(0,1),step=5.,detector='fd',n_eval=96,seed=31,
         oracle_scale=None,random_rotations=0):
    from models import load_fit
    meta=json.loads((Path(data)/'meta.json').read_text())
    with np.load(Path(data)/'test.npz') as f: W=np.asarray(f['W'],dtype=float)
    zdim=meta['z_dim'];i,j=pair
    if not (0<=i<zdim and 0<=j<zdim and i!=j):raise ValueError('Rotation indices must be distinct Z axes')
    if detector=='hessian':
        raise ValueError('For rotation use finite differences on fixed predictions (exact coordinate map).')
    if oracle_scale is None:
        model,info=load_fit(fit)
        pred=(lambda x:model.predict_proba(x)[:,1]) if info['method']=='ebm' and meta['label']=='bernoulli' else model.predict
    else:
        from oracle import oracle_predict
        pred=oracle_predict(meta['model'],zdim,oracle_scale)
    angles=np.arange(-90,90+step*.1,step)
    transforms=[(f'{a:g}deg',givens(zdim,i,j,a),float(a)) for a in angles]
    if random_rotations:
        rng=np.random.default_rng(seed+122)
        for k in range(random_rotations):
            q,_=np.linalg.qr(rng.normal(size=(zdim,zdim)))
            if np.linalg.det(q)<0:q[:,0]*=-1
            transforms.append((f'random{k:04d}',q,None))
    records=[];output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,r,angle in transforms:
        matrix,result=evaluate_rotation(pred,W,r,zdim,meta['model'],detector,n_eval,seed)
        records.append({'rotation':name,'angle_deg':angle,'zx_l1':result['zx_l1'],
                        'zx_effective_degree':result['zx_effective_degree'],
                        'cr3_stimulus_modulation_l1':float(np.nansum(result.get('valid_stimulus_z_modulation',[])))
                                if meta['model']=='CR3' else None})
    (output/'rotation_curve.json').write_text(json.dumps({'model':meta['model'],'label':meta['label'],'detector':detector,
        'oracle_scale':oracle_scale,'pair':list(pair),'records':records},indent=2),encoding='utf8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=['zx_l1','zx_effective_degree']
    if meta['model']=='CR3':keys=['cr3_stimulus_modulation_l1']
    for key in keys:
        curve=[r for r in records if r['angle_deg'] is not None and r[key] is not None]
        if not curve:continue
        fig,ax=plt.subplots(figsize=(8,4))
        ax.plot([x['angle_deg'] for x in curve],[x[key] for x in curve],marker='.')
        ax.set(xlabel='Z rotation (degrees)',ylabel=key,title=f"{meta['model']} {key}")
        ax.grid(alpha=.3);fig.tight_layout();fig.savefig(output/f'{key}.png',dpi=150);plt.close(fig)
    print(json.dumps({'saved':str(output),'n_rotations':len(records),'valid_metrics':keys},indent=2))
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--fit')
    p.add_argument('--output',required=True);p.add_argument('--axis-i',type=int,default=0)
    p.add_argument('--axis-j',type=int,default=1);p.add_argument('--step',type=float,default=5.)
    p.add_argument('--detector',choices=['fd','friedman','anova'],default='fd')
    p.add_argument('--n-eval',type=int,default=96);p.add_argument('--seed',type=int,default=31)
    p.add_argument('--oracle-scale',choices=['logit','prob','softplus'])
    p.add_argument('--random-rotations',type=int,default=0)
    a=p.parse_args()
    if a.oracle_scale is None and not a.fit:p.error('--fit required unless --oracle-scale')
    scan(a.data,a.fit,a.output,(a.axis_i,a.axis_j),a.step,a.detector,
         a.n_eval,a.seed,a.oracle_scale,a.random_rotations)
