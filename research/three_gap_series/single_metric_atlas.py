"""Single-metric G1 falsification atlas; never promotes on plane scans alone.

Reuses frozen analytic fields, never the reference experiment's rotation/loss.
One condition per invocation; exactly the same catalogue for all conditions.
"""
import argparse
import copy
import csv
import itertools
import json
from pathlib import Path
import time
import numpy as np
import torch
from core import load, dump, sha
from strict_hessian import CATALOG, HessianLoss, is_combination

ROOT=Path(__file__).resolve().parent


def canonical_singles():
    selected={};aliases={}
    for name,spec in CATALOG.items():
        if is_combination(name):continue
        key=dict(spec)
        if key['family']=='jd' and key['weight']==0:key.pop('self_score',None)
        # Identical objective, including the normalization and sample domain.
        if key.get('stat')=='meanabs' and key.get('score')=='l1':key['stat']='local'
        signature=json.dumps(key,sort_keys=True)
        if signature in selected:aliases[name]=selected[signature]
        else:selected[signature]=name
    return list(selected.values()),aliases


class BatchLoss:
    """Share numerical transforms across independent objectives, not their scores."""
    def __init__(self,fields,cov,d,names):
        first=next(n for n in names if CATALOG[n]['family']!='jd')
        base=HessianLoss(fields,cov,d,first)
        self.losses={}
        for name in names:
            if CATALOG[name]['family']=='jd':obj=HessianLoss(fields,cov,d,name)
            else:obj=copy.copy(base);obj.metric=name;obj.spec=CATALOG[name]
            self.losses[name]=obj
        self.base=base

    def __call__(self,r):
        memo={}
        # These methods depend on fields/covariance and their explicit arguments,
        # never on a candidate's penalty specification. Cache only for this R.
        for method in ['_ux','_uu','_full','_third_values','_support_value']:
            bound=getattr(self.base,method)
            def cached(a,*args,_bound=bound,_method=method):
                key=(_method,*args)
                if key not in memo:memo[key]=_bound(a,*args)
                return memo[key]
            for obj in self.losses.values():setattr(obj,method,cached)
        values={name:obj(r) for name,obj in self.losses.items()}
        # Restore original class methods; otherwise the next R would capture a
        # closure holding an obsolete transform from the previous rotation.
        for obj in self.losses.values():
            for method in ['_ux','_uu','_full','_third_values','_support_value']:
                obj.__dict__.pop(method,None)
        return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default=str(ROOT.parents[1]))
    p.add_argument('--model',choices=['C1','C2','C3','CR3'],required=True)
    p.add_argument('--beta',type=float,required=True)
    p.add_argument('--zdim',type=int,required=True)
    p.add_argument('--pca',type=int,choices=[0,1],required=True)
    p.add_argument('--reference',required=True)
    p.add_argument('--random-rotations',type=int,default=32)
    p.add_argument('--label',default='s01_single_metric_atlas')
    p.add_argument('--angle-tolerance',type=float,default=6.)
    args=p.parse_args();started=time.time();torch.set_num_threads(2)
    reference=Path(args.reference).resolve()
    if not reference.is_relative_to(ROOT):raise ValueError('Reference outside independent research folder')
    source=json.loads(reference.read_text(encoding='utf8'))
    a=source['args']
    if source.get('scope')!='hessian_only' or a['source']!='formula' or a['estimator']!='direct':
        raise ValueError('Atlas requires exact analytic Hessians, not inferred fields')
    for key in ['model','beta','zdim','pca']:
        if a[key]!=getattr(args,key):raise ValueError('Wrong reference condition: '+key)
    context=load(args)
    if context['hashes']!=source['hashes']:raise ValueError('Frozen source hashes changed')
    cache=Path(source['field_cache']).resolve()
    if not cache.is_relative_to(ROOT):raise ValueError('Field cache outside research folder')
    with np.load(cache,allow_pickle=False) as z:
        if str(z['checkpoint_hash'])!=context['hashes']['checkpoint']:raise ValueError('Wrong cache checkpoint')
        fields={key:z[key] for key in ['h','uu_h','uu_t','third'] if key in z.files}
    grid=source['protocol']['hessian_grid'];protocol=source['protocol']
    x=(context['data']['x'][grid['stimulus_rows']]-protocol['x_mean'])/protocol['x_sd']
    fields['strict_x']=np.tile(x,(len(grid['subject_ids']),1))
    fields['strict_subject_groups']=np.repeat(np.arange(len(grid['subject_ids'])),len(x))
    cov=np.asarray(source['preprocessing']['covariance']);d=args.zdim
    names,aliases=canonical_singles();batch=BatchLoss(fields,cov,d,names)
    out=ROOT/'results'/args.label/args.model;out.mkdir(parents=True,exist_ok=True)
    identity=batch(np.eye(d));curves={n:[] for n in names};random_best={n:None for n in names}
    rng=np.random.default_rng(a['seed']+1421)
    # Verify the shared computation against direct independent evaluations.
    qr,_=np.linalg.qr(rng.normal(size=(d,d)));shared=batch(qr)
    for name in names:
        independent=HessianLoss(fields,cov,d,name)(qr)
        if not np.isclose(shared[name],independent,rtol=1e-10,atol=1e-11):raise AssertionError(name)
    print(args.model,len(names),'canonical singles; shared evaluation verified',flush=True)
    for i,j in itertools.combinations(range(d),2):
        for degrees in np.linspace(0,90,37):
            r=np.eye(d);angle=np.deg2rad(degrees);r[i,i]=r[j,j]=np.cos(angle);r[i,j]=-np.sin(angle);r[j,i]=np.sin(angle)
            values=batch(r)
            for name,value in values.items():curves[name].append(dict(i=i,j=j,angle=float(degrees),loss=value))
        print(args.model,'plane',i,j,'complete; seconds',round(time.time()-started,1),flush=True)
    for index in range(args.random_rotations):
        r,_=np.linalg.qr(rng.normal(size=(d,d)))
        from scipy.optimize import linear_sum_assignment
        ii,jj=linear_sum_assignment(-abs(r));axis_error=float(np.rad2deg(np.arccos(abs(r[ii,jj]).min())))
        for name,value in batch(r).items():
            previous=random_best[name]
            if previous is None or value<previous['loss']:random_best[name]=dict(loss=value,axis_error=axis_error,r=r)
    rows=[]
    for name in names:
        plane_info=[];counter=False;flat=False
        for i,j in itertools.combinations(range(d),2):
            plane=[row for row in curves[name] if (row['i'],row['j'])==(i,j)]
            best=min(plane,key=lambda row:row['loss']);span=float(np.ptp([row['loss'] for row in plane]))
            near=min(row['loss'] for row in plane if min(row['angle'],90-row['angle'])<=args.angle_tolerance)
            wrong=[row['loss'] for row in plane if min(row['angle'],90-row['angle'])>args.angle_tolerance]
            failed=min(wrong)<near-1e-6*(1+abs(near))
            is_flat=span<=1e-5*(1+abs(identity[name]))
            counter|=failed;flat|=is_flat
            plane_info.append(dict(i=i,j=j,minimum_angle=best['angle'],axis_error=min(best['angle'],90-best['angle']),
                loss_range=span,flat=is_flat,counterexample=failed))
        random=random_best[name]
        random_counter=bool(random is not None and random['axis_error']>args.angle_tolerance and random['loss']<identity[name]-1e-6*(1+abs(identity[name])))
        # A random point beating identity is diagnostic only: a nearby correct
        # optimum could beat it. Only the plane counterexample rejects at screen.
        state='PLANE_COUNTEREXAMPLE' if counter else 'FLAT_PLANE' if flat else 'NEEDS_JOINT_OPTIMIZATION'
        rows.append(dict(metric=name,specification=CATALOG[name],state=state,identity_loss=identity[name],
            planes=plane_info,random_best=random,random_beats_identity=random_counter,
            g1_pass=None,explanation='Screen only; no full rotation optimization, no G2/G3 promotion'))
    with (out/'planes.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['metric','i','j','angle','loss']);w.writeheader()
        for name in names:
            for row in curves[name]:w.writerow(dict(metric=name,**row))
    dump(out/'atlas.json',dict(args=vars(args),source_hashes=context['hashes'],reference_summary=str(reference),
        reference_sha256=sha(reference),field_cache_sha256=sha(cache),channel=a['channel'],seed=a['seed'],
        catalogue=names,aliases=aliases,results=rows,seconds=time.time()-started,
        code_hashes={path.name:sha(path) for path in ROOT.glob('*.py')}))
    print(args.model,'atlas finished',round(time.time()-started,1),flush=True)


if __name__=='__main__':main()
