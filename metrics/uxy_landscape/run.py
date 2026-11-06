"""First independent metrics experiment: fixed UXY landscapes, not rotation fitting."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import time
from itertools import combinations

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'np/src')]
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from metrics.uxy_landscape.data import prepare,history,corr,sha256,Y_NAMES
from metrics.uxy_landscape.dependence import Dependence
from metrics.uxy_landscape.regression import Regressions,poly,symmetric_basis

HERE=Path(__file__).resolve().parent


def jsonable(v):
    if isinstance(v,dict):return {str(k):jsonable(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [jsonable(x) for x in v]
    if isinstance(v,np.ndarray):return jsonable(v.tolist())
    if isinstance(v,np.generic):return jsonable(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    if isinstance(v,Path):return str(v)
    return v


def write_json(p,obj):Path(p).write_text(json.dumps(jsonable(obj),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def write_csv(p,rows):
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def plane_rotation(k,i,j,angle):
    a=np.eye(k);r=np.deg2rad(angle)
    a[i,i]=a[j,j]=np.cos(r);a[i,j]=-np.sin(r);a[j,i]=np.sin(r)
    return a


def rotation_cases(k,cfg):
    yield dict(name='u_star',family='reference',plane='reference',angle=0.,A=np.eye(k))
    for i,j in combinations(range(k),2):
        for angle in cfg['angles']:
            yield dict(name=f'plane_{i+1}_{j+1}_{angle:+g}',family='plane',plane=f'u{i+1}_u{j+1}',
                       angle=float(angle),A=plane_rotation(k,i,j,angle))
    pairs=list(combinations(range(k),2))
    weights=np.linspace(.7,1.3,len(pairs))*np.where(np.arange(len(pairs))%2, -1.,1.)
    for angle in cfg['angles']:
        a=np.eye(k)
        for (i,j),weight in zip(pairs,weights):a=plane_rotation(k,i,j,angle*weight)@a
        yield dict(name=f'joint_{angle:+g}',family='joint_path',plane='joint_ramped',angle=float(angle),A=a)
    rng=np.random.default_rng(cfg['seed']+876)
    for index in range(cfg['random_rotations']):
        a=np.linalg.qr(rng.normal(size=(k,k)))[0]
        if np.linalg.det(a)<0:a[0]*=-1
        yield dict(name=f'haar_{index:03d}',family='random_rotation',plane='joint_random',angle=np.nan,A=a)


def recovery(u,source):
    theta=source['theta'];tr=source['tr'];te=source['te']
    train=corr(theta[tr],u[tr]);test=corr(theta[te],u[te])
    i,j=linear_sum_assignment(-train*train)
    off=np.ones(test.shape,bool);off[i,j]=False
    return dict(recovery_mean_r2=float(np.mean(test[i,j]**2)),
                recovery_min_r2=float(np.min(test[i,j]**2)),
                recovery_fixed_axis_r2=float(np.mean(np.diag(test)**2)),
                recovery_off_mean_r2=float(np.mean(test[off]**2)),
                recovery_off_max_abs_r=float(np.max(np.abs(test[off]))),
                signed_test_corr=test,train_assignment=np.column_stack([i,j]))


def compute_model(model,cfg,out):
    from threadpoolctl import threadpool_limits
    thread_limits=threadpool_limits(limits=1)
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=True);tick=time.perf_counter()
    source=prepare(model,cfg);u0=source['u'];k=u0.shape[1];subject=source['subject'];assoc=source['assoc']
    if (out/'sampling.json').exists():
        previous=json.loads((out/'sampling.json').read_text())
        if previous['hashes']!=source['hashes']:
            raise FileExistsError('Input provenance changed; preserve outputs and use --tag for a new run')
    context_hash=hashlib.sha256(source['x'].tobytes()+source['y'].tobytes()+subject.tobytes()).digest()
    write_json(out/'reference.json',source['reference'])
    np.savez_compressed(out/'reference_and_data.npz',A_star=source['reference']['A_star'],
        z_mean=source['reference']['z_mean'],U_star=u0,theta=source['theta'],
        train=source['tr'],test=source['te'],validation=source['val'],
        X=source['x'],X_raw=source['x_raw'],Y=source['y'],Y_names=np.array(Y_NAMES),
        subject=subject,row_ids=source['row_ids'],association_rows=assoc,
        decoder_probability=source['decoder_probability'],
        x_mean=source['x_mean'],x_scale=source['x_scale'])
    write_json(out/'sampling.json',dict(hashes=source['hashes'],analysis_subject_ids=source['chosen'],
        n_rows=len(subject),association_subjects=len(assoc),association_has_one_trial_per_subject=True,
        fold_subject_ids=[np.unique(subject[te]) for tr,te in source['split']],
        noise_sd=source['noise_sd'],Y_names=Y_NAMES,
        decoder_choice='single fixed Bernoulli draw per actual trial, reused at every angle',
        source_note='S5 already encodes observed choices; this is descriptive analysis, not fresh predictive validation'))
    dependent=Dependence(source['x'][assoc],source['y'][assoc],cfg)
    cases=list(rotation_cases(k,cfg))
    old,excluded=history(source,cfg) if cfg['include_history'] else ([],[])
    if cfg.get('smoke'):old=old[:4]
    for item in old:
        cases.append(dict(name=item['name'],family='historical',plane='historical',angle=np.nan,U=item['U'],source=item['path']))
    theta_u=source['target'];theta_u=(theta_u-theta_u[source['tr']].mean(0))/np.sqrt(np.mean(np.var(theta_u[source['tr']],axis=0)))
    cases.append(dict(name='oracle_theta_positive_control',family='oracle_control',plane='oracle',angle=np.nan,U=theta_u))
    write_json(out/'history_exclusions.json',excluded)
    write_json(out/'case_inventory.json',[{key:value for key,value in case.items() if key!='U'} for case in cases])
    print(f'[{model}] TRAIN-OLS reference TEST mean r2={np.mean(source["reference"]["test_diagonal_r2"]):.4f}; cases={len(cases)}, history={len(old)}',flush=True)
    proxy=Regressions(u0[subject],source['x'],source['y'],source['split'],cfg)
    np.savez_compressed(out/'reference_proxy.npz',coefficients=proxy.coefficient,
                        functional_quadratic_cv_r2=proxy.quad_score,functional_linear_cv_r2=proxy.linear_score)
    scalars=[];axis_cases=[];axis_names=None;metadata=[];dedupe={};quality=[]
    cache=out/'case_cache';cache.mkdir(exist_ok=True)
    for index,case in enumerate(cases):
        a=case.get('A',np.eye(k));u=u0@a.T if 'A' in case else case['U']
        # Exact (rounded) reuse only; no sign/permutation alignment by theta.
        digest=hashlib.sha256(context_hash+np.round(u,11).tobytes()).hexdigest()
        case_id=f'{index:04d}'
        metadata.append(dict(case_id=case_id,name=case['name'],family=case['family'],plane=case['plane'],
                             angle=case['angle'],source=case.get('source',''),digest=digest))
        rec=recovery(u,source);quality.append(rec)
        path=cache/(digest+'.npz')
        if path.exists():
            with np.load(path) as saved:
                names=saved['metric_names'].tolist();values=saved['values'];an=saved['axis_names'].tolist();av=saved['axis_values']
        else:
            metrics,axis=dependent.compute(u[subject[assoc]])
            if 'A' in case:
                regress,ra=proxy.compute(a,u[subject])
            else:
                other=Regressions(u[subject],source['x'],source['y'],source['split'],cfg)
                regress,ra=other.compute(np.eye(k),u[subject])
                del other
            metrics.update(regress);axis.update(ra)
            # U-only moment diagnostics intentionally ignore Y and repeat across its versions.
            vv=u[np.unique(subject)];vv=vv-vv.mean(0)
            variance=np.mean(vv**2,axis=0).clip(1e-20)
            metrics['axis_excess_kurtosis_squared_sum']=np.repeat(np.sum((np.mean(vv**4,axis=0)/variance**2-3)**2),len(Y_NAMES))
            metrics['axis_fourth_moment_sum']=np.repeat(np.sum(np.mean(vv**4,axis=0)),len(Y_NAMES))
            metrics['U_total_variance_control']=np.repeat(np.mean(np.sum(vv*vv,axis=1)),len(Y_NAMES))
            names=list(metrics);values=np.stack(list(metrics.values()))
            an=list(axis);av=np.stack(list(axis.values()))
            np.savez_compressed(path,metric_names=np.array(names),values=values,axis_names=np.array(an),axis_values=av)
        if axis_names is None:axis_names=an;metric_names=names
        assert an==axis_names and names==metric_names
        axis_cases.append(av)
        for response_index,response in enumerate(Y_NAMES):
            row=dict(model=model,case_id=case_id,name=case['name'],family=case['family'],plane=case['plane'],
                     angle=case['angle'],response=response,**{key:rec[key] for key in rec if key not in ['signed_test_corr','train_assignment']})
            row.update({name:float(values[i,response_index]) for i,name in enumerate(names)})
            scalars.append(row)
        dedupe.setdefault(digest,case_id)
        if index%10==0 or index==len(cases)-1:
            print(f'[{model}] {index+1}/{len(cases)} cases, {len(dedupe)} unique; {time.perf_counter()-tick:.1f}s',flush=True)
    write_csv(out/'metric_values.csv',scalars);write_csv(out/'case_metadata.csv',metadata)
    np.savez_compressed(out/'axis_metrics.npz',values=np.stack(axis_cases),metric_names=np.array(axis_names),
                        case_ids=np.array([v['case_id'] for v in metadata]),Y_names=np.array(Y_NAMES))
    np.savez_compressed(out/'coordinate_quality.npz',signed_test_corr=np.stack([v['signed_test_corr'] for v in quality]),
                        train_assignment=np.stack([v['train_assignment'] for v in quality]))
    write_json(out/'catalog.json',dict(metric_names=metric_names,axis_metric_families=axis_names,
        n_scalar_metrics=len(metric_names),n_axis_families=len(axis_names),
        warning='Axis RMS and entropy are aggregates of the same estimator, not independent evidence; proxy metrics share one regression',
        NA='Continuous kNN CMI is not applied to binary outcomes; CODEC is undefined when its empirical denominator vanishes'))
    # Exact controls are checked only among rotations of the same reference.
    rotation_rows=[r for r in scalars if r['family'] in ['reference','plane','joint_path','random_rotation']]
    controls=[name for name in metric_names if name.endswith('_control')]
    checks={}
    for name in controls:
        checks[name]=max(float(np.ptp([r[name] for r in rotation_rows if r['response']==response])) for response in Y_NAMES)
    assert max(checks.values())<1e-7,checks
    write_json(out/'complete.json',dict(model=model,cases=len(cases),unique_cases=len(dedupe),historical_labels=len(old),
        scalar_metrics=len(metric_names),axis_families=len(axis_names),invariant_max_range=checks,
        elapsed_seconds=time.perf_counter()-tick,source_hashes=source['hashes']))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=['C1','C2','C3','CR3']);p.add_argument('--smoke',action='store_true')
    p.add_argument('--no-history',action='store_true');p.add_argument('--no-plots',action='store_true')
    p.add_argument('--workers',type=int,default=None)
    p.add_argument('--tag',default='');args=p.parse_args()
    cfg=json.loads((HERE/'config.json').read_text());cfg['tag']=args.tag;cfg['smoke']=args.smoke
    if args.smoke:cfg.update(analysis_subjects=32,trials_per_subject=3,association_subjects=32,angles=[-90,0,45,90],random_rotations=1,profile_x_points=6,workers=1)
    if args.workers is not None:cfg['workers']=args.workers
    if args.no_history:cfg['include_history']=False
    torch.set_num_threads(1)
    hashes={f.name:sha256(f) for f in HERE.iterdir() if f.suffix in ['.py','.json']}
    fingerprint=hashlib.sha256(json.dumps(dict(config=cfg,code=hashes),sort_keys=True).encode()).hexdigest()[:16]
    out=ROOT/'outputs/metrics/uxy_landscape'/('run_'+fingerprint);out.mkdir(parents=True,exist_ok=True)
    write_json(out/'protocol.json',dict(config=cfg,source_hashes=hashes,exploratory=True,
        goal='Describe UXY metric landscapes around supervised TRAIN-OLS coordinates; no rotation objective selected',
        engine_boundary='Dependence and Regressions receive U,X,Y, fold indices and fixed configuration only'))
    models=args.models or cfg['models']
    if cfg['workers']>1 and len(models)>1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(cfg['workers'],len(models))) as pool:
            futures=[pool.submit(compute_model,model,cfg,out/model) for model in models]
            paths=[future.result() for future in futures]
    else:
        paths=[compute_model(model,cfg,out/model) for model in models]
    from metrics.uxy_landscape.reporting import report
    report(out,paths,plots=not args.no_plots)
    print(f'[COMPLETE] {out}',flush=True)


if __name__=='__main__':main()
