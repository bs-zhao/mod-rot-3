"""Resumable batches of independent, model-blind single Hessian objectives.

All three gaps are measured in one batch. Downstream results of a G1 failure
remain diagnostic controls, never an accepted three-gap success.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from core import load,sha,dump,correspondence
from strict_hessian import CATALOG,is_combination
from single_metric_atlas import BatchLoss
from moment_optimizer import optimize

ROOT=Path(__file__).resolve().parent


def reference_path(label,model):
    paths=list((ROOT/'results'/label).glob(f'{model}_*/**/summary.json'))
    if len(paths)!=1:raise ValueError(f'Expected one frozen reference: {label} {model}, got {len(paths)}')
    return paths[0]


def ensure_reference(args,stage):
    if stage=='G1':return reference_path('h13_modulation_combination_g1',args.model)
    if args.inference=='gaussian256':return reference_path('h08_g2_resolution' if stage=='G2' else 'h09_g3_frozen_decoder',args.model)
    label=f'b01_fields_{args.inference}_{stage}'
    existing=list((ROOT/'results'/label).glob(f'{args.model}_*/**/summary.json'))
    if existing:return existing[0]
    kernel='anova2' if args.inference=='anova2' else 'anova3'
    command=[sys.executable,str(ROOT/'run.py'),'--repo',args.repo,'--model',args.model,'--beta',str(args.beta),
        '--zdim',str(args.zdim),'--pca',str(args.pca),'--source','formula' if stage=='G2' else 'decoder',
        '--channel','logit','--estimator','tensor','--metric','hs_ux_rms_soft003','--samples','4096',
        '--starts','1','--maxiter','1','--seed','20261128','--g1-angle-tolerance','6','--uu-full-domain',
        '--hessian-grid','128','--grid-stimuli','128','--degree','3','--ridge','1e-6',
        '--train-subjects','3500','--validation-subjects','750','--rbf-centers','256',
        '--rbf-length','.75','--utility-ridge','1e-5','--stimulus-kernel',kernel,'--label',label]
    logs=ROOT/'results'/'batch_logs';logs.mkdir(exist_ok=True)
    print(args.model,args.inference,'building',stage,'numeric fields',flush=True)
    with (logs/f'{label}_{args.model}.log').open('w',encoding='utf8') as f:
        subprocess.run(command,check=True,stdout=f,stderr=subprocess.STDOUT)
    return reference_path(label,args.model)


def read_bundle(path,context,args):
    source=json.loads(path.read_text(encoding='utf8'))
    if source.get('scope')!='hessian_only' or source['hashes']!=context['hashes']:raise ValueError('Incompatible frozen source')
    for key in ['model','beta','zdim','pca']:
        if source['args'][key]!=getattr(args,key):raise ValueError('Condition mismatch '+key)
    cache=Path(source['field_cache']).resolve()
    if not cache.is_relative_to(ROOT) or not cache.is_file():raise ValueError('Missing independent-folder numerical fields')
    with np.load(cache,allow_pickle=False) as f:
        if str(f['checkpoint_hash'])!=context['hashes']['checkpoint']:raise ValueError('Checkpoint mismatch')
        fields={key:f[key] for key in ['h','uu_h'] if key in f.files}
    protocol=source['protocol'];grid=protocol['hessian_grid']
    x=(context['data']['x'][grid['stimulus_rows']]-protocol['x_mean'])/protocol['x_sd']
    fields['strict_x']=np.tile(x,(len(grid['subject_ids']),1))
    fields['strict_subject_groups']=np.repeat(np.arange(len(grid['subject_ids'])),len(x))
    if len(fields['h'])!=len(fields['strict_x']):raise ValueError('Wrong numerical grid')
    provenance=dict(summary=str(path),summary_sha256=sha(path),field_sha256=sha(cache),
        source=source['args']['source'],estimator=source['args']['estimator'],hashes=source['hashes'],
        field_args={k:source['args'].get(k) for k in ['seed','degree','ridge','rbf_centers','rbf_length','utility_ridge','stimulus_kernel']},
        inference_diagnostics={k:(source.get('estimator_info') or {}).get(k) for k in ['validation_rmse','block_errors']})
    # No old rotation, labels, supervised metric or loss is returned.
    return fields,np.asarray(source['preprocessing']['covariance']),provenance


def plane_grid(batch,d):
    curves={name:[] for name in batch.losses}
    angles=np.unique(np.r_[np.linspace(0,90,37),np.linspace(0,6,13),np.linspace(84,90,13)])
    for i,j in itertools.combinations(range(d),2):
        for angle in angles:
            r=np.eye(d);a=np.deg2rad(angle);r[i,i]=r[j,j]=np.cos(a);r[i,j]=-np.sin(a);r[j,i]=np.sin(a)
            for name,value in batch(r).items():curves[name].append(dict(i=i,j=j,angle=float(angle),loss=value))
    return curves


def plane_diagnostics(curve,identity_loss,tolerance=6.):
    minima=[];counter=None;ranges=[]
    for i,j in sorted(set((p['i'],p['j']) for p in curve)):
        plane=[p for p in curve if (p['i'],p['j'])==(i,j)];best=min(plane,key=lambda p:p['loss'])
        span=float(np.ptp([p['loss'] for p in plane]));ranges.append(span)
        near=min(p['loss'] for p in plane if min(p['angle'],90-p['angle'])<=tolerance)
        far=min((p for p in plane if min(p['angle'],90-p['angle'])>tolerance),key=lambda p:p['loss'])
        if far['loss']<near-1e-6*(1+abs(near)) and (counter is None or far['loss']<counter['loss']):counter=far
        minima.append(dict(i=i,j=j,angle=best['angle'],error=min(best['angle'],90-best['angle']),span=span))
    return dict(minima=minima,counterexample=counter,flat=bool(min(ranges)<=1e-5*(1+abs(identity_loss))),
        maximum_plane_error=max(p['error'] for p in minima))


def analytic_stage(args,stage,metric,loss,curves,reference_curve):
    diagnostic=plane_diagnostics(curves,loss(np.eye(args.zdim)))
    if diagnostic['counterexample'] is not None or diagnostic['flat']:
        best=None;solutions=[];mode='PLANE_COUNTEREXAMPLE' if diagnostic['counterexample'] else 'FLAT_UNRESOLVED'
        axis_pass=False
    else:
        best,solutions=optimize(loss,args.starts,20261129,args.maxiter);mode='FULL_ROTATION'
        axis_pass=bool(best['axis_error']<=6. and diagnostic['maximum_plane_error']<=6.)
    result=dict(stage=stage,metric=metric,specification=loss.spec,mode=mode,best=best,solutions=solutions,
        planes=diagnostic,curve=curves,axis_pass=axis_pass,pass_gate=axis_pass)
    if stage=='G2':
        exact=np.asarray([p['loss'] for p in reference_curve]);estimated=np.asarray([p['loss'] for p in curves])
        correlations=[]
        for ij in sorted(set((p['i'],p['j']) for p in curves)):
            mask=np.array([(p['i'],p['j'])==ij for p in curves]);a=exact[mask];b=estimated[mask]
            correlations.append(float(np.corrcoef(a,b)[0,1]) if np.ptp(a)>1e-14 and np.ptp(b)>1e-14 else 0.)
        error=float(np.linalg.norm(exact-estimated)/max(np.linalg.norm(exact),1e-20))
        result['curve_relative_error']=error;result['minimum_plane_correlation']=min(correlations)
        result['pass_gate']=bool(axis_pass and error<=.1 and min(correlations)>=.9)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default=str(ROOT.parents[1]));p.add_argument('--model',choices=['C1','C2','C3','CR3'],required=True)
    p.add_argument('--beta',type=float,required=True);p.add_argument('--zdim',type=int,required=True);p.add_argument('--pca',type=int,required=True,choices=[0,1])
    p.add_argument('--inference',choices=['gaussian256','anova2','anova3'],default='gaussian256')
    p.add_argument('--starts',type=int,default=8);p.add_argument('--maxiter',type=int,default=160)
    p.add_argument('--label',default='b01_three_gap_batch');p.add_argument('--only',nargs='+')
    p.add_argument('--prefix',help='Predeclared catalogue family prefix; default hs_b01_')
    args=p.parse_args();torch.set_num_threads(2);started=time.time();context=load(args)
    names=sorted(n for n in CATALOG if n.startswith(args.prefix or 'hs_b01_'))+['hs_ux_rms_soft003']
    if args.only:
        if any(n not in names for n in args.only):raise ValueError('Unknown metric')
        names=args.only
    if any(is_combination(n) for n in names):raise ValueError('Only single objectives in this batch')
    out=ROOT/'results'/args.label/args.model/args.inference/f'starts{args.starts}'
    out.mkdir(parents=True,exist_ok=True)
    stages={};provenances={};curves={}
    for stage in ['G1','G2','G3']:
        ref=ensure_reference(args,stage);fields,cov,provenance=read_bundle(ref,context,args)
        stages[stage]=BatchLoss(fields,cov,args.zdim,names);provenances[stage]=provenance
        if stage!='G3':curves[stage]=plane_grid(stages[stage],args.zdim)
        print(args.model,args.inference,stage,'prepared',len(names),'single metrics',flush=True)
    manifest=dict(args=vars(args),catalogue={n:CATALOG[n] for n in names},fields=provenances,
        rule='All three stages measured; downstream passes of upstream failures are controls only',
        optimizer='L-BFGS-B with derivative of Hessian-only scalar loss with respect to rotation angles',
        source_hashes=context['hashes'],code_hashes={p.name:sha(p) for p in ROOT.glob('*.py')})
    protocol_args=dict(vars(args))
    if args.prefix is None:protocol_args.pop('prefix')
    manifest['protocol_sha256']=hashlib.sha256(json.dumps(dict(args=protocol_args,specs=manifest['catalogue'],fields=provenances),sort_keys=True).encode()).hexdigest()
    dump(out/'manifest.json',manifest)
    completed=[]
    for index,metric in enumerate(names):
        destination=out/(metric+'.json')
        if destination.exists():
            saved=json.loads(destination.read_text(encoding='utf8'))
            if saved['protocol_sha256']!=manifest['protocol_sha256']:raise ValueError('Cannot reuse changed protocol')
            completed.append(metric);continue
        row=dict(model=args.model,metric=metric,inference=args.inference,starts=args.starts,
            protocol_sha256=manifest['protocol_sha256'],specification=CATALOG[metric],gaps={})
        for stage in ['G1','G2']:
            row['gaps'][stage]=analytic_stage(args,stage,metric,stages[stage].losses[metric],curves[stage][metric],curves['G1'][metric])
        loss=stages['G3'].losses[metric];best,solutions=optimize(loss,args.starts,20261129,args.maxiter)
        # Commit R before reading true T or supervised TEST diagnostics.
        locked=out/(metric+'_rotation.json');dump(locked,dict(r=best['r'],metric=metric,protocol_sha256=manifest['protocol_sha256']))
        with np.load(context['latent_path'],allow_pickle=False) as latent:truth=latent['theta_true']
        test=context['splits']['test'];evaluation=correspondence(context['v'][test]@best['r'].T,truth[test])
        oracle_path=list((ROOT/'results'/'oracles').glob(f'{args.model}_*pca{args.pca}.json'))
        if len(oracle_path)!=1:raise ValueError('Missing fixed supervised reference')
        oracle=json.loads(oracle_path[0].read_text(encoding='utf8'))
        if oracle['hashes']!=context['hashes']:raise ValueError('Wrong oracle hashes')
        reference=oracle['rotation_test'];gap=abs(evaluation['matched_mean']-reference['matched_mean'])
        # Raw latent identity is never reported as a true-parameter axis error.
        row['gaps']['G3']=dict(evaluation=evaluation,oracle=reference,mean_gap=gap,
            pass_gate=bool(gap<=.01 and evaluation['off_max']<=reference['off_max']+.02),
            eligible=bool(row['gaps']['G1']['pass_gate'] and row['gaps']['G2']['pass_gate']),
            best={k:v for k,v in best.items() if k!='axis_error'},solutions=[{k:v for k,v in s.items() if k!='axis_error'} for s in solutions],
            rotation_lock_file=str(locked))
        row['all_local_gates']=all(g['pass_gate'] for g in row['gaps'].values())
        dump(destination,row);completed.append(metric)
        dump(out/'progress.json',dict(completed=completed,total=len(names),seconds=time.time()-started))
        print(args.model,args.inference,index+1,'/',len(names),metric,
            'gates',[row['gaps'][s]['pass_gate'] for s in ['G1','G2','G3']],
            'G3mean',round(evaluation['matched_mean'],4),'seconds',round(time.time()-started,1),flush=True)


if __name__=='__main__':main()
