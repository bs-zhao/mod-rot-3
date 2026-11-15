"""EXP18: acquire a finite matrix, close the decoder, then fit only W,Y."""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as C
import surrogates as S
from common.decoder import decoder_eta
from common.analytic import ETA

def acquire(c,source,channel,train_subjects,rows_per_subject):
    d=c['Vtrain'].shape[1]
    data,truth=C.io.load_all(c['model'],5e-5 if c['model']=='CR3' else .0005,d,0,c['repeat'])
    v=data['V']; rng=np.random.default_rng(1800)
    c=dict(c)
    if source=='analytic':
        q=C.rotations(d,1,1831)[0].numpy(); v=truth.std()@q.T
        c['V']=v; c['Vtrain']=C.tensor(v[c['ids']]); c['diagnostic_oracle']=q.T; c['source']='matrix_analytic'
    # Split by subjects before selecting trials. No TEST rows used to fit/select.
    train=np.sort(rng.choice(data['train_ids'],min(train_subjects,len(data['train_ids'])),replace=False))
    valid=np.sort(rng.choice(data['val_ids'],min(64,len(data['val_ids'])),replace=False))
    net=None
    if source=='nn': net,_=C.io.load_decoder(data['paths']['checkpoint'],C.io.pick_device('auto'))
    arrays={}
    for label,ids in [('train',train),('validation',valid)]:
        rows=np.concatenate([rng.choice(np.flatnonzero(data['subject_index']==i),rows_per_subject,replace=False) for i in ids])
        si=data['subject_index'][rows]; x=data['x'][rows]; W=np.concatenate([v[si],x],axis=1)
        if source=='choice': y=data['y'][rows]
        elif source=='nn':
            y=decoder_eta(net,data['z'],x,si,C.io.pick_device('auto'),chunk=32768)
            if channel=='prob': y=1/(1+np.exp(-np.clip(y,-500,500)))
        else:
            with torch.no_grad(): y=ETA[c['model']](C.tensor(truth.theta[si]),C.tensor(x)).numpy()
            if channel=='prob': y=1/(1+np.exp(-np.clip(y,-500,500)))
        arrays[label+'_W']=W; arrays[label+'_Y']=y; arrays[label+'_subject_ids']=si
        arrays[label+'_row_ids']=rows
    del net,data,truth
    return c,arrays

def main(a):
    channel='prob' if a.y_source=='choice' else a.channel
    c=C.load(a.model,a.repeat,channel,a.subjects,a.trials)
    c,mat=acquire(c,a.y_source,channel,a.fit_subjects,a.rows_per_subject)
    out=C.result_dir('exp18_matrix_only',a.tag,a.model,a.repeat,channel+'_'+a.y_source)
    big=C.ROOT/'outputs/three_gaps/post_exp13_20261114/matrices'; big.mkdir(parents=True,exist_ok=True)
    matrix_path=big/f'{a.model}_rep{a.repeat}_{channel}_{a.y_source}_{a.tag}.npz'
    np.savez_compressed(matrix_path,**mat)
    # The fitter receives only matrices. No decoder derivative/query references are passed.
    w,y,v,vy=[mat[k] for k in ('train_W','train_Y','validation_W','validation_Y')]
    d=c['Vtrain'].shape[1]
    definitions={'polynomial':S.Polynomial,'kernel_ridge':S.Kernel,'GP384':S.GP,
                 'mlp31':lambda:S.MLP(31,0,a.fit_steps,d),'mlp32':lambda:S.MLP(32,0,a.fit_steps,d),
                 'mlp33':lambda:S.MLP(33,0,a.fit_steps,d),'smooth_mlp':lambda:S.MLP(31,.01,a.fit_steps,d)}
    if a.only: definitions={k:definitions[k] for k in a.only.split(',')}
    si=np.arange(min(16,len(c['ids']))); ki=np.arange(min(16,len(c['X'])))
    points=np.concatenate([np.repeat(c['Vtrain'][si].numpy(),len(ki),axis=0),
                           np.tile(c['X'][ki].numpy(),(len(si),1))],axis=1)
    methods={}; fitting={}; derivatives={}; models={}
    for name,factory in definitions.items():
        t=time.perf_counter(); fitw,fity=w,y
        if name in ('mlp31','mlp32','mlp33'):
            subjects=np.unique(mat['train_subject_ids']); rng=np.random.default_rng(int(name[-2:]))
            keep=rng.choice(subjects,max(8,int(.8*len(subjects))),replace=False)
            mask=np.isin(mat['train_subject_ids'],keep); fitw,fity=w[mask],y[mask]
        model=factory().fit(fitw,fity,v,vy)
        g,tt,tx=model.derivatives(points,c['Vtrain'].shape[1]); mm=C.moments(C.tensor(tx),C.tensor(tt))
        methods[name]=C.Objective(c['Vtrain'],lambda A,mm=mm:C.terms(A,mm))
        pred=model.predict(v)
        fitting[name]=dict(validation_mse=float(np.mean((pred-vy)**2)),seconds=time.perf_counter()-t,
                           probability_outside_fraction=float(((pred<0)|(pred>1)).mean()) if channel=='prob' else None,
                           candidates=model.candidates)
        derivatives[name]=(g,tt,tx); models[name]=model
        print('FIT',name,fitting[name]['validation_mse'],flush=True)
    mlps=[derivatives[k] for k in ('mlp31','mlp32','mlp33') if k in derivatives]
    if len(mlps)==3:
        gg,tt,tx=[np.mean([z[i] for z in mlps],axis=0) for i in range(3)]
        mm=C.moments(C.tensor(tx),C.tensor(tt))
        methods['mlp_ensemble']=C.Objective(c['Vtrain'],lambda A:C.terms(A,mm))
        # Independent frozen-Z surrogate landscapes for EXP17B; never average different NP coordinate systems.
        cal=C.rotations(c['Vtrain'].shape[1],200,1); A=C.effective(cal,C.cov(c['Vtrain']))
        lands=np.stack([C.terms(A,C.moments(C.tensor(z[2]),C.tensor(z[1]))).numpy() for z in mlps])
        rho=[max(0,float(np.mean([np.corrcoef(lands[i,:,k],lands[j,:,k])[0,1]
                for i,j in ((0,1),(0,2),(1,2))]))) for k in range(2)]
        C.dump(out/'independent_decoder_reliability.json',dict(rho=rho,shared_encoder=True,shared_Z=True,
                    original_NP_queries_during_fit=0,limitation='Different initializations and 80% TRAIN-subject subsamples; overlapping data'))
        if sum(rho)>0: methods['ensemble_reliability']=C.Objective(c['Vtrain'],lambda A:C.terms(A,mm),rho)
    np.savez_compressed(out/'estimated_fields.npz',points=points,**{name+'_'+k:z[i] for name,z in derivatives.items()
                                                                         for i,k in enumerate(('g','UU','UX'))})
    C.dump(out/'prediction.json',fitting)
    modeldir=C.ROOT/'outputs/three_gaps/post_exp13_20261114/surrogates'/a.tag/f'{a.model}_rep{a.repeat}_{channel}_{a.y_source}'
    modeldir.mkdir(parents=True,exist_ok=True)
    import pickle
    for name,m in models.items():
        with (modeldir/(name+'.pkl')).open('wb') as f: pickle.dump(m,f)
    C.run_methods(c,out,methods,a,dict(evidence='strict matrix-only '+a.y_source,
                   matrix_path=matrix_path,matrix_sha256=C.io.sha256(matrix_path),
                   selection='held-out subjects Y-MSE only; no T/derivatives in fit or model choice',
                   train_subjects=np.unique(mat['train_subject_ids']),validation_subjects=np.unique(mat['validation_subject_ids']),
                   derivative_points='first 16 selected TRAIN subjects x first 16 fixed stimuli',
                   extrapolation='continuous derivative at fixed bank X is an estimator extension, not identified by bank values alone',
                   fit_rows=len(w),validation_rows=len(v),original_decoder_queries_during_fit_or_search=0,
                   limitations=['new stimulus-design NP training and splines not part of this first round']))
    if a.stage not in ('cache','selftest'):
        from audit import evaluate_fields
        C.dump(out/'postfit_structure_audit.json',evaluate_fields(c,points,derivatives,a.y_source,channel))

if __name__=='__main__':
    p=C.parser(__doc__); p.set_defaults(subjects=64,trials=64)
    p.add_argument('--y-source',choices=['analytic','nn','choice'],default='nn')
    p.add_argument('--fit-subjects',type=int,default=64); p.add_argument('--rows-per-subject',type=int,default=64)
    p.add_argument('--fit-steps',type=int,default=1000); p.add_argument('--only',default='')
    main(p.parse_args())
