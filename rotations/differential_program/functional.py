"""Integrated stimulus representation of exact latent derivative fields.

This projects only in X, never fits a polynomial in latent coordinates.
Hyperparameters use frozen-logit and gradient prediction on withheld X.
All pure/mixed latent tensor entries are retained. A separate outer X split
is kept out of capacity selection. No true parameter is accessed.
"""
from itertools import combinations_with_replacement,permutations
import math
import numpy as np
import torch
from scipy.linalg import cho_factor,cho_solve
from .frozen_decoder import _restore
from .jets import directional


def extract(source,cfg):
    model,device=_restore(source,dict(teacher_device=cfg['device']))
    k=source['rotation_dim'];x=source['x_fit'];mask=source['mask_fit'];n=len(x)
    fixed=n>1 and np.array_equal(mask[0],mask[1]) and np.array_equal(x[0,mask[0]],x[1,mask[1]])
    pool=x[0,mask[0]] if fixed else x[mask]
    mean=pool.astype(float).mean(0);scale=np.maximum(pool.astype(float).std(0),1e-6)
    rng=np.random.default_rng(cfg['seed']+773)
    ns=min(n,cfg['anchor_subjects']);nx=min(len(pool),cfg['fit_x']+cfg['outer_x'])
    subjects=np.sort(rng.choice(n,ns,False));trials=np.sort(rng.choice(len(pool),nx,False))
    common=pool[trials].astype(float);del pool
    q=source['q_fit'][subjects];z=source['z_fit'][subjects];b=np.linalg.pinv(source['W']);mu=source['pca'].mean_
    residual=z-mu-q@b.T
    q_all=np.repeat(q,nx,axis=0);resid_all=np.repeat(residual,nx,axis=0);x_all=np.tile(common,(ns,1))
    blocks={(r,0):np.zeros((len(q_all),)+(k,)*r) for r in [1,2,3]};eta=np.zeros(len(q_all));validation=[]
    bt=torch.tensor(b,dtype=torch.float32,device=device);mt=torch.tensor(mu,dtype=torch.float32,device=device)
    for begin in range(0,len(q_all),cfg['batch_points']):
        end=min(len(q_all),begin+cfg['batch_points'])
        qb=torch.tensor(q_all[begin:end],dtype=torch.float32,device=device)
        rb=torch.tensor(resid_all[begin:end],dtype=torch.float32,device=device);xb=torch.tensor(x_all[begin:end],dtype=torch.float32,device=device)
        def fun(v):
            logits=model.decoder(torch.cat((xb,mt+v@bt.T+rb),dim=-1));return logits[:,1]-logits[:,0]
        eta[begin:end]=fun(qb).detach().cpu().numpy()
        for degree in [1,2,3]:
            for group in combinations_with_replacement(range(k),degree):
                value=directional(fun,qb,group).detach().cpu().numpy()
                for variant in set(permutations(group)):blocks[degree,0][(slice(begin,end),)+variant]=value
            if begin==0:print(f'[functional jets] latent degree {degree}, points {end}/{len(q_all)}',flush=True)
        vrng=np.random.default_rng(cfg['seed']+1901+begin)
        for step in [.25,.5]:
            delta=vrng.normal(size=(end-begin,k))*step/np.sqrt(k)
            with torch.no_grad():actual=(fun(qb+torch.tensor(delta,dtype=torch.float32,device=device))-fun(qb)).cpu().numpy().astype(float)
            pred=np.zeros(end-begin)
            for degree in [1,2,3]:
                letters='abc'[:degree]
                pred+=np.einsum('n'+letters+','+','.join('n'+c for c in letters)+'->n',blocks[degree,0][begin:end],*[delta]*degree,optimize=True)/math.factorial(degree)
            validation.append(dict(step=step,points=end-begin,squared_error=float(np.sum((pred-actual)**2)),squared_effect=float(np.sum(actual**2)),max_abs=float(np.max(np.abs(pred-actual)))))
    metadata=dict(fit_subject_ids=source['chosen'][subjects],common_x=common,x_mean=mean,x_scale=scale,n_subjects=ns,n_x=nx,
                  q=q,residual=residual,source_z=z,taylor_validation=validation,eta=eta.reshape(ns,nx),pool_mode='shared_bank' if fixed else 'pooled_fit')
    return blocks,metadata


def feature_matrix(x,mean,scale,degree,train):
    standardized=(x-mean)/scale;d=x.shape[-1]
    terms=[()]+[g for r in range(1,degree+1) for g in combinations_with_replacement(range(d),r)]
    raw=np.column_stack([np.ones(len(x))]+[np.prod(standardized[:,group],axis=1) for group in terms[1:]])
    center=raw[train].mean(0);sd=np.maximum(raw[train].std(0),1e-8);center[0]=0.;sd[0]=1.
    return (raw-center)/sd,terms,center,sd


def solve(features,target,ids,ridge):
    a=features[ids];gram=a.T@a;penalty=np.ones(gram.shape[0])*len(ids)*ridge;penalty[0]=1e-10
    gram.flat[::len(gram)+1]+=penalty
    return cho_solve(cho_factor(gram,check_finite=False),a.T@target[ids],check_finite=False)


def project(blocks,metadata,cfg):
    ns=metadata['n_subjects'];nx=metadata['n_x'];k=blocks[1,0].shape[1]
    order=np.random.default_rng(cfg['seed']+901).permutation(nx)
    fit=order[:min(cfg['fit_x'],nx-cfg['outer_x'])];outer=order[len(fit):]
    inner=fit[:int(.7*len(fit))];val=fit[int(.7*len(fit)):]
    targets={0:metadata['eta'].reshape(ns,nx).T}
    for r in [1,2,3]:targets[r]=np.moveaxis(blocks[r,0].reshape(ns,nx,k**r),1,0).reshape(nx,ns*k**r)
    trials=[];choices=[]
    for degree in cfg['degrees']:
        phi,terms,center,sd=feature_matrix(metadata['common_x'],metadata['x_mean'],metadata['x_scale'],degree,inner)
        for ridge in cfg['ridges']:
            errors={}
            for r in [0,1]:
                target=targets[r];coeff=solve(phi,target,inner,ridge);prediction=phi[val]@coeff
                errors[r]=float(np.sum((prediction-target[val])**2)/max(np.sum((target[val]-target[inner].mean(0))**2),1e-30))
            score=.5*(errors[0]+errors[1])
            trials.append(dict(degree=degree,ridge=ridge,channels=len(terms),logit_relative_mse=errors[0],gradient_relative_mse=errors[1],score=score))
            choices.append((score,degree,ridge))
    _,degree,ridge=min(choices,key=lambda value:(value[0],value[1],-value[2]))
    phi,terms,center,sd=feature_matrix(metadata['common_x'],metadata['x_mean'],metadata['x_scale'],degree,inner)
    output={};fidelity={};fitted={}
    for r,target in targets.items():
        coeff=solve(phi,target,fit,ridge);prediction=phi@coeff;fitted[r]=coeff
        for label,ids in [('fit',fit),('outer',outer)]:
            error=float(np.linalg.norm(prediction[ids]-target[ids])/max(np.linalg.norm(target[ids]),1e-15))
            r2=float(1-np.sum((prediction[ids]-target[ids])**2)/max(np.sum((target[ids]-target[fit].mean(0))**2),1e-30))
            fidelity[f'{label}_degree{r}']=dict(relative_error=error,r2=r2)
        if r:
            # Full tensor residuals are encoded isotropically: their Frobenius
            # cost is rotation invariant because projection in X commutes with
            # latent rotation. Do not introduce raw-trial sparsity via residuals.
            fields=coeff.T.reshape((ns,)+(k,)*r+(len(terms),))
            output[r,1]=fields
    info=dict(degree=degree,ridge=ridge,channels=len(terms),
              residual_encoding='isotropic full-tensor Frobenius escape code; rotation invariant and excluded from argmin',
              choice_rule='min mean held-out-X relative MSE for frozen logit and full latent gradient; no theta',
              cv=trials,fidelity=fidelity,fit_x_ids=fit,outer_x_ids=outer,terms=terms,feature_center=center,feature_scale=sd)
    return output,info,fitted
