"""Exact frozen logit derivatives in whitened q and standardized raw X.

No stimulus feature projection and no surrogate refit. Local residual latent
coordinates are retained. Every unique tensor entry including pure powers is
computed using nested forward-mode AD and copied to symmetric positions.
"""
from itertools import combinations_with_replacement,permutations
import json
import math
import numpy as np
import torch
from .frozen_decoder import _restore

BLOCKS=[(1,0),(2,0),(3,0),(1,1),(1,2),(2,1)]


def directional(fun,base,indices):
    current=fun
    for axis in indices:
        tangent=torch.zeros_like(base);tangent[:,axis]=1
        previous=current
        def current(value,previous=previous,tangent=tangent):
            return torch.func.jvp(previous,(value,),(tangent,))[1]
    return current(base)


def build(source,cfg):
    model,device=_restore(source,dict(teacher_device=cfg['device']))
    k=source['rotation_dim'];d=source['x_fit'].shape[-1]
    x=source['x_fit'];mask=source['mask_fit'];n=len(x)
    fixed=n>1 and np.array_equal(mask[0],mask[1]) and np.array_equal(x[0,mask[0]],x[1,mask[1]])
    pool=x[0,mask[0]] if fixed else x[mask]
    mean=pool.astype(float).mean(0);scale=np.maximum(pool.astype(float).std(0),1e-6)
    rng=np.random.default_rng(cfg['seed']+773)
    ns=min(cfg['anchor_subjects'],n);nx=min(cfg['anchor_x'],len(pool))
    subjects=np.sort(rng.choice(n,ns,False));trials=np.sort(rng.choice(len(pool),nx,False))
    common=np.asarray(pool[trials],float);del pool
    lower=(x[mask].min(0)-mean)/scale;upper=(x[mask].max(0)-mean)/scale
    q=source['q_fit'][subjects];z=source['z_fit'][subjects];b=np.linalg.pinv(source['W']);mu=source['pca'].mean_
    residual=z-mu-q@b.T
    base=np.concatenate((np.repeat(q,nx,0),np.tile((common-mean)/scale,(ns,1))),axis=1)
    resid=np.repeat(residual,nx,0)
    blocks={(r,s):np.zeros((len(base),)+(k,)*r+(d,)*s,dtype=np.float64) for r,s in BLOCKS}
    basis=torch.tensor(b,dtype=torch.float32,device=device);center=torch.tensor(mu,dtype=torch.float32,device=device)
    xm=torch.tensor(mean,dtype=torch.float32,device=device);xs=torch.tensor(scale,dtype=torch.float32,device=device)
    validation=[]
    for begin in range(0,len(base),cfg['batch_points']):
        end=min(len(base),begin+cfg['batch_points'])
        vb=torch.tensor(base[begin:end],dtype=torch.float32,device=device)
        rb=torch.tensor(resid[begin:end],dtype=torch.float32,device=device)
        def fun(v):
            zz=center+v[:,:k]@basis.T+rb;xx=xm+v[:,k:]*xs
            logits=model.decoder(torch.cat((xx,zz),dim=-1))
            return logits[:,1]-logits[:,0]
        for r,s in BLOCKS:
            for qi in combinations_with_replacement(range(k),r):
                for xi in combinations_with_replacement(range(d),s):
                    g=directional(fun,vb,qi+tuple(k+j for j in xi)).detach().cpu().numpy().astype(float)
                    for qp in set(permutations(qi)):
                        for xp in set(permutations(xi)):
                            blocks[r,s][(slice(begin,end),)+qp+xp]=g
            if begin==0:print(f'[jets] block q^{r} X^{s} computed, points {end}/{len(base)}',flush=True)
        # Finite perturbation audit of the joint Taylor effect difference.
        vrng=np.random.default_rng(cfg['seed']+begin+1901)
        for step in [.25,.5]:
            dq=vrng.normal(size=(end-begin,k))*step/np.sqrt(k)
            dx=vrng.normal(size=(end-begin,d))*step/np.sqrt(d)
            # Keep X perturbations in the observed marginal range.
            dx=np.clip(base[begin:end,k:]+dx,lower,upper)-base[begin:end,k:]
            perturb=np.column_stack((dq,dx))
            delta=torch.tensor(perturb,dtype=torch.float32,device=device)
            onlyx=delta.clone();onlyx[:,:k]=0
            with torch.no_grad():actual=(fun(vb+delta)-fun(vb+onlyx)).cpu().numpy().astype(float)
            predicted=np.zeros(end-begin)
            for (r,s),tensor in blocks.items():
                letters='abcdef'[:r+s]
                operands=[tensor[begin:end]]+[dq]*r+[dx]*s
                equation='n'+letters+','+','.join('n'+letter for letter in letters)+'->n'
                predicted+=np.einsum(equation,*operands,optimize=True)/(math.factorial(r)*math.factorial(s))
            validation.append(dict(step=step,points=end-begin,squared_error=float(np.sum((predicted-actual)**2)),squared_effect=float(np.sum(actual**2)),max_abs=float(np.max(np.abs(predicted-actual)))))
    if not all(np.isfinite(v).all() for v in blocks.values()):raise ValueError('Nonfinite derivative tensor')
    metadata=dict(fit_subject_ids=source['chosen'][subjects],common_x=common,x_mean=mean,x_scale=scale,
                  n_subjects=ns,n_x=nx,pool_mode='shared_bank' if fixed else 'pooled_fit',
                  taylor_validation=validation,q=q,residual=residual,source_z=z)
    return blocks,metadata
