"""Select a numerical Taylor radius using frozen-function error, never theta."""
import math
import numpy as np
import torch
from .frozen_decoder import _restore


def calibrate(source,metadata,blocks,cfg):
    model,device=_restore(source,dict(teacher_device=cfg['device']))
    k=source['rotation_dim'];d=metadata['common_x'].shape[-1];nx=metadata['n_x'];ns=metadata['n_subjects']
    q=np.repeat(metadata['q'],nx,axis=0);r=np.repeat(metadata['residual'],nx,axis=0)
    x=np.tile(metadata['common_x'],(ns,1));mean=metadata['x_mean'];scale=metadata['x_scale']
    s=(x-mean)/scale
    rng=np.random.default_rng(cfg['seed']+4511)
    ids=np.sort(rng.choice(len(q),min(cfg['calibration_points'],len(q)),False))
    b=np.linalg.pinv(source['W']);mu=source['pca'].mean_
    direction_q=rng.normal(size=(len(ids),k))/np.sqrt(k);direction_x=rng.normal(size=(len(ids),d))/np.sqrt(d)
    observed=source['x_fit'][source['mask_fit']];lower=(observed.min(0)-mean)/scale;upper=(observed.max(0)-mean)/scale
    qb=torch.tensor(q[ids],dtype=torch.float32,device=device);rb=torch.tensor(r[ids],dtype=torch.float32,device=device)
    bt=torch.tensor(b,dtype=torch.float32,device=device);mt=torch.tensor(mu,dtype=torch.float32,device=device)
    def eta(qv,xv):
        with torch.no_grad():
            z=mt+qv@bt.T+rb;out=model.decoder(torch.cat((xv,z),dim=-1));return (out[:,1]-out[:,0]).cpu().numpy().astype(float)
    rows=[];selected=None
    # Lexicographic largest q radius, then largest X radius satisfying the
    # predeclared error cap on both deterministic FIT calibration halves.
    for hq in cfg['hq_grid']:
        for hx in cfg['hx_grid']:
            dq=direction_q*hq;dx=np.clip(s[ids]+direction_x*hx,lower,upper)-s[ids]
            xt=torch.tensor(mean+(s[ids]+dx)*scale,dtype=torch.float32,device=device)
            actual=eta(qb+torch.tensor(dq,dtype=torch.float32,device=device),xt)-eta(qb,xt)
            prediction=np.zeros(len(ids))
            for (rq,rx),tensor in blocks.items():
                letters='abcdef'[:rq+rx]
                equation='n'+letters+','+','.join('n'+c for c in letters)+'->n'
                prediction+=np.einsum(equation,tensor[ids],*[dq]*rq,*[dx]*rx,optimize=True)/(math.factorial(rq)*math.factorial(rx))
            halves=[]
            for subset in [slice(0,None,2),slice(1,None,2)]:
                halves.append(float(np.linalg.norm(prediction[subset]-actual[subset])/max(np.linalg.norm(actual[subset]),1e-15)))
            row=dict(hq=hq,hx=hx,relative_rmse_halves=halves,max_absolute=float(np.max(np.abs(prediction-actual))),accepted=max(halves)<=cfg['error_cap'])
            rows.append(row)
            if row['accepted']:selected=row;break
        if selected is not None:break
    if selected is None:raise ValueError('No Taylor radius meets the fixed function-error cap')
    return selected,rows
