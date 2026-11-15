"""Pointwise fields; decoder derivatives and analytic Gap1 are kept distinct."""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from torch.func import jacrev,vmap
import core as C
from common.decoder import decoder_h
from common.derivs import base_derivatives
from common.analytic import make_h

def get(c,source='decoder'):
    sig=hashlib.sha256(np.asarray(c['ids']).tobytes()+c['X'].numpy().tobytes()).hexdigest()[:12]
    out=C.ROOT/'outputs/three_gaps/post_exp13_20261114/fields'
    out.mkdir(parents=True,exist_ok=True)
    path=out/f"{c['model']}_rep{c['repeat']}_{source}_{sig}.pt"
    data,truth=C.io.load_all(c['model'],5e-5 if c['model']=='CR3' else .0005,c['Vtrain'].shape[1],0,c['repeat'])
    if source=='analytic':
        c=dict(c); c['V']=truth.std(); c['Vtrain']=C.tensor(c['V'][c['ids']]); c['source']='analytic'
    if path.exists(): return c,torch.load(path,weights_only=False)
    dev=C.io.pick_device('auto')
    if source=='decoder':
        net,_=C.io.load_decoder(data['paths']['checkpoint'],dev)
        h=decoder_h(net,data['mu_z'],data['Minv'],'logit')
        log=C.load(c['model'],c['repeat'],'logit',len(c['ids']),len(c['X']))
        # A full-cache call must not resample / reorder. Reconstruct the exact subset by IDs/X.
        raw=torch.load(log['cache_path'],weights_only=False)
        si=np.array([list(raw['ids']).index(i) for i in c['ids']])
        ki=np.array([int(torch.argmin((raw['X']-x).square().sum(1))) for x in c['X']])
        b={k:raw[k][si][:,ki].clone() for k in ('g','Htt','Htx','h')}
    else:
        h=make_h(c['model'],'logit',truth.mu,truth.sd)
        b=base_derivatives(h,c['Vtrain'].to(dev),c['X'].to(dev),which=('g','Htt','Htx'),chunk=1024)
        b={k:v.cpu() for k,v in b.items()}
    S,K=len(c['ids']),len(c['X']); d=c['Vtrain'].shape[1]; p=c['X'].shape[1]
    vv=c['Vtrain'][:,None].expand(S,K,d).reshape(-1,d).to(dev)
    xx=c['X'][None].expand(S,K,p).reshape(-1,p).to(dev)
    gx=jacrev(h,1)
    parts=[vmap(gx)(vv[i:i+2048],xx[i:i+2048]).detach().cpu() for i in range(0,len(vv),2048)]
    b['gx']=torch.cat(parts).reshape(S,K,p)
    b={k:C.tensor(v) for k,v in b.items()}
    torch.save(b,path)
    return c,b

def components(b):
    p=b['h'].sigmoid(); s=p*(1-p); t=s*(1-2*p)
    outer_uu=b['g'][..., :,None]*b['g'][...,None,:]
    outer_ux=b['g'][..., :,None]*b['gx'][...,None,:]
    eta=(b['Htx'],b['Htt'])
    a=(s[...,None,None]*eta[0],s[...,None,None]*eta[1])
    bb=(t[...,None,None]*outer_ux,t[...,None,None]*outer_uu)
    return {'eta':eta,'A':a,'B':bb,'p':(a[0]+bb[0],a[1]+bb[1]),
            'gg':(outer_ux,outer_uu),'p1gg':(s[...,None,None]*outer_ux,s[...,None,None]*outer_uu)}

def transformed(b,name):
    x=b['h']
    if name=='tanh':
        t=x.tanh(); a=1-t.square(); bb=-2*t*a
    elif name=='softplus': a=x.sigmoid(); bb=a*(1-a)
    elif name=='asinh': a=(1+x.square()).rsqrt(); bb=-x*(1+x.square()).pow(-1.5)
    else: raise ValueError(name)
    return (a[...,None,None]*b['Htx']+bb[...,None,None]*b['g'][..., :,None]*b['gx'][...,None,:],
            a[...,None,None]*b['Htt']+bb[...,None,None]*b['g'][..., :,None]*b['g'][...,None,:])
