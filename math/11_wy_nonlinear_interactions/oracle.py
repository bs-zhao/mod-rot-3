"""SEPARATE ORACLE EVALUATION ONLY. Never import this from any blind method.
Accepts eval/test W but uses true generation functions exclusively for scoring.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.special import expit
from interactions import detect,save_result
ROOT=Path(__file__).resolve().parents[2]


def logits(model,W,z_dim):
    z=np.asarray(W[:,:z_dim],dtype=float)
    x=np.asarray(W[:,z_dim:],dtype=float)
    if model=='toy_a':
        z1,z2,z3=z.T;x1,x2=x.T
        return z1*x1**z3+z1*z2*x2**z3
    if model=='toy_b':
        z1,z2,z3=z.T;x1,x2=x.T
        return (z1+z2)*x1**z3+(z1*z1-z2*z2)*x2**z3
    if model in ('C1','C2','C3'):
        sys.path.insert(0,str(ROOT/'np'/'src'))
        from modlearn_rot.generators.registry import get_model
        gen=get_model(model)
        return np.einsum('nd,nd->n',gen.design_matrix(x),z)
    if model=='CR3':
        sys.path.insert(0,str(ROOT/'np'/'reference'))
        from s0_cr2_design_recovery import logits as cpt_logits
        return np.asarray(cpt_logits(x,z),dtype=float).reshape(-1)
    raise ValueError(model)


def oracle_predict(model,z_dim,scale):
    def pred(w):
        eta=logits(model,np.asarray(w),z_dim)
        if scale=='logit':return eta
        if scale=='prob':return expit(eta)
        if scale=='softplus':return np.logaddexp(0,eta)
        raise ValueError(scale)
    return pred


def evaluate(data,output,scale='prob',detector='fd',n_eval=96,seed=31,order=3):
    data=Path(data);meta=json.loads((data/'meta.json').read_text())
    if meta.get('latent_source') != 'oracle_coordinates':
        raise ValueError('Analytic oracle requires true theta coordinates; NP Z is not theta. Use separate oracle-coordinate data.')
    with np.load(data/'oracle_reference.npz') as d:
        W=np.asarray(d['W']);eta_ref=np.asarray(d['eta'])
    predict=oracle_predict(meta['model'],meta['z_dim'],scale)
    if not np.allclose(logits(meta['model'],W,meta['z_dim']),eta_ref,atol=3e-5):
        raise AssertionError('Generator and sealed oracle reference disagree')
    if detector=='hessian':raise ValueError('Use finite difference for analytic oracle (no trained network).')
    matrix,details=detect(predict,W,meta['z_dim'],meta['model'],detector,n_eval,seed,order)
    details.update({'oracle_scale':scale,'oracle_only':True})
    save_result(matrix,details,output,{**meta,'label':scale})
    print(json.dumps(details,indent=2))
    return matrix,details


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--scale',choices=['logit','prob','softplus'],default='prob')
    p.add_argument('--detector',choices=['fd','friedman','anova'],default='fd')
    p.add_argument('--n-eval',type=int,default=96);p.add_argument('--order',type=int,default=3)
    p.add_argument('--seed',type=int,default=31)
    a=p.parse_args();evaluate(a.data,a.output,a.scale,a.detector,a.n_eval,a.seed,a.order)
