"""Post-fit oracle diagnostic, never a rotation loss or selection rule.

Compare frozen decoder values and gradients to simulator derivatives transported
through a TRAIN-fitted affine theta(q) chart. Gradient mismatch includes chart
approximation; it must not automatically be called decoder distortion.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'np'/'src'),str(ROOT/'np'/'reference')]
import argparse
import json
import numpy as np
import torch
from scipy.special import expit
from rotations.stimulus_contrast.data import corr
from rotations.stimulus_contrast.run import write_json
from rotations.mlp_teacher_rotation.teacher import _restore


def evaluate_model(model,device,z,x,b):
    logits=[];grads=[]
    for begin in range(0,len(z),16):
        zz=torch.tensor(z[begin:begin+16],dtype=torch.float32,device=device)
        xx=torch.tensor(x,dtype=torch.float32,device=device).unsqueeze(0).expand(len(zz),-1,-1)
        def fun(v):
            y=model.decode(xx,v);return y[...,1]-y[...,0]
        gs=[]
        for axis in range(b.shape[1]):
            y,g=torch.func.jvp(fun,(zz,),(torch.tensor(b[:,axis],dtype=torch.float32,device=device).expand_as(zz),))
            gs.append(g.detach().cpu().numpy())
        logits.append(y.detach().cpu().numpy());grads.append(np.stack(gs,axis=1))
    return np.concatenate(logits).astype(float),np.concatenate(grads).astype(float)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    run=Path(args.run);meta=json.loads((run/'fit.json').read_text());model_name=meta['model']
    out=run/'oracle_structure_audit';out.mkdir(exist_ok=False)
    mapping=np.load(run/'mapping.npz');direct=np.load(run/'gradients.npz')
    base=ROOT/'outputs'/'choice'/model_name
    with np.load(base/'s1_generate'/'s1_generate__dataset.npz') as d:theta=d['parameters'].astype(float)
    with np.load(base/'s4_evaluate'/meta['condition']/f"repeat{meta['repeat']:02d}"/'s4_evaluate__latent.npz') as d:z=d['z'].astype(float)
    q=(z-mapping['mu'])@mapping['W'].T;train=mapping['train'];test=mapping['test']
    regression=np.linalg.lstsq(np.column_stack((np.ones(len(train)),q[train])),theta[train],rcond=None)[0]
    predicted=np.column_stack((np.ones(len(q)),q))@regression
    theta_r2=1-np.sum((predicted[test]-theta[test])**2,axis=0)/np.sum((theta[test]-theta[train].mean(0))**2,axis=0)
    chosen=np.sort(np.random.default_rng(meta['config']['seed']).choice(train,min(2500,len(train)),False))
    fit_anchor=chosen[direct['anchor_fit_subject_ids']]
    test_anchor=np.sort(np.random.default_rng(1901).choice(test,min(160,len(test)),False))
    x=direct['common_x'].astype(float)
    source=dict(checkpoint_path=base/'s3_train'/meta['condition']/f"repeat{meta['repeat']:02d}"/'s3_train__checkpoint.pt',
                z=z,x_fit=x[None])
    model,device=_restore(source,dict(teacher_device='cuda'))
    if model_name=='CR3':
        from s0_cr2_design_recovery import logits
        def oracle(t):return logits(x[None,:,:],t[:,None,:])
    else:
        from modlearn_rot.generators.registry import get_model
        generator=get_model(model_name)
        def oracle(t):return generator.linear_predictor(x,t)
    reports={};arrays={}
    for split,ids in [('FIT_anchors',fit_anchor),('TEST_anchors',test_anchor)]:
        y,g=evaluate_model(model,device,z[ids],x,mapping['B'])
        t=theta[ids];eta=oracle(t)
        derivatives=[]
        for axis in range(t.shape[1]):
            step=1e-4*theta[train,axis].std()
            shift=np.zeros_like(t);shift[:,axis]=step
            derivatives.append((oracle(t+shift)-oracle(t-shift))/(2*step))
        gt=np.stack(derivatives,axis=1)
        transported=np.einsum('kp,npx->nkx',regression[1:],gt)
        model_prob=expit(y);true_prob=expit(eta)
        reports[split]=dict(subject_count=len(ids),x_count=len(x),
            probability_rmse=float(np.sqrt(np.mean((model_prob-true_prob)**2))),
            logit_r2=float(1-np.sum((y-eta)**2)/np.sum((eta-eta.mean())**2)),
            gradient_relative_frobenius_error=float(np.linalg.norm(g-transported)/np.linalg.norm(transported)),
            gradient_cosine=float(np.sum(g*transported)/(np.linalg.norm(g)*np.linalg.norm(transported))),
            gradient_relative_error_by_axis=(np.sqrt(np.sum((g-transported)**2,axis=(0,2))/np.sum(transported**2,axis=(0,2)))).tolist())
        arrays[split+'_ids']=ids;arrays[split+'_teacher_gradient']=g;arrays[split+'_transported_oracle_gradient']=transported
        arrays[split+'_teacher_logit']=y;arrays[split+'_oracle_logit']=eta
    write_json(out/'summary.json',dict(model=model_name,source_run=str(run),theta_affine_train_fitted_test_r2=theta_r2.tolist(),
                affine_chart=regression.tolist(),reports=reports,
                limitation='Oracle gradients transported through a TRAIN affine theta(q) approximation. Residual mismatch includes chart geometry and decoder mismatch. No objective weights or rotations are changed.'))
    np.savez_compressed(out/'arrays.npz',**arrays)
    print(json.dumps(dict(model=model_name,theta_affine_test_r2=theta_r2.tolist(),reports=reports),indent=2))


if __name__=='__main__':main()
