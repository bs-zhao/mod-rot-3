"""Disambiguate float32 finite-difference cancellation on weak PCA axes."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'np'/'src')]
import argparse
import json
import numpy as np
import torch
from rotations.mlp_teacher_rotation.teacher import _restore
from rotations.stimulus_contrast.run import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    run=Path(args.run);meta=json.loads((run/'fit.json').read_text());mapping=np.load(run/'mapping.npz');direct=np.load(run/'gradients.npz')
    base=ROOT/'outputs'/'choice'/meta['model']
    with np.load(base/'s4_evaluate'/meta['condition']/f"repeat{meta['repeat']:02d}"/'s4_evaluate__latent.npz') as data:z=data['z'].astype(float)
    train=mapping['train'];chosen=np.sort(np.random.default_rng(meta['config']['seed']).choice(train,min(2500,len(train)),False))
    ids=chosen[direct['anchor_fit_subject_ids'][:8]];x=direct['common_x'][:12]
    source=dict(checkpoint_path=base/'s3_train'/meta['condition']/f"repeat{meta['repeat']:02d}"/'s3_train__checkpoint.pt',z=z,x_fit=x[None])
    model,device=_restore(source,dict(teacher_device='cuda'));reports={}
    for dtype in [torch.float32,torch.float64]:
        model=model.to(dtype=dtype)
        xb=torch.tensor(x,dtype=dtype,device=device).unsqueeze(0).expand(len(ids),-1,-1)
        zz=torch.tensor(z[ids],dtype=dtype,device=device)
        def fun(v):
            y=model.decode(xb,v);return y[...,1]-y[...,0]
        rows=[]
        for axis in range(mapping['B'].shape[1]):
            tangent=torch.tensor(mapping['B'][:,axis],dtype=dtype,device=device).expand_as(zz)
            _,g=torch.func.jvp(fun,(zz,),(tangent,));g=g.detach().cpu().numpy().astype(float)
            record=dict(axis=axis,gradient_rms=float(np.sqrt(np.mean(g*g))),
                        max_absolute_difference_from_saved_ad=float(np.max(np.abs(g-direct['logit'][:8,axis,:12]))))
            for eps in [.001,.01,.1]:
                with torch.no_grad():fd=((fun(zz+eps*tangent)-fun(zz-eps*tangent))/(2*eps)).cpu().numpy().astype(float)
                record[f'fd_relative_error_eps_{eps}']=float(np.linalg.norm(fd-g)/max(np.linalg.norm(g),1e-15))
            rows.append(record)
        reports[str(dtype)]=rows
    write_json(run/'derivative_precision_audit.json',reports);print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
