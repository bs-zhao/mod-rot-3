"""Held-out subject/X finite interventions for FIT-detected functional roles.

No theta is loaded. This validates function behavior beyond infinitesimal AD,
without changing the frozen candidate or the supported-axis detector.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'np/src')]
import argparse
import json
from types import SimpleNamespace
import numpy as np
import torch
from rotations.differential_program.data import load_frozen
from rotations.differential_program.frozen_decoder import _restore
from rotations.differential_program.run import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    out=Path(args.run);fit=json.loads((out/'fit.json').read_text());cfg=fit['config']
    if (out/'finite_equivariance_audit.json').exists():raise FileExistsError('Preserve completed audit')
    source=load_frozen(SimpleNamespace(model=fit['model'],condition=fit['condition'],repeat=fit['repeat'],
            seed=cfg['seed'],whiten_floor=cfg['whiten_floor'],fit_max_subjects=cfg['fit_max_subjects']))
    accepted=[item for item in fit['symmetries'] if item['accepted']]
    if not accepted:
        write_json(out/'finite_equivariance_audit.json',dict(status='no supported functional axis'));return
    with np.load(fit['jet_cache']) as f:old_x=f['common_x']
    rng=np.random.default_rng(9101);ids=np.sort(rng.choice(source['te'],min(128,len(source['te'])),False))
    x=source['x_fit'];mask=source['mask_fit']
    fixed=len(x)>1 and np.array_equal(mask[0],mask[1]) and np.array_equal(x[0,mask[0]],x[1,mask[1]])
    pool=x[0,mask[0]] if fixed else x[mask]
    used={tuple(row) for row in old_x}
    candidate_ids=rng.permutation(len(pool))[:min(len(pool),1024)]
    eligible=[i for i in candidate_ids if tuple(pool[i].astype(float)) not in used]
    common=np.asarray(pool[eligible[:64]],float)
    if len(common)<16:raise ValueError('Insufficient independent X anchors')
    z=source['z'][ids];b=np.linalg.pinv(source['W']);model,device=_restore(source,dict(teacher_device='cuda'))
    def eta(zz):
        values=[]
        with torch.no_grad():
            for begin in range(0,len(zz),16):
                zb=torch.tensor(zz[begin:begin+16],dtype=torch.float32,device=device)
                xb=torch.tensor(common,dtype=torch.float32,device=device)[None].expand(len(zb),-1,-1)
                logits=model.decode(xb,zb);values.append((logits[...,1]-logits[...,0]).cpu().numpy())
        return np.concatenate(values).astype(float)
    baseline=eta(z);rows=[]
    mapping=np.load(out/'mapping.npz');a=mapping['A']
    for axis,role in enumerate(accepted):
        direction=b@a[axis]
        for step in [-.5,-.25,.25,.5]:
            shifted=eta(z+step*direction);change=shifted-baseline
            if role['property']=='output_translation':
                fitted=baseline+change.mean(1,keepdims=True)
            else:
                factor=np.sum(shifted*baseline,axis=1,keepdims=True)/np.maximum(np.sum(baseline**2,axis=1,keepdims=True),1e-15)
                fitted=baseline*factor
            residual=np.sum((shifted-fitted)**2,axis=1);effect=np.sum(change*change,axis=1)
            rows.append(dict(axis=axis,property=role['property'],step=step,
                    weighted_residual_fraction=float(residual.sum()/max(effect.sum(),1e-15)),
                    weighted_relative_rmse=float(np.sqrt(residual.sum()/max(effect.sum(),1e-15))),
                    median_subject_residual_fraction=float(np.median(residual/np.maximum(effect,1e-15))),
                    change_rms=float(np.sqrt(np.mean(change**2)))))
    write_json(out/'finite_equivariance_audit.json',dict(subjects=len(ids),new_x=len(common),subject_split='NP TEST',
                x_protocol='FIT-observed stimuli excluded from symmetry-fitting anchors',rows=rows,
                no_theta=True,note='Descriptive frozen-function intervention, not independent data/NP training validation'))
    print(json.dumps(dict(model=fit['model'],repeat=fit['repeat'],rows=rows),indent=2))


if __name__=='__main__':main()
