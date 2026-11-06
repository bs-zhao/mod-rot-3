"""Constrained module program: fix only function-supported equivariant axes.

Same translation/gain tests, thresholds, objective and optimizer for all models.
No axis is named after a simulator parameter until the post-fit audit.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'np'/'src')]
import argparse
import hashlib
import json
import time
import subprocess
import numpy as np
import torch
from rotations.differential_program.data import load_frozen,sha256
from rotations.differential_program.objectives import ProgramObjective,optimize
from rotations.differential_program.jets import build
from rotations.differential_program.frozen_decoder import _restore
from rotations.differential_program.symmetries import detect
from rotations.differential_program.run import write_json,checks,evaluate,alignment

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=['C1','C2','C3','CR3'])
    p.add_argument('--condition',default='beta0p0005_lambda0_ctx0p25to0p8_L8');p.add_argument('--repeat',type=int,default=1);p.add_argument('--seed',type=int,default=None)
    args=p.parse_args();torch.set_num_threads(1);cfg=json.loads((HERE/'equivariant_config.json').read_text())
    cfg['seed']=cfg['seed'] if args.seed is None else args.seed;args.seed=cfg['seed'];args.whiten_floor=cfg['whiten_floor'];args.fit_max_subjects=cfg['fit_max_subjects']
    tick=time.perf_counter();source=load_frozen(args);k=source['rotation_dim'];d=source['x_fit'].shape[-1]
    hashes={str(f.relative_to(ROOT)):sha256(f) for f in sorted(HERE.glob('*.py'))}
    payload=dict(model=args.model,condition=args.condition,repeat=args.repeat,config=cfg,source_hashes=hashes,
                 source_z_sha=source['latent_sha'],source_checkpoint_sha=source['checkpoint_sha'],source_dataset_sha=source['dataset_sha'])
    fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:16]
    out=source['output_root']/'rotations'/'differential_equivariant'/args.condition/f'repeat{args.repeat:02d}'/('run_'+fingerprint)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    cache_payload=dict(z=source['latent_sha'],checkpoint=source['checkpoint_sha'],seed=cfg['seed'],ns=cfg['anchor_subjects'],nx=cfg['anchor_x'],
                       generator_sha=sha256(HERE/'jets.py'),decoder_sha=sha256(HERE/'frozen_decoder.py'),loader_sha=sha256(HERE/'data.py'))
    cache_hash=hashlib.sha256(json.dumps(cache_payload,sort_keys=True).encode()).hexdigest()[:16]
    cache=source['output_root']/'rotations'/'differential_program'/args.condition/f'repeat{args.repeat:02d}'/('jet_cache_'+cache_hash+'.npz')
    if cache.exists():
        with np.load(cache) as f:
            blocks={tuple(map(int,key.split('_')[1:])):f[key] for key in f.files if key.startswith('block_')}
            metadata={key:f[key] for key in f.files if not key.startswith('block_')}
        metadata.update(json.loads(cache.with_suffix('.json').read_text()))
    else:
        cache.parent.mkdir(parents=True,exist_ok=True);blocks,metadata=build(source,cfg)
        np.savez_compressed(cache,**{f'block_{r}_{s}':v for (r,s),v in blocks.items()},**{key:v for key,v in metadata.items() if isinstance(v,np.ndarray)})
        write_json(cache.with_suffix('.json'),{key:v for key,v in metadata.items() if not isinstance(v,np.ndarray)})
    model,device=_restore(source,dict(teacher_device=cfg['device']))
    x=torch.tensor(metadata['common_x'],dtype=torch.float32,device=device);z=torch.tensor(metadata['source_z'],dtype=torch.float32,device=device)
    with torch.no_grad():
        logits=model.decode(x[None].expand(len(z),-1,-1),z);eta=(logits[...,1]-logits[...,0]).cpu().numpy().astype(float)
    g=blocks[1,0].reshape(metadata['n_subjects'],metadata['n_x'],k)
    frame,fixed,reports=detect(eta,g,cfg['symmetry_relative_cap'],cfg['symmetry_gap_multiplier'],cfg['symmetry_activity_floor'])
    print('[symmetries]',[(r['property'],r['accepted'],r['residual_ratio']) for r in reports],flush=True)
    program=ProgramObjective(blocks,k,d,cfg['h'],cfg['code_floor'])
    scores={'equivariant_code':lambda a:program.module_loss(a,'module_code'),'equivariant_l1':lambda a:program.module_loss(a,'group_l1')}
    tf=torch.tensor(frame,dtype=torch.float64)
    def expand(free):
        matrix=torch.eye(k,dtype=torch.float64);matrix[fixed:,fixed:]=free;return matrix@tf
    rotations={'identity':np.eye(k)};fits={};initializations={}
    for name,fun in scores.items():
        if k-fixed<2:
            a=frame;all_a=frame[None];records=[dict(start=0,loss=float(fun(tf)),success=True,nfev=1,message='no free plane')]
        else:
            free,records,all_free=optimize(lambda v:fun(expand(v)),k-fixed,cfg['seed'],cfg['starts'],cfg['maxiter'])
            a=expand(torch.tensor(free)).detach().numpy();all_a=np.stack([expand(torch.tensor(v)).detach().numpy() for v in all_free])
        rotations[name]=a;initializations[name]=all_a
        fits[name]=dict(identity_module_loss=float(fun(torch.eye(k,dtype=torch.float64))),optimized_module_loss=float(fun(torch.tensor(a))),
                        starts=records,start_alignment=[alignment(a,b) for b in all_a],fixed_axis_error=float(np.max(np.abs(a[:fixed]-frame[:fixed]))) if fixed else 0.)
    a=rotations[cfg['primary']];audit=checks(source,metadata,blocks,a)
    if audit['orthogonality_max_abs']>1e-8 or audit['full_distribution_error']>1e-8:raise ValueError('Rotation contract failed')
    np.savez_compressed(out/'rotations.npz',**rotations);np.savez_compressed(out/'initializations.npz',**initializations)
    q=(source['z']-source['pca'].mean_)@source['W'].T
    np.savez_compressed(out/'mapping.npz',A=a,W=source['W'],B=np.linalg.pinv(source['W']),mu=source['pca'].mean_,U=q@a.T,train=source['tr'],test=source['te'],fit_subject_ids=source['chosen'])
    write_json(out/'fit.json',dict(**payload,fits=fits,symmetries=reports,fixed_axes=fixed,symmetry_frame=frame,numerical_audit=audit,jet_cache=str(cache),
        objective_note='minimize module code subject to detected fixed functional axes; reported angle curves show module part only, not constraint feasibility',
        environment=dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()),elapsed_fit_seconds=time.perf_counter()-tick))
    evaluate(source,rotations,scores,out)
    write_json(out/'complete.json',dict(elapsed_seconds=time.perf_counter()-tick,evaluated=True,exploratory=True))


if __name__=='__main__':main()
