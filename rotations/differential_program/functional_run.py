"""Integrated-X functional program code; same loss and CV rule for all models."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'np'/'src')]
import argparse
import hashlib
import json
import math
import time
import subprocess
import numpy as np
import torch
from rotations.differential_program.data import load_frozen,sha256
from rotations.differential_program.objectives import ProgramObjective,optimize
from rotations.differential_program.functional import extract,project
from rotations.differential_program.run import write_json,checks,evaluate,alignment

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True,choices=['C1','C2','C3','CR3'])
    parser.add_argument('--condition',default='beta0p0005_lambda0_ctx0p25to0p8_L8');parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--seed',type=int,default=None);args=parser.parse_args();torch.set_num_threads(1)
    cfg=json.loads((HERE/'functional_config.json').read_text());cfg['seed']=cfg['seed'] if args.seed is None else args.seed
    args.seed=cfg['seed'];args.whiten_floor=cfg['whiten_floor'];args.fit_max_subjects=cfg['fit_max_subjects']
    start=time.perf_counter();source=load_frozen(args);k=source['rotation_dim']
    hashes={str(f.relative_to(ROOT)):sha256(f) for f in sorted(HERE.glob('*.py'))}
    payload=dict(model=args.model,condition=args.condition,repeat=args.repeat,config=cfg,source_hashes=hashes,
                 source_z_sha=source['latent_sha'],source_checkpoint_sha=source['checkpoint_sha'],source_dataset_sha=source['dataset_sha'])
    root=source['output_root']/'rotations'/'differential_functional'/args.condition/f'repeat{args.repeat:02d}'
    fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:16];out=root/('run_'+fingerprint)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    cache_payload=dict(z=source['latent_sha'],checkpoint=source['checkpoint_sha'],seed=cfg['seed'],ns=cfg['anchor_subjects'],nx=cfg['fit_x']+cfg['outer_x'],
                       extractor_sha=sha256(HERE/'functional.py'),decoder_sha=sha256(HERE/'frozen_decoder.py'),directional_sha=sha256(HERE/'jets.py'))
    cache=root/('jet_cache_'+hashlib.sha256(json.dumps(cache_payload,sort_keys=True).encode()).hexdigest()[:16]+'.npz')
    if cache.exists():
        with np.load(cache) as f:
            blocks={tuple(map(int,key.split('_')[1:])):f[key] for key in f.files if key.startswith('block_')}
            metadata={key:f[key] for key in f.files if not key.startswith('block_')}
        metadata.update(json.loads(cache.with_suffix('.json').read_text()))
    else:
        blocks,metadata=extract(source,cfg)
        np.savez_compressed(cache,**{f'block_{r}_{s}':v for (r,s),v in blocks.items()},**{key:v for key,v in metadata.items() if isinstance(v,np.ndarray)})
        write_json(cache.with_suffix('.json'),{key:v for key,v in metadata.items() if not isinstance(v,np.ndarray)})
    print(f'[functional jets] ready in {time.perf_counter()-start:.1f}s',flush=True)
    projected,projection_info,coefficients=project(blocks,metadata,cfg)
    d=next(iter(projected.values())).shape[-1]
    program=ProgramObjective(projected,k,d,cfg['h'],cfg['code_floor'])
    scores={'functional_code':lambda a:program.module_loss(a,'module_code'),'functional_l1':lambda a:program.module_loss(a,'group_l1')}
    rotations={'identity':np.eye(k)};fits={};initializations={}
    for name,fun in scores.items():
        tick=time.perf_counter();a,runs,all_a=optimize(fun,k,cfg['seed'],cfg['starts'],cfg['maxiter']);rotations[name]=a;initializations[name]=all_a
        fits[name]=dict(identity=float(fun(torch.eye(k,dtype=torch.float64))),optimized=float(fun(torch.tensor(a))),starts=runs,
                        start_alignment=[alignment(a,b) for b in all_a],seconds=time.perf_counter()-tick)
        print(f'[fit] {name}: {fits[name]["identity"]:.6g} -> {fits[name]["optimized"]:.6g}',flush=True)
    a=rotations[cfg['primary']];audit=checks(source,metadata,blocks,a)
    if audit['orthogonality_max_abs']>1e-8 or audit['full_distribution_error']>1e-8:raise ValueError('Rotation contract failed')
    np.savez_compressed(out/'rotations.npz',**rotations);np.savez_compressed(out/'initializations.npz',**initializations)
    q=(source['z']-source['pca'].mean_)@source['W'].T
    np.savez_compressed(out/'mapping.npz',A=a,W=source['W'],B=np.linalg.pinv(source['W']),mu=source['pca'].mean_,U=q@a.T,train=source['tr'],test=source['te'],fit_subject_ids=source['chosen'])
    np.savez_compressed(out/'projection_coefficients.npz',**{f'degree{r}':value for r,value in coefficients.items()})
    taylor={}
    for step in [.25,.5]:
        vals=[v for v in metadata['taylor_validation'] if v['step']==step]
        taylor[step]=dict(relative_rmse=math.sqrt(sum(v['squared_error'] for v in vals)/max(sum(v['squared_effect'] for v in vals),1e-30)),max_abs=max(v['max_abs'] for v in vals))
    write_json(out/'fit.json',dict(**payload,fits=fits,projection=projection_info,numerical_audit=audit,taylor_effect_validation=taylor,
                jet_cache=str(cache),environment=dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
                git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()),elapsed_fit_seconds=time.perf_counter()-start))
    evaluate(source,rotations,scores,out)
    write_json(out/'complete.json',dict(elapsed_seconds=time.perf_counter()-start,evaluated=True,exploratory=True))


if __name__=='__main__':main()
