"""Function-error calibrated differential program, using exactly the same rule per model."""
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
from rotations.differential_program.calibration import calibrate
from rotations.differential_program.run import write_json,checks,evaluate,alignment

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=['C1','C2','C3','CR3'])
    p.add_argument('--condition',default='beta0p0005_lambda0_ctx0p25to0p8_L8');p.add_argument('--repeat',type=int,default=1);p.add_argument('--seed',type=int,default=None)
    args=p.parse_args();torch.set_num_threads(1);cfg=json.loads((HERE/'calibrated_config.json').read_text())
    cfg['seed']=cfg['seed'] if args.seed is None else args.seed;args.seed=cfg['seed'];args.whiten_floor=cfg['whiten_floor'];args.fit_max_subjects=cfg['fit_max_subjects']
    tick=time.perf_counter();source=load_frozen(args);k=source['rotation_dim'];d=source['x_fit'].shape[-1]
    hashes={str(f.relative_to(ROOT)):sha256(f) for f in sorted(HERE.glob('*.py'))}
    payload=dict(model=args.model,condition=args.condition,repeat=args.repeat,config=cfg,source_hashes=hashes,
                 source_z_sha=source['latent_sha'],source_checkpoint_sha=source['checkpoint_sha'],source_dataset_sha=source['dataset_sha'])
    fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:16]
    out=source['output_root']/'rotations'/'differential_calibrated'/args.condition/f'repeat{args.repeat:02d}'/('run_'+fingerprint)
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
    selected,calibration=calibrate(source,metadata,blocks,cfg)
    print('[calibration]',selected,flush=True)
    program=ProgramObjective(blocks,k,d,selected['hq'],cfg['code_floor'],hx=selected['hx'])
    scores={'calibrated_code':lambda a:program.module_loss(a,'module_code'),'calibrated_l1':lambda a:program.module_loss(a,'group_l1')}
    rotations={'identity':np.eye(k)};fits={};initializations={}
    for name,fun in scores.items():
        a,records,all_a=optimize(fun,k,cfg['seed'],cfg['starts'],cfg['maxiter']);rotations[name]=a;initializations[name]=all_a
        fits[name]=dict(identity=float(fun(torch.eye(k,dtype=torch.float64))),optimized=float(fun(torch.tensor(a))),starts=records,start_alignment=[alignment(a,b) for b in all_a])
    a=rotations[cfg['primary']];audit=checks(source,metadata,blocks,a)
    if audit['orthogonality_max_abs']>1e-8 or audit['full_distribution_error']>1e-8:raise ValueError('Rotation contract failed')
    np.savez_compressed(out/'rotations.npz',**rotations);np.savez_compressed(out/'initializations.npz',**initializations)
    q=(source['z']-source['pca'].mean_)@source['W'].T
    np.savez_compressed(out/'mapping.npz',A=a,W=source['W'],B=np.linalg.pinv(source['W']),mu=source['pca'].mean_,U=q@a.T,train=source['tr'],test=source['te'],fit_subject_ids=source['chosen'])
    write_json(out/'fit.json',dict(**payload,fits=fits,calibration=calibration,selected_scale=selected,numerical_audit=audit,jet_cache=str(cache),
        environment=dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()),elapsed_fit_seconds=time.perf_counter()-tick))
    evaluate(source,rotations,scores,out)
    write_json(out/'complete.json',dict(elapsed_seconds=time.perf_counter()-tick,evaluated=True,exploratory=True))


if __name__=='__main__':main()
