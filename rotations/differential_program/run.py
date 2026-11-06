"""Standalone frozen derivative program fit; truths accessed after persistence."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'np'/'src')]
import argparse
import hashlib
import json
import csv
import time
import subprocess
import platform
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from rotations.differential_program.data import load_frozen,sha256,corr
from rotations.differential_program.frozen_decoder import _restore
from rotations.differential_program.jets import build
from rotations.differential_program.objectives import ProgramObjective,RecurrenceObjective,optimize,rotation

HERE=Path(__file__).resolve().parent


def jsonable(value):
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,(np.integer,np.floating)):return value.item()
    return value


def write_json(path,obj):Path(path).write_text(json.dumps(jsonable(obj),indent=2,allow_nan=False),encoding='utf-8')


def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def alignment(a,b):
    score=np.abs(a@b.T);i,j=linear_sum_assignment(-score)
    return float(np.mean(score[i,j]))


def checks(source,metadata,blocks,a):
    model,device=_restore(source,dict(teacher_device='cuda'));model=model.double()
    b=np.linalg.pinv(source['W']);k=len(a)
    z=metadata['source_z'][:2];q=metadata['q'][:2];r=metadata['residual'][:2];x=metadata['common_x'][:4]
    u=q@a.T;z_back=source['pca'].mean_+(u@a)@b.T+r
    xb=torch.tensor(x,dtype=torch.float64,device=device).unsqueeze(0).expand(len(z),-1,-1)
    def distribution(zz):
        with torch.no_grad():return torch.softmax(model.decode(xb,torch.tensor(zz,dtype=torch.float64,device=device)),-1).cpu().numpy()
    def eta(zz):
        with torch.no_grad():
            out=model.decode(xb,torch.tensor(zz,dtype=torch.float64,device=device));return (out[...,1]-out[...,0]).cpu().numpy()
    base=eta(z);step=.01;direction=b[:,0]
    plus=eta(z+step*direction);minus=eta(z-step*direction)
    fd1=(plus-minus)/(2*step);fd2=(plus-2*base+minus)/step**2
    fd3=(eta(z+2*step*direction)-2*plus+2*minus-eta(z-2*step*direction))/(2*step**3)
    exact=[blocks[m,0].reshape(metadata['n_subjects'],metadata['n_x'],*((k,)*m))[:2,:4][(slice(None),slice(None))+(0,)*m] for m in [1,2,3]]
    return dict(orthogonality_max_abs=float(np.max(np.abs(a@a.T-np.eye(k)))),
                row_column_error=float(np.max(np.abs(u-(a@q.T).T))),latent_roundtrip=float(np.max(np.abs(z_back-z))),
                full_distribution_error=float(np.max(np.abs(distribution(z)-distribution(z_back)))),
                finite_difference_axis0_relative=[float(np.linalg.norm(fd-e)/max(np.linalg.norm(e),1e-15)) for fd,e in zip([fd1,fd2,fd3],exact)],
                finite_difference_axis0_max_absolute=[float(np.max(np.abs(fd-e))) for fd,e in zip([fd1,fd2,fd3],exact)])


def evaluate(source,rotations,scores,out):
    # This function is entered only after rotations/mapping/fit.json are saved.
    from modlearn_rot.generators.registry import get_model
    with np.load(source['output_root']/'s1_generate'/'s1_generate__dataset.npz',allow_pickle=False) as f:theta=f['parameters']
    names=['alpha','lambda_loss','gamma','tau'] if source['model']=='CR3' else list(get_model(source['model']).parameter_names)
    q=(source['z']-source['pca'].mean_)@source['W'].T;tr=source['tr'];te=source['te'];k=q.shape[1]
    target=(theta[tr]-theta[tr].mean(0))/theta[tr].std(0)
    ref,_=orthogonal_procrustes(q[tr],target);ref=ref.T
    if np.linalg.det(ref)<0:ref[-1]*=-1
    all_rotations=dict(rotations,supervised_reference=ref);reports={};rows=[]
    for name,a in all_rotations.items():
        u=q@a.T;c=corr(theta[te],u[te]);ct=corr(theta[tr],u[tr]);i,j=linear_sum_assignment(-ct*ct);it,jt=linear_sum_assignment(-c*c)
        off=np.ones(c.shape,bool);off[i,j]=False
        report=dict(parameter_names=names,signed_test_corr=c,train_matching=dict(zip(map(str,i),map(int,j))),
                    parameter_r2=dict(zip(names,(c[i,j]**2).tolist())),mean_r2=float(np.mean(c[i,j]**2)),
                    test_optimal_mean_r2=float(np.mean(c[it,jt]**2)),min_r2=float(np.min(c[i,j]**2)),
                    off_mean_r2=float(np.mean(c[off]**2)),off_max_abs_r=float(np.max(np.abs(c[off]))))
        report['losses']={metric:float(fun(torch.tensor(a,dtype=torch.float64)).detach()) for metric,fun in scores.items()}
        reports[name]=report
        rows.append(dict(method=name,**{key:report[key] for key in ['mean_r2','test_optimal_mean_r2','min_r2','off_mean_r2','off_max_abs_r']},**report['losses']))
        fig,ax=plt.subplots(figsize=(6,5));ax.imshow(c*c,vmin=0,vmax=1,cmap='Blues')
        ax.set_xticks(range(k),[f'u{v+1}' for v in range(k)]);ax.set_yticks(range(len(names)),names);ax.set_title(name+' | exploratory TEST')
        for row in range(len(names)):
            for col in range(k):ax.text(col,row,f'{c[row,col]**2:.2f}',ha='center',va='center',fontsize=9)
        fig.tight_layout();fig.savefig(out/(name+'__r2.png'),dpi=150);plt.close(fig)
    write_json(out/'evaluation.json',reports);write_csv(out/'comparison.csv',rows)
    # Explicitly diagnostic scans, never used to select objectives or directions.
    curves=[]
    for metric,fun in scores.items():
        for i in range(k):
            for j in range(i+1,k):
                for angle in np.linspace(-60,60,49):
                    h=np.eye(k);h[i,i]=h[j,j]=np.cos(np.deg2rad(angle));h[i,j]=-np.sin(np.deg2rad(angle));h[j,i]=np.sin(np.deg2rad(angle))
                    a=h@ref
                    curves.append(dict(metric=metric,plane=f'{i+1}-{j+1}',angle=float(angle),loss=float(fun(torch.tensor(a,dtype=torch.float64)).detach())))
    write_csv(out/'supervised_angle_curves.csv',curves)
    np.savez_compressed(out/'supervised_reference.npz',A=ref)
    print(json.dumps(jsonable(dict(folder=str(out),results=rows)),indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=['C1','C2','C3','CR3'])
    p.add_argument('--condition',default='beta0p0005_lambda0_ctx0p25to0p8_L8');p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--seed',type=int,default=None);p.add_argument('--no-evaluate',action='store_true');args=p.parse_args()
    torch.set_num_threads(1)
    cfg=json.loads((HERE/'config.json').read_text());cfg['seed']=cfg['seed'] if args.seed is None else args.seed
    args.seed=cfg['seed'];args.whiten_floor=cfg['whiten_floor'];args.fit_max_subjects=cfg['fit_max_subjects']
    start=time.perf_counter();source=load_frozen(args);k=source['rotation_dim'];d=source['x_fit'].shape[-1]
    hashes={str(f.relative_to(ROOT)):sha256(f) for f in sorted(HERE.glob('*.py'))}
    payload=dict(model=args.model,condition=args.condition,repeat=args.repeat,config=cfg,source_hashes=hashes,
                 source_z_sha=source['latent_sha'],source_checkpoint_sha=source['checkpoint_sha'],source_dataset_sha=source['dataset_sha'])
    fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:16]
    root=source['output_root']/'rotations'/'differential_program'/args.condition/f'repeat{args.repeat:02d}'
    out=root/('run_'+fingerprint)
    if out.exists():raise FileExistsError(f'Preserve previous output: {out}')
    out.mkdir(parents=True)
    cache_payload=dict(z=source['latent_sha'],checkpoint=source['checkpoint_sha'],seed=cfg['seed'],ns=cfg['anchor_subjects'],nx=cfg['anchor_x'],
                       generator_sha=sha256(HERE/'jets.py'),decoder_sha=sha256(HERE/'frozen_decoder.py'),loader_sha=sha256(HERE/'data.py'))
    cache_hash=hashlib.sha256(json.dumps(cache_payload,sort_keys=True).encode()).hexdigest()[:16]
    cache=root/('jet_cache_'+cache_hash+'.npz');meta_cache=cache.with_suffix('.json')
    jet_start=time.perf_counter()
    if cache.exists():
        with np.load(cache) as f:
            blocks={tuple(map(int,key.split('_')[1:])):f[key] for key in f.files if key.startswith('block_')}
            metadata={key:f[key] for key in f.files if not key.startswith('block_')}
        metadata.update(json.loads(meta_cache.read_text()))
    else:
        blocks,metadata=build(source,cfg)
        np.savez_compressed(cache,**{f'block_{r}_{s}':v for (r,s),v in blocks.items()},**{key:v for key,v in metadata.items() if isinstance(v,np.ndarray)})
        write_json(meta_cache,{key:v for key,v in metadata.items() if not isinstance(v,np.ndarray)})
    print(f'[jets] ready in {time.perf_counter()-jet_start:.1f}s; cache {cache}',flush=True)
    program=ProgramObjective(blocks,k,d,cfg['h'],cfg['code_floor'])
    rec=RecurrenceObjective(blocks,k,metadata['n_subjects'],metadata['n_x'],cfg['h'])
    scores={'module_code':lambda a:program.module_loss(a,'module_code'),'group_l1':lambda a:program.module_loss(a,'group_l1'),'recurrence':rec.loss}
    rotations={'identity':np.eye(k)};fits={};start_arrays={}
    for name in cfg['candidates']:
        tick=time.perf_counter();a,runs,all_a=optimize(scores[name],k,cfg['seed'],cfg['starts'],cfg['maxiter'])
        rotations[name]=a;start_arrays[name]=all_a
        fits[name]=dict(identity=float(scores[name](torch.eye(k,dtype=torch.float64))),optimized=float(scores[name](torch.tensor(a))),
                        starts=runs,start_alignment=[alignment(a,b) for b in all_a],seconds=time.perf_counter()-tick)
        print(f'[fit] {name}: {fits[name]["identity"]:.6g} -> {fits[name]["optimized"]:.6g}, {time.perf_counter()-tick:.1f}s',flush=True)
    a=rotations[cfg['primary']];audit=checks(source,metadata,blocks,a)
    if audit['orthogonality_max_abs']>1e-8 or audit['full_distribution_error']>1e-8:raise ValueError('Rotation contract failed')
    np.savez_compressed(out/'rotations.npz',**rotations);np.savez_compressed(out/'initializations.npz',**start_arrays)
    q=(source['z']-source['pca'].mean_)@source['W'].T
    np.savez_compressed(out/'mapping.npz',A=a,W=source['W'],B=np.linalg.pinv(source['W']),mu=source['pca'].mean_,U=q@a.T,train=source['tr'],test=source['te'],fit_subject_ids=source['chosen'])
    taylor={}
    for step in [.25,.5]:
        vals=[v for v in metadata['taylor_validation'] if v['step']==step]
        taylor[step]=dict(relative_rmse=math.sqrt(sum(v['squared_error'] for v in vals)/max(sum(v['squared_effect'] for v in vals),1e-30)),max_abs=max(v['max_abs'] for v in vals))
    write_json(out/'fit.json',dict(**payload,fits=fits,numerical_audit=audit,taylor_effect_validation=taylor,jet_cache=str(cache),
                environment=dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
                                 git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()),
                elapsed_fit_seconds=time.perf_counter()-start))
    if not args.no_evaluate:evaluate(source,rotations,scores,out)
    write_json(out/'complete.json',dict(elapsed_seconds=time.perf_counter()-start,evaluated=not args.no_evaluate,exploratory=True))


if __name__=='__main__':main()
