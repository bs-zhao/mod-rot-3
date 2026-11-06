"""Standalone fit, numerical audit and strictly post-fit parameter evaluation."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'np'/'src')]
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from rotations.stimulus_contrast.method import matrices, loss, optimize, from_angles
from rotations.stimulus_contrast.data import load_frozen, corr, sha256
from rotations.stimulus_contrast.frozen_decoder import _restore, direct_fingerprints

HERE = Path(__file__).resolve().parent


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def alignment(a,b):
    score=np.abs(a@b.T)
    i,j=linear_sum_assignment(-score)
    return float(np.mean(score[i,j]))


def numerical_audit(source, cfg, a, direct):
    model,device=_restore(source,cfg)
    z=np.asarray(source['z_fit'][:8],float)
    w=source['W']; mu=source['pca'].mean_; b=np.linalg.pinv(w)
    q=(z-mu)@w.T; residual=z-mu-q@b.T; u=q@a.T
    recovered=mu+(u@a)@b.T+residual
    x=torch.tensor(direct['common_x'][:12],device=device).unsqueeze(0).expand(len(z),-1,-1)
    def probs(v):
        with torch.no_grad():
            return torch.softmax(model.decode(x,torch.tensor(v,dtype=torch.float32,device=device)),-1).cpu().numpy()
    p=probs(z)
    ids=direct['anchor_fit_subject_ids'][:8]
    zg=source['z_fit'][ids].astype(float)
    ag=torch.tensor(direct['common_x'][:12],device=device).unsqueeze(0).expand(len(ids),-1,-1)
    def logits(v):
        with torch.no_grad():
            raw=model.decode(ag,torch.tensor(v,dtype=torch.float32,device=device))
            return (raw[...,1]-raw[...,0]).cpu().numpy().astype(float)
    fd=[]
    for axis in range(len(a)):
        eps=.01
        approx=(logits(zg+eps*b[:,axis])-logits(zg-eps*b[:,axis]))/(2*eps)
        exact=direct['logit'][:8,axis,:12]
        fd.append(float(np.linalg.norm(approx-exact)/max(np.linalg.norm(exact),1e-12)))
    return dict(orthogonal_max_abs=float(np.max(np.abs(a@a.T-np.eye(len(a))))),
                row_column_direction_max_abs=float(np.max(np.abs(u-(a@q.T).T))),
                latent_roundtrip_max_abs=float(np.max(np.abs(z-recovered))),
                full_distribution_max_abs=float(np.max(np.abs(p-probs(recovered)))),
                truncation_distribution_max_abs=float(np.max(np.abs(p-probs(mu+q@b.T)))),
                finite_difference_relative_error_by_axis=fd)


def evaluate(theta,q,train,test,rotations,folder,names,m):
    # Called only after unsupervised matrices and fit metadata are persisted.
    t=(theta[train]-theta[train].mean(0))/theta[train].std(0)
    if t.shape[1] == q.shape[1]:
        reference,_=orthogonal_procrustes(q[train],t)
        reference=reference.T
        if np.linalg.det(reference)<0:
            reference[-1]*=-1
    else:
        raise ValueError('Supervised square reference requires k == number of parameters')
    rotations=dict(rotations,supervised_reference=reference)
    reports={}; rows=[]
    for name,a in rotations.items():
        u=q@a.T
        c=corr(theta[test],u[test]); c_train=corr(theta[train],u[train])
        i,j=linear_sum_assignment(-(c_train*c_train))
        it,jt=linear_sum_assignment(-(c*c))
        off=np.ones(c.shape,bool); off[i,j]=False
        matched=c[i,j]**2
        reports[name]=dict(signed_test_corr=c.tolist(),signed_train_corr=c_train.tolist(),
                          reference_train_matching=dict(zip(map(str,i),map(int,j))),
                          parameter_names=names,parameter_r2=dict(zip(names,matched.tolist())),
                          train_matched_test_mean_r2=float(matched.mean()),
                          test_optimal_mean_r2=float(np.mean(c[it,jt]**2)),
                          min_r2=float(matched.min()),offtarget_mean_r2=float(np.mean(c[off]**2)),
                          offtarget_max_abs_r=float(np.max(np.abs(c[off]))),
                          contrast_loss=loss(a,m))
        rows.append(dict(method=name,**{k:reports[name][k] for k in (
            'train_matched_test_mean_r2','test_optimal_mean_r2','min_r2',
            'offtarget_mean_r2','offtarget_max_abs_r','contrast_loss')}))
        fig,ax=plt.subplots(figsize=(6,4.8)); im=ax.imshow(c*c,vmin=0,vmax=1,cmap='Blues')
        ax.set_xticks(range(c.shape[1]),[f'u{i+1}' for i in range(c.shape[1])]); ax.set_yticks(range(len(names)),names)
        for row in range(len(names)):
            for col in range(c.shape[1]):
                ax.text(col,row,f'{c[row,col]**2:.2f}',ha='center',va='center')
        ax.set_title(name+' | exploratory TEST'); fig.colorbar(im,ax=ax); fig.tight_layout()
        fig.savefig(folder/(name+'__r2.png'),dpi=150); plt.close(fig)
    write_json(folder/'evaluation.json',reports); write_csv(folder/'comparison.csv',rows)
    # Plane scans through the supervised diagnostic frame. Not an optimization input.
    scans=[]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for i in range(len(reference)):
        for j in range(i+1,len(reference)):
            values=[]
            for angle in np.linspace(-90,90,73):
                v=np.zeros(len(reference)*(len(reference)-1)//2)
                pairs=list(zip(*np.triu_indices(len(reference),1)))
                v[pairs.index((i,j))]=np.deg2rad(angle)
                a=from_angles(v,len(reference))@reference
                c=corr(theta[test],q[test]@a.T)
                ri,ci=linear_sum_assignment(-c*c)
                row=dict(plane=f'{i+1}-{j+1}',angle=float(angle),loss=loss(a,m),
                         test_optimal_r2=float(np.mean(c[ri,ci]**2)))
                scans.append(row); values.append(row)
            axes[0].plot([r['angle'] for r in values],[r['loss'] for r in values],label=f'{i+1}-{j+1}')
            axes[1].plot([r['angle'] for r in values],[r['test_optimal_r2'] for r in values])
    axes[0].set_ylabel('Contrast logit loss'); axes[1].set_ylabel('TEST-optimal mean r²')
    for ax in axes: ax.set_xlabel('Angle from supervised reference (degrees)'); ax.axvline(0,c='gray',lw=.7)
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(folder/'reference_angle_curves.png',dpi=160); plt.close(fig)
    write_csv(folder/'reference_angle_curves.csv',scans)
    np.savez_compressed(folder/'supervised_diagnostic.npz',A=reference)
    return rows


def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True,choices=['C1','C2','C3','CR3'])
    p.add_argument('--condition',default='beta0p0005_lambda0_ctx0p25to0p8_L8'); p.add_argument('--repeat',type=int,default=1)
    args=p.parse_args(); cfg=json.loads((HERE/'config.json').read_text())
    args.representation='pca'; args.rotation_dim=None; args.fit_max_subjects=2500
    args.whiten_floor=cfg['whiten_floor']; args.seed=cfg['seed']
    started=time.perf_counter(); source=load_frozen(args)
    payload=dict(model=args.model,condition=args.condition,repeat=args.repeat,config=cfg,
                 latent_sha=source['latent_sha'],checkpoint_sha=source['checkpoint_sha'])
    h=hashlib.sha256(json.dumps(payload,sort_keys=True).encode())
    for f in sorted(HERE.glob('*.py')): h.update(f.read_bytes())
    folder=source['output_root']/'rotations'/'stimulus_contrast'/args.condition/f'repeat{args.repeat:02d}'/('run_'+h.hexdigest()[:16])
    if folder.exists(): raise FileExistsError(f'Preserve prior output: {folder}')
    folder.mkdir(parents=True)
    direct=None; cache_used=None
    for old in sorted((source['output_root']/'rotations'/'mlp_teacher_rotation'/args.condition/f'repeat{args.repeat:02d}').glob('run_*')):
        if not (old/'teacher__summary.json').exists(): continue
        meta=json.loads((old/'rotation__fit.json').read_text())
        if meta['source_z_sha256']!=source['latent_sha'] or meta['source_checkpoint_sha256']!=source['checkpoint_sha']: continue
        if any(meta['config'].get(k)!=cfg[k] for k in ('seed','direct_anchor_subjects','direct_anchor_x')): continue
        with np.load(old/'rotation__matrix.npz') as mat:
            if not np.array_equal(mat['fit_ids'],source['chosen']): continue
        with np.load(old/'rotation__method_arrays.npz') as data:
            expected=np.linalg.pinv(source['W']).T
            if not np.allclose(data['mlp_dz_dq'],expected,atol=1e-6): continue
            direct=dict(logit=data['mlp_direct_logit_gradient'],probability=data['mlp_direct_probability_gradient'],
                        common_x=data['mlp_common_x'],anchor_fit_subject_ids=data['mlp_anchor_fit_subject_ids'],dz_dq=data['mlp_dz_dq'])
        cache_used=str(old); break
    if direct is None:
        model,device=_restore(source,cfg); direct=direct_fingerprints(model,device,source,cfg)
    np.savez_compressed(folder/'gradients.npz',**{k:v for k,v in direct.items() if isinstance(v,np.ndarray)})
    k=source['rotation_dim']; rotations={'identity':np.eye(k)}; fits={}; ms={}
    for name,mode,contrast in [('contrast_logit','logit',True),('pooled_logit','logit',False),('contrast_probability','probability',True)]:
        ms[name]=matrices(direct[mode],contrast)
        a,runs,all_a=optimize(ms[name],cfg['seed'],cfg['starts'],cfg['maxiter'])
        rotations[name]=a
        fits[name]=dict(identity_loss=loss(np.eye(k),ms[name]),optimized_loss=loss(a,ms[name]),starts=runs,
                        start_alignment=[alignment(a,b) for b in all_a],
                        relative_context_covariance_energy=float(np.sum(ms[name]**2)/np.sum(matrices(direct[mode],False)**2)))
    a=rotations[cfg['primary']]; stability=[]
    for seed in cfg['stability_seeds']:
        alt_cfg=dict(cfg,seed=seed); model,device=_restore(source,alt_cfg)
        alt=direct_fingerprints(model,device,source,alt_cfg)
        am=matrices(alt['logit']); ar,rr,aa=optimize(am,seed,cfg['starts'],cfg['maxiter'])
        rotations[f'anchor_seed_{seed}']=ar
        stability.append(dict(seed=seed,axis_alignment=alignment(a,ar),identity_loss=loss(np.eye(k),am),
                              original_rotation_loss=loss(a,am),refit_loss=loss(ar,am)))
    audit=numerical_audit(source,cfg,a,direct)
    q=(source['z']-source['pca'].mean_)@source['W'].T
    np.savez_compressed(folder/'rotations.npz',**rotations)
    np.savez_compressed(folder/'mapping.npz',A=a,W=source['W'],B=np.linalg.pinv(source['W']),mu=source['pca'].mean_,
                        full_affine_linear=a@source['W'],offset=-a@source['W']@source['pca'].mean_,
                        U=q@a.T,train=source['tr'],test=source['te'],validation=source['val'])
    env=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
             gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
             git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    env['source_sha256']={str(f.relative_to(ROOT)):sha256(f) for f in sorted(HERE.glob('*.py'))}
    write_json(folder/'fit.json',dict(**payload,source_dataset_sha=source['dataset_sha'],cache=cache_used,
                                    fits=fits,anchor_stability=stability,numerical_audit=audit,environment=env,
                                    elapsed_fit_seconds=time.perf_counter()-started))
    if audit['full_distribution_max_abs']>2e-6 or audit['orthogonal_max_abs']>1e-8:
        raise RuntimeError('Numerical contract failed')
    # Optimization complete and all unsupervised choices saved. Read truth only now.
    from modlearn_rot.generators.dataset import load_dataset
    from modlearn_rot.generators.registry import get_model
    data=load_dataset(source['output_root']/'s1_generate'/'s1_generate__dataset.npz')
    names=['alpha','lambda_loss','gamma','tau'] if args.model=='CR3' else list(get_model(args.model).parameter_names)
    rows=evaluate(np.asarray(data.parameters),q,source['tr'],source['te'],rotations,folder,names,ms['contrast_logit'])
    write_json(folder/'complete.json',dict(elapsed_seconds=time.perf_counter()-started,exploratory=True))
    print(json.dumps(dict(folder=str(folder),results=rows,audit=audit,stability=stability),indent=2),flush=True)


if __name__=='__main__': main()
