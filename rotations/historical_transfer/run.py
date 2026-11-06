"""One independent suite: transfer every historical rotation attempt to frozen NP."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import subprocess
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'np/src')]
import numpy as np
import torch
from rotations.historical_transfer.data import load_frozen, sha256
from rotations.historical_transfer.fields import representation, observed_rows, common_design, DecoderField
from rotations.historical_transfer.methods import regression_attempts, ddm_attempts
from rotations.historical_transfer.evaluation import write_json, write_csv, evaluate

HERE=Path(__file__).resolve().parent


def subspace_diagnostics(base,cfg):
    """A1 dimension-selection diagnostics, reported without changing the fixed K."""
    z=base['z'][base['tr']]
    centered=z-z.mean(0)
    eig=np.linalg.eigvalsh(centered.T@centered/(len(z)-1))[::-1]
    cumulative=np.cumsum(eig)/eig.sum()
    rng=np.random.default_rng(cfg['seed']+991)
    null=[]
    for _ in range(100):
        shuffled=np.column_stack([rng.permutation(centered[:,j]) for j in range(centered.shape[1])])
        null.append(np.linalg.eigvalsh(shuffled.T@shuffled/(len(z)-1))[::-1])
    threshold=np.quantile(null,.95,axis=0)
    ratios=eig[:-1]/np.maximum(eig[1:],1e-30)
    return dict(train_eigenvalues=eig,pca_cumulative=cumulative,
                k_pca90=int(np.searchsorted(cumulative,.9)+1),k_parallel_analysis=int(np.sum(eig>threshold)),
                k_eigengap=int(np.argmax(ratios)+1),parallel_95_threshold=threshold,
                retained_k=base['rotation_dim'],
                policy='Controlled fixed K from config for all methods; A1 rules are diagnostics, no theta used')


def audit(field,a):
    source=field.source
    q=source['q'][source['chosen'][:8]]
    z=source['z_fit'][:8]
    u=q@a.T
    inverse=np.linalg.inv(a)
    residual=z-source['mu']-q@source['B'].T
    back=source['mu']+(u@inverse.T)@source['B'].T+residual
    x=source['x_fit'][:8,0]
    eta=field.values(z,x)
    eta_back=field.values(back,x)
    def sigmoid(v):return torch.sigmoid(torch.tensor(v)).numpy()
    return dict(inverse_max_abs=float(np.max(np.abs(a@inverse-np.eye(len(a))))),
                orthogonality_max_abs=float(np.max(np.abs(a@a.T-np.eye(len(a))))),
                condition=float(np.linalg.cond(a)),latent_roundtrip_max_abs=float(np.max(np.abs(z-back))),
                logit_roundtrip_max_abs=float(np.max(np.abs(eta-eta_back))),
                probability_roundtrip_max_abs=float(np.max(np.abs(sigmoid(eta)-sigmoid(eta_back)))))


def fit_model(base,cfg,args,source_hashes):
    k=base['rotation_dim']
    results=[]
    # Compare all methods at a fixed representation, then repeat under the DDM-native
    # raw-top-coordinate standardization. No theta enters either choice.
    for repr_name in cfg['representations']:
        source=representation(base,repr_name,k,cfg['whiten_floor'])
        payload=dict(model=base['model'],condition=args.condition,repeat=args.repeat,
                     representation=source['representation'],config=cfg,
                     source_hashes=source_hashes,latent_sha=source['latent_sha'],
                     checkpoint_sha=source['checkpoint_sha'],dataset_sha=source['dataset_sha'])
        fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:16]
        parent=source['output_root']/'rotations/historical_transfer'/args.condition/f'repeat{args.repeat:02d}'
        out=parent/(repr_name+'__'+fingerprint)
        if out.exists() and (out/'fit_complete.json').exists():
            manifest=json.loads((out/'fit_complete.json').read_text())
            results.append((source,out,manifest['methods']))
            print(f'[reuse completed immutable fit] {out}',flush=True)
            continue
        if out.exists():
            raise FileExistsError(f'Incomplete output preserved for audit: {out}; use --tag to create a separate run')
        out.mkdir(parents=True)
        tick=time.perf_counter()
        write_json(out/'protocol.json',payload)
        write_json(out/'subspace_diagnostics.json',subspace_diagnostics(base,cfg))
        field=DecoderField(source,cfg)
        x,subject,trials,weight=observed_rows(source,cfg)
        full_x,full_subject,full_trials,full_weight=observed_rows(source,cfg,every_trial=True)
        common_x,integration_x,integration_weight,design=common_design(source,cfg)
        print(f'[{base["model"]}/{repr_name}] fields: regression {len(x)}, observed Fisher {len(full_x)}, common {len(common_x)}, integration {len(integration_x)}',flush=True)
        observed=field.evaluate(x,subject)
        full_observed=field.evaluate(full_x,full_subject)
        common=field.common(common_x,finite_difference=True)
        integrated=field.common(integration_x)
        fd_rel=float(np.linalg.norm(common['fd_logit']-common['logit'])/max(np.linalg.norm(common['logit']),1e-30))
        write_json(out/'design.json',dict(integration=design,choice_components=['choice'],
                    rt_components_applicable=False,regression_subjects=len(source['chosen']),
                    regression_rows=len(x),fisher_observed_rows=len(full_x),
                    finite_difference_step=cfg['fd_step'],finite_difference_relative_error=fd_rel,
                    preprocessing_train_only=True,theta_used_for_fit=False))
        np.savez_compressed(out/'fields.npz',x=x,subject=subject,trial=trials,weight=weight,
                    observed_probability=observed['probability'],observed_logit=observed['logit'],
                    common_x=common_x,integration_x=integration_x,integration_weight=integration_weight,
                    common_fd_logit=common['fd_logit'],common_logit=common['logit'],common_p=common['p'],
                    full_observed_x=full_x,full_observed_subject=full_subject,full_observed_trial=full_trials,
                    full_observed_weight=full_weight,full_observed_logit=full_observed['logit'],
                    full_observed_p=full_observed['p'],integrated_logit=integrated['logit'],integrated_p=integrated['p'])
        names=[]
        def emit(name,a,fit,starts):
            target=out/name
            target.mkdir()
            checks=audit(field,a)
            oblique=name=='ddm__jvolmax_oblique'
            if checks['latent_roundtrip_max_abs']>1e-8 or checks['probability_roundtrip_max_abs']>1e-8:
                raise RuntimeError(f'{name} failed exact inverse audit: {checks}')
            if not oblique and checks['orthogonality_max_abs']>1e-8:
                raise RuntimeError(f'{name} failed orthogonality audit')
            np.savez_compressed(target/'mapping.npz',A=a,A_inverse=np.linalg.inv(a),
                                W=source['W'],B=source['B'],mu=source['mu'],
                                U=source['q']@a.T,train=source['tr'],test=source['te'],
                                fit_subject_ids=source['chosen'])
            if starts is not None:np.savez_compressed(target/'starts.npz',matrices=starts)
            write_json(target/'fit.json',dict(method=name,fit=fit,audit=checks,
                       transform_family='invertible_oblique' if oblique else 'orthogonal',
                       theta_accessed=False,elapsed_suite_seconds=time.perf_counter()-tick))
            names.append(name)
            print(f'[{base["model"]}/{repr_name}] saved {name} ({time.perf_counter()-tick:.1f}s)',flush=True)
        emit('identity',np.eye(k),dict(no_fit=True),None)
        regression_attempts(x,subject,observed,cfg,emit)
        ddm_attempts(common,integrated,full_observed,integration_weight,full_weight,cfg,emit)
        write_json(out/'fit_complete.json',dict(methods=names,elapsed_seconds=time.perf_counter()-tick,
                    theta_accessed=False,frozen_before_evaluation=True))
        results.append((source,out,names))
        del field
        if torch.cuda.is_available():torch.cuda.empty_cache()
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=['C1','C2','C3','CR3'],default=['C1','C2','C3','CR3'])
    p.add_argument('--condition',default=None);p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--representations',nargs='+',choices=['pca_whitened','raw_top_standardized'])
    p.add_argument('--no-evaluate',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--tag',default='');args=p.parse_args()
    cfg=json.loads((HERE/'config.json').read_text())
    if args.smoke:
        cfg.update(starts=1,maxiter=8,jvol_steps=12,fit_max_subjects=12,
                   trials_per_subject=12,common_x=12,quadrature_order=2,shape_grid=51)
    if args.representations:cfg['representations']=args.representations
    cfg['tag']=args.tag;cfg['smoke']=args.smoke
    args.condition=args.condition or cfg['condition']
    torch.set_num_threads(1)
    source_hashes={f.name:sha256(f) for f in sorted(HERE.iterdir()) if f.suffix in ['.py','.json']}
    jobs=[]
    for model in args.models:
        settings=SimpleNamespace(model=model,condition=args.condition,repeat=args.repeat,
            seed=cfg['seed'],whiten_floor=cfg['whiten_floor'],fit_max_subjects=cfg['fit_max_subjects'],
            rotation_dim=cfg['rotation_dimensions'][model])
        base=load_frozen(settings)
        jobs.extend(fit_model(base,cfg,args,source_hashes))
    # Every requested fit in both representations is persisted before any theta
    # file member is read; all variants are reported, no automatic winner selection.
    if not args.no_evaluate:
        rows=[]
        for source,out,names in jobs:
            rows.extend(evaluate(source,out,names))
        index=ROOT/'outputs/choice/historical_transfer'
        index.mkdir(parents=True,exist_ok=True)
        fingerprint=hashlib.sha256(json.dumps(dict(config=cfg,hashes=source_hashes,
            paths=[str(job[1]) for job in jobs]),sort_keys=True).encode()).hexdigest()[:16]
        report=index/('summary_'+fingerprint)
        report.mkdir(exist_ok=True)
        write_csv(report/'comparison.csv',rows)
        write_json(report/'runs.json',dict(config=cfg,runs=[str(job[1]) for job in jobs],
                   git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   python=sys.executable,torch=torch.__version__,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
        print(f'[complete] {report}',flush=True)


if __name__=='__main__':main()
