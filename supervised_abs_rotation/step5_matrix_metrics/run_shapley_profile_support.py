"""BU/BV: sparsity of complete conditional Shapley UX response profiles."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as profiles
import learned_response_profiles as learned
import conditional_ux as conditional
import shapley_function_profiles as shapley
import operator_mode_support as mode_support
import ux_metric_scope as scope
from run_operator_mode_support import audit_frames
from validation_response_policy import response_fields


def analyze(source,original,bank_folder,output,bank_factory=profiles.MatrixBank,extra_frames=(),extra_descriptions=(),
            degree=3,centers=32,device='cuda',model_factory=None,protocol_overrides=None):
    original,output=Path(original),Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Retain earlier Shapley profile run: {output}')
    old=json.loads((original/'protocol.json').read_text())
    if old['status']!='COMPLETE' or hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest()!=old['scores_sha256']:
        raise ValueError('Frozen source wave changed')
    bank=bank_factory(source,bank_folder);data=bank.data
    if bank.protocol['manifest_sha256']!=old['source_matrix_manifest_sha256']:raise ValueError('Actual response source changed')
    output.mkdir(parents=True)
    protocol=dict(status='RUNNING',model=data.index['model'],all_subjects=data.n,all_original_rows=data.m,
        conditional_model='complete polynomial plus generic radial kernel, unpenalized affine span, GCV ridge selection',
        degree=degree,radial_centers=centers,seed=471,conditional_means_fitted_on_all_subjects=True,
        conditional_model_generalization_independently_verified=False,
        value_function='estimated conditional response mean for every proper U coalition; observed response for full coalition; zero centered empty coalition',
        decomposition='exact all-coalition Shapley function shares; no sampled permutations',
        independence_of_inputs_assumed=False,interpretation='finite conditional explanation profiles, not exact population Shapley variance effects',
        source_response_quality=bank.quality,response_transform=bank.protocol.get('response_transform','identity'),
        response_channel_provenance={c:response_fields(c) for c in data.channels},
        source_matrix_manifest_sha256=bank.protocol['manifest_sha256'],source_scores_sha256=old['scores_sha256'],
        prediction_or_position_veto=False,extra_frozen_frames=list(extra_descriptions),
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in ('run_shapley_profile_support.py','shapley_function_profiles.py','operator_mode_support.py'))).hexdigest())
    if protocol_overrides:protocol.update(protocol_overrides)
    wave.json_file(output/'protocol.json',protocol)
    with np.load(original/'W_only_geometry.npz',allow_pickle=False) as saved:
        old_directions,old_membership=saved['directions'],saved['membership']
    maps=[old_directions[axes].T for axes in old_membership[:len(data.paths)]]+list(extra_frames)
    directions,membership=conditional.direction_classes(maps)
    frame_keys=list(dict.fromkeys(tuple(sorted(axes)) for axes in membership))
    frame_index=np.array([frame_keys.index(tuple(sorted(axes))) for axes in membership])
    np.savez_compressed(output/'W_only_geometry.npz',directions=directions,membership=membership,
                        frame_axes=np.array(frame_keys),case_frame_index=frame_index)
    operators=[np.load(original/f'X_witness_column{j}.npy',allow_pickle=False) for j in range(data.dx)]
    tasks,targets,discarded,audit=profiles.task_modes(bank.responses,operators,tolerance=1e-8)
    wave.json_file(output/'matrix_function_modes.json',dict(tasks=audit,maximum_discarded_relative_energy=float(np.max(discarded.sum(1)/data.n))))
    latent,projection,slices,compression=shapley.compress_targets(targets)
    wave.json_file(output/'shared_target_compression.json',compression)
    np.savez_compressed(output/'shared_response_basis.npz',latent=latent,projection=projection,task_slices=np.array(slices),tasks=np.array(tasks))
    y=torch.tensor(latent,dtype=torch.float64,device=device)
    task_projection=[torch.tensor(projection[:,a:b],dtype=torch.float64,device=device) for a,b in slices]
    z=data.u-data.u.mean(0);z/=np.sqrt(np.var(data.u,axis=0).mean())
    factory=model_factory or shapley.CoalitionModels
    models=factory(z,y,output/'conditional_models',degree=degree,centers=centers,device=device)
    if hasattr(models,'protocol_info'):
        if models.protocol_info['source_matrix_manifest_sha256']!=bank.protocol['manifest_sha256']:
            raise ValueError('Conditional emulator factory uses different actual matrices')
        protocol.update(models.protocol_info);wave.json_file(output/'protocol.json',protocol)
    if hasattr(models,'response_audits'):
        wave.json_file(output/'full_model_response_audits.json',models.response_audits(task_projection,tasks,data.channels))
    width=max(t.shape[1] for t in targets);frame_scores={};numerical=[]
    for f,axes in enumerate(frame_keys):
        q=directions[list(axes)].T
        if hasattr(models,'frame_spectra'):
            spectra,frame_audit=models.frame_spectra(q,task_projection,tasks,len(data.channels),data.dx)
        else:
            shares,error=models.frame_shares(q)
            spectra=np.zeros((len(data.channels),len(shares),data.dx,width));maximum_trace_error=0.
            for t,(c,j) in enumerate(tasks):
                component=shares@task_projection[t]
                eigen=torch.linalg.svdvals(component).square()/data.n
                energy=component.square().sum((1,2))/data.n
                relative=torch.max(torch.abs(eigen.sum(1)-energy)/energy.clamp_min(1e-30))
                maximum_trace_error=max(maximum_trace_error,float(relative))
                if relative>1e-8:raise ArithmeticError('Shapley profile SVD did not preserve component energy')
                spectra[c,:,j,:eigen.shape[1]]=eigen.cpu().numpy()
            frame_audit=dict(relative_squared_efficiency_error=error,maximum_SVD_trace_relative_error=maximum_trace_error)
        np.savez_compressed(output/f'frame{f:03d}_profile_spectra.npz',**{getattr(models,'spectrum_key','squared_singular_values'):spectra})
        numerical.append(dict(frame=f,cached_conditional_models=len(models.models),**frame_audit))
        scores=[]
        for c,channel in enumerate(data.channels):
            prefix=getattr(models,'metric_prefix','shapley')
            values={prefix+'_profile_'+k:v for k,v in mode_support.scores(spectra[c]).items()}
            values.update({prefix+'_energy_'+k:v for k,v in conditional.sparsity(spectra[c].sum(-1)).items()})
            scores.extend(dict(channel=channel,metric=m,loss=value,is_control=not scope.primary(m)) for m,value in values.items())
        frame_scores[f]=scores
        wave.json_file(output/'numerical_audits.json',numerical)
        if (f+1)%4==0 or f+1==len(frame_keys):wave.json_file(output/'conditional_model_audits.json',models.audits)
        print(f'[ShapleyProfiles] {data.index["model"]}: frames {f+1}/{len(frame_keys)}, '
              f'{len(models.models)} conditional subspace models, latent rank {latent.shape[1]}',flush=True)
    if hasattr(models,'finalize'):protocol.update(models.finalize())
    blind=[]
    for k,f in enumerate(frame_index):
        blind.extend(dict(anonymous_case=f'case_{k:03d}' if k<len(data.paths) else f'audit_frame_{k-len(data.paths):03d}',**r) for r in frame_scores[int(f)])
    wave.csv_file(output/'blind_scores.csv',blind)
    digest=hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    reveal=json.loads((original/'reveal.json').read_text())['mapping']
    wave.json_file(output/'reveal.json',dict(scores_sha256=digest,mapping=reveal))
    grouped=defaultdict(dict);controls={};extra=[]
    for r in blind:
        key=r['channel'],r['metric'];controls[key]=r['is_control']
        if r['anonymous_case'] not in reveal:extra.append(r)
        else:grouped[key][reveal[r['anonymous_case']]]=r['loss']
    rows=[dict(channel=c,metric=m,**wave.st.evaluate_curve(losses,data.index['cases']),is_control=controls[c,m],
        baseline_loss=losses['baseline'],minimum_relative_grid_margin=(min(losses.values())-losses['baseline'])/max(abs(losses['baseline']),1e-30))
        for (c,m),losses in grouped.items()]
    wave.csv_file(output/'metric_summary.csv',rows)
    if extra:wave.csv_file(output/'frozen_frame_assessment.csv',[dict(**r,baseline_loss=grouped[r['channel'],r['metric']]['baseline'],
        relative_change_percent=100*(r['loss']-grouped[r['channel'],r['metric']]['baseline'])/
            max(abs(grouped[r['channel'],r['metric']]['baseline']),1e-30)) for r in extra])
    wave.json_file(output/'read_audit.json',dict(files=sorted(set(data.read_files)),forbidden_files_read=[],all_subjects=data.n,all_original_rows=data.m))
    protocol.update(status='COMPLETE',scores_sha256=digest,conditional_models=len(models.models),shared_target_rank=latent.shape[1])
    wave.json_file(output/'protocol.json',protocol)
    print(f'[ShapleyProfiles] {data.index["model"]}: {sum(r["all_planes_pass"] and not r["is_control"] for r in rows)} primary grid combinations',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',action='store_true')
    p.add_argument('--model',choices=wave.io.DIMS,default='CR3');p.add_argument('--polynomial-link',action='store_true')
    p.add_argument('--centers',type=int,default=32);p.add_argument('--degree',type=int,default=3)
    p.add_argument('--run-tag');args=p.parse_args()
    with threadpool_limits(limits=4):
        torch.set_num_threads(4);tag=args.run_tag or ('BV' if args.polynomial_link else 'BU');common=None;all_rows=[]
        evidence=wave.ROOT/f'outputs/supervised_abs_rotation/wave{tag}_Shapley_profiles_four_model_evidence';evidence.mkdir(parents=True,exist_ok=True)
        for model in (('CR3','C1','C2','C3') if args.batch else (args.model,)):
            source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
            original=parent/('waveBR_polynomial_link_kernel_seed430' if args.polynomial_link else 'waveBC2_conditional_kernel_centers32_seed430')
            folder=parent/('matrix_response_polynomial_global_warp_seed438' if args.polynomial_link else 'matrix_response_table_seed383')
            output=parent/f'wave{tag}_conditional_Shapley_profiles_seed471'
            if not output.exists():
                frames,descriptions=audit_frames(parent,model)
                analyze(source,original,folder,output,bank_factory=learned.MatrixBank if args.polynomial_link else profiles.MatrixBank,
                        extra_frames=frames,extra_descriptions=descriptions,degree=args.degree,centers=args.centers)
            protocol=json.loads((output/'protocol.json').read_text())
            if protocol['degree']!=args.degree or protocol['radial_centers']!=args.centers:
                raise ValueError('Existing run uses different conditional regression capacity; choose a new run tag')
            if protocol['status']!='COMPLETE' or hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()!=protocol['scores_sha256']:
                raise ValueError('Shapley profile experiment incomplete or changed')
            with (output/'metric_summary.csv').open() as stream:rows=list(csv.DictReader(stream))
            passed={(r['channel'],r['metric']) for r in rows if r['all_planes_pass']=='True' and r['is_control']!='True'}
            common=passed if common is None else common&passed;all_rows.extend(dict(model=model,**r) for r in rows)
            wave.csv_file(evidence/'all_model_evidence.csv',all_rows)
            wave.json_file(evidence/'policy.json',dict(status='COMPLETE' if not args.batch or model=='C3' else 'RUNNING',
                completed_models=list(dict.fromkeys(r['model'] for r in all_rows)),
                common=[dict(channel=c,metric=m) for c,m in sorted(common)],no_MSE_or_position_veto=True))
            print(f'[Wave{tag}] {model}: {len(passed)} grid combinations, {len(common)} common',flush=True)
