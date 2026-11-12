"""BH: evidence-selected U-subset functions and their UX participation support."""
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
import group_sparse_gp as gp
import conditional_ui_witness as ui
import conditional_ux as conditional
import matrix_function_profiles as profiles
import tail_support
import ux_metric_scope as scope


def sufficient(phi,target,rows,center_target=False):
    x=phi[rows];y=target[:,rows]
    mean=y.mean(1,keepdim=True) if center_target else torch.zeros_like(y[:,:1])
    y=y-mean
    return x.T@x,x.T@y,y.square().sum((1,2)),mean


def analyze(source,original,folder,output,counterexample=None,degrees=(3,5),radial_centers=32,
            training_steps=60,refit_steps=120,seed=448,device='cuda',dictionary_factory=gp.dictionary):
    original,output=Path(original),Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Retain earlier group GP run: {output}')
    old=json.loads((original/'protocol.json').read_text())
    if old['status']!='COMPLETE' or hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest()!=old['scores_sha256']:
        raise ValueError('Original matrix source experiment changed')
    bank=profiles.MatrixBank(source,folder);data=bank.data
    if bank.protocol['manifest_sha256']!=old['source_matrix_manifest_sha256']:raise ValueError('Matrix response cache changed')
    output.mkdir(parents=True)
    protocol=dict(status='RUNNING',model=data.index['model'],all_subjects=data.n,all_original_rows=data.m,
        seed=seed,degrees=list(degrees),radial_centers=radial_centers,training_steps=training_steps,refit_steps=refit_steps,
        prior='one learned coefficient variance per anonymous U-subset, normalized group feature trace',
        function_ensemble='equal original energy per X/channel; expected GP density, not independent mode copies',
        source_matrix_manifest_sha256=bank.protocol['manifest_sha256'],source_scores_sha256=old['scores_sha256'],
        source_response_quality=bank.quality,prediction_quality_policy='continuous, no fixed MSE veto',
        validation_interpretation='subject-heldout response functions, conditional on fixed W-only radial dictionary and target compression',
        graph='sum posterior component-function energies over all subsets containing Ui for each Xj',
        uncertainty_variant='also add posterior coefficient covariance contribution',
        decomposition_caveat='subset components can overlap under dependent empirical U; this is not unique functional ANOVA',
        dictionary_factory=dictionary_factory.__module__+'.'+dictionary_factory.__name__,
        dictionary_code_sha256=hashlib.sha256(Path(dictionary_factory.__code__.co_filename).read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in ('run_group_sparse_gp.py','group_sparse_gp.py'))).hexdigest())
    wave.json_file(output/'protocol.json',protocol)
    with np.load(original/'W_only_geometry.npz',allow_pickle=False) as saved:
        dirs,old_membership=saved['directions'],saved['membership']
    maps=[dirs[axes].T for axes in old_membership]
    if counterexample is not None:
        counterexample=Path(counterexample);frozen=json.loads((counterexample/'blind_freeze.json').read_text())
        digest=hashlib.sha256((counterexample/'blind_rotations.npz').read_bytes()).hexdigest()
        if digest!=frozen['rotations_sha256']:raise ValueError('Frozen counterexample changed')
        with np.load(counterexample/'blind_rotations.npz',allow_pickle=False) as saved:maps.append(saved['optimized'].copy())
        protocol['fixed_counterexample_sha256']=digest
    directions,membership=conditional.direction_classes(maps)
    frame_keys=list(dict.fromkeys(tuple(sorted(axes)) for axes in membership))
    frame_index=np.array([frame_keys.index(tuple(sorted(axes))) for axes in membership])
    np.savez_compressed(output/'W_only_geometry.npz',directions=directions,membership=membership,
                        frame_axes=np.array(frame_keys),case_frame_index=frame_index)
    operators=[np.load(original/f'X_witness_column{j}.npy',allow_pickle=False) for j in range(data.dx)]
    tasks,targets,discarded,mode_audit=profiles.task_modes(bank.responses,operators,tolerance=1e-8)
    wave.json_file(output/'matrix_function_modes.json',dict(tasks=mode_audit,
        maximum_discarded_relative_energy=float(np.max(discarded.sum(1)/data.n))))
    width=max(t.shape[1] for t in targets);values=np.zeros((len(tasks),data.n,width))
    for t,target in enumerate(targets):values[t,:,:target.shape[1]]=target
    target=torch.tensor(values,dtype=torch.float64,device=device)
    z=data.u-data.u.mean(0);z/=np.sqrt(np.var(data.u,axis=0).mean())
    radial,radial_audit=ui.nystrom(z,radial_centers,seed)
    wave.json_file(output/'W_only_radial_audit.json',radial_audit)
    order=np.random.default_rng(seed+1).permutation(data.n);train=np.sort(order[:int(.75*data.n)]);validation=np.sort(order[int(.75*data.n):])
    all_rows=np.arange(data.n);scores,quality,histories=[],[],[]
    for f,axes in enumerate(frame_keys):
        rotated=z@directions[list(axes)].T
        for degree in degrees:
            phi_np,groups,masks,feature_audit=dictionary_factory(rotated,degree,radial,reference=train)
            phi=torch.tensor(phi_np,dtype=torch.float64,device=device);group=torch.tensor(groups,device=device)
            gram,cross,yy,mean=sufficient(phi,target,train,center_target=True)
            starts=[]
            for initial_noise in (.03,.3):
                state,loss,history=gp.fit(gram,cross,yy,group,len(train),steps=training_steps,noise_initial=initial_noise)
                starts.append((state,loss));histories.append(dict(frame=f,degree=degree,phase='training',initial_noise=initial_noise,history=history))
            selected=np.argmin(np.array([value[1] for value in starts]),axis=0)
            state=(torch.stack([value[0][0][t] for t,value in enumerate([starts[s] for s in selected])]),
                   torch.stack([value[0][1][t] for t,value in enumerate([starts[s] for s in selected])]))
            _,coef,_=gp.system(gram,cross,yy,group,*state,len(train))
            prediction=phi[validation]@coef+mean
            validation_error=(prediction-target[:,validation]).square().sum((1,2))/target[:,validation].square().sum((1,2)).clamp_min(1e-30)
            full_np,full_groups,full_masks,full_audit=dictionary_factory(rotated,degree,radial)
            if not np.array_equal(full_groups,groups):raise ArithmeticError('Group dictionary changed its subset indexing')
            scale_ratio=torch.tensor(np.array(full_audit['group_scales'])/np.array(feature_audit['group_scales']),device=device)
            initial=(state[0]*scale_ratio.square(),state[1])
            phi=torch.tensor(full_np,dtype=torch.float64,device=device)
            gram,cross,yy,_=sufficient(phi,target,all_rows)
            state,loss,history=gp.fit(gram,cross,yy,group,data.n,steps=refit_steps,initial=initial)
            histories.append(dict(frame=f,degree=degree,phase='all_subject_refit',history=history))
            _,coef,covariance=gp.system(gram,cross,yy,group,*state,data.n)
            point,posterior,components=gp.component_energies(gram,coef,covariance,group,masks,data.n,data.d)
            refit_error=(phi@coef-target).square().sum((1,2))/target.square().sum((1,2)).clamp_min(1e-30)
            np.savez_compressed(output/f'frame{f:03d}_degree{degree}.npz',group_masks=masks,group_variances=state[0].cpu().numpy(),
                noise_variances=state[1].cpu().numpy(),component_energies=components,mean_U_energies=point,posterior_U_energies=posterior,
                evidence_per_subject=loss/data.n,training_selected_start=selected)
            for t,(c,j) in enumerate(tasks):quality.append(dict(frame=f,degree=degree,channel=data.channels[c],X_column=j,
                subject_validation_nmse=float(validation_error[t]),all_subject_refit_nmse=float(refit_error[t])))
            for c,channel in enumerate(data.channels):
                ids=[t for t,(cc,j) in enumerate(tasks) if cc==c]
                for variant,energy in (('posterior_mean',point),('posterior_expected',posterior)):
                    for metric,value in conditional.sparsity(energy[ids].T).items():
                        scores.append(dict(frame=f,channel=channel,metric=f'poly{degree}_rbf{radial_centers}_{variant}_'+metric,
                                           loss=value,is_control=not scope.primary(metric)))
                group_support=[tail_support.scores(components[t]) for t in ids]
                for metric in group_support[0]:
                    for label,func in (('mean',np.mean),('worst',np.max)):
                        scores.append(dict(frame=f,channel=channel,metric=f'poly{degree}_rbf{radial_centers}_component_groups_X{label}_'+metric,
                            loss=float(func([s[metric] for s in group_support])),is_control=not scope.primary(metric)))
                for label,func in (('mean',np.mean),('worst',np.max)):
                    scores.append(dict(frame=f,channel=channel,metric=f'poly{degree}_rbf{radial_centers}_ensemble_GP_NLL_X{label}',
                        loss=float(func(loss[ids])/data.n),is_control=False))
        wave.csv_file(output/'function_quality.csv',quality)
        wave.json_file(output/'optimization_histories.json',histories)
        print(f'[WaveBH] {data.index["model"]}: subset GP frames {f+1}/{len(frame_keys)}',flush=True)
    frame_scores=defaultdict(list)
    for r in scores:frame_scores[r['frame']].append(r)
    blind=[]
    for k,index in enumerate(frame_index):
        for r in frame_scores[index]:blind.append(dict(anonymous_case=f'case_{k:03d}' if k<len(data.paths) else 'audit_frame_000',
            **{key:r[key] for key in ('channel','metric','loss','is_control')}))
    wave.csv_file(output/'blind_scores.csv',blind)
    digest=hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    reveal=json.loads((original/'reveal.json').read_text())['mapping']
    wave.json_file(output/'reveal.json',dict(scores_sha256=digest,mapping=reveal))
    grouped=defaultdict(dict);controls={};extras={}
    for r in blind:
        key=r['channel'],r['metric'];controls[key]=r['is_control']
        if r['anonymous_case']=='audit_frame_000':extras[key]=r['loss']
        else:grouped[key][reveal[r['anonymous_case']]]=r['loss']
    rows=[dict(channel=c,metric=m,**wave.st.evaluate_curve(v,data.index['cases']),is_control=controls[c,m],baseline_loss=v['baseline'],
        minimum_relative_grid_margin=(min(v.values())-v['baseline'])/max(abs(v['baseline']),1e-30)) for (c,m),v in grouped.items()]
    wave.csv_file(output/'metric_summary.csv',rows)
    if extras:wave.csv_file(output/'frozen_counter_assessment.csv',[dict(channel=c,metric=m,loss=value,
        baseline_loss=grouped[c,m]['baseline'],relative_margin=(value-grouped[c,m]['baseline'])/max(abs(grouped[c,m]['baseline']),1e-30))
        for (c,m),value in extras.items()])
    wave.json_file(output/'read_audit.json',dict(files=sorted(set(data.read_files)),forbidden_files_read=[],
        all_subjects=data.n,all_original_rows=data.m))
    protocol.update(status='COMPLETE',scores_sha256=digest)
    wave.json_file(output/'protocol.json',protocol)
    print(f'[WaveBH] {data.index["model"]}: {sum(r["all_planes_pass"] and not r["is_control"] for r in rows)} primary grid combinations',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',action='store_true');args=p.parse_args()
    with threadpool_limits(limits=4):
        torch.set_num_threads(4);common=None;all_rows=[]
        evidence=wave.ROOT/'outputs/supervised_abs_rotation/waveBH_group_sparse_GP_four_model_evidence';evidence.mkdir(parents=True,exist_ok=True)
        for model in (('CR3','C1','C2','C3') if args.batch else ('CR3',)):
            source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics';output=parent/'waveBH_group_sparse_GP_seed448'
            if not output.exists():analyze(source,parent/'waveBC2_conditional_kernel_centers32_seed430',parent/'matrix_response_table_seed383',output,
                counterexample=parent/'waveBC3_conditional_kernel_blind_seed439' if model=='CR3' else None)
            protocol=json.loads((output/'protocol.json').read_text())
            if protocol['status']!='COMPLETE' or hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()!=protocol['scores_sha256']:
                raise ValueError('Group GP experiment incomplete or changed')
            with (output/'metric_summary.csv').open() as stream:rows=list(csv.DictReader(stream))
            passed={(r['channel'],r['metric']) for r in rows if r['all_planes_pass']=='True' and r['is_control']!='True'}
            common=passed if common is None else common&passed;all_rows.extend(dict(model=model,**r) for r in rows)
            wave.csv_file(evidence/'all_model_evidence.csv',all_rows)
            wave.json_file(evidence/'policy.json',dict(status='COMPLETE' if model==('C3' if args.batch else 'CR3') else 'RUNNING',
                completed_models=list(dict.fromkeys(r['model'] for r in all_rows)),common=[dict(channel=c,metric=m) for c,m in sorted(common)],fixed_MSE_veto=False))
            print(f'[WaveBH] {model}: {len(passed)} grid combinations, {len(common)} common',flush=True)
