"""BS/BT: conditional UX operator spectra measure edge-and-function sparsity."""
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
import conditional_ui_witness as ui
import conditional_ux as conditional
import operator_mode_support as mode_support
import ux_metric_scope as scope


def audit_frames(parent,model):
    frames=[];descriptions=[]
    if model=='CR3':
        for name in ('waveBC3_conditional_kernel_blind_seed439','waveBK2_complete_response_blind_seed456'):
            folder=parent/name;frozen=json.loads((folder/'blind_freeze.json').read_text())
            digest=hashlib.sha256((folder/'blind_rotations.npz').read_bytes()).hexdigest()
            if digest!=frozen['rotations_sha256']:raise ValueError('Previously frozen frame changed')
            with np.load(folder/'blind_rotations.npz',allow_pickle=False) as saved:frames.append(saved['optimized'].copy())
            descriptions.append(dict(source=name,frame_sha256=digest))
    folder=wave.ROOT/'outputs/supervised_abs_rotation/waveBP2_maximum_decrease_GP_refinement'/model
    frozen=json.loads((folder/'blind_freeze.json').read_text())
    digest=hashlib.sha256((folder/'refined_frames.npz').read_bytes()).hexdigest()
    if digest!=frozen['frames_sha256']:raise ValueError('Frozen GP matrix search changed')
    with (folder/'refined_scores.csv').open() as stream:rows=list(csv.DictReader(stream))
    eligible=[k for k,r in enumerate(rows) if r['arm']=='blind' and float(r['epsilon'])==.03]
    k=min(eligible,key=lambda k:float(rows[k]['loss']))
    with np.load(folder/'refined_frames.npz',allow_pickle=False) as saved:frames.append(saved['frames'][k].copy())
    descriptions.append(dict(source='BP2 blind eps=.03',frame_sha256=digest,index=k))
    return frames,descriptions


def analyze(source,original,bank_folder,output,bank_factory=profiles.MatrixBank,extra_frames=(),extra_descriptions=(),device='cuda'):
    original,output=Path(original),Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Retain earlier operator spectrum run: {output}')
    old=json.loads((original/'protocol.json').read_text())
    if old['status']!='COMPLETE' or hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest()!=old['scores_sha256']:
        raise ValueError('Frozen geometry/source wave changed')
    bank=bank_factory(source,bank_folder);data=bank.data
    if bank.protocol['manifest_sha256']!=old['source_matrix_manifest_sha256']:raise ValueError('Actual response source changed')
    output.mkdir(parents=True)
    protocol=dict(status='RUNNING',model=data.index['model'],all_subjects=data.n,all_original_rows=data.m,
        Ui_centers=32,seed=430,source_response_quality=bank.quality,
        source_matrix_manifest_sha256=bank.protocol['manifest_sha256'],source_scores_sha256=old['scores_sha256'],
        response_transform=bank.protocol.get('response_transform','identity'),
        statistic='squared singular values of B_i.T G_j/N, with unchanged BC conditional Ui features and actual X-response modes',
        target='per-X support of Ui-by-functional-mode entries; a rank-one edge reduces to ordinary UX edge support',
        primary_view='Ui innovation with other-U context; marginal-only view is an interaction-blind control',
        prediction_quality_policy='continuous, no fixed MSE veto',extra_frozen_frames=list(extra_descriptions),
        interpretation='finite conditional operator complexity diagnostic; not a calibrated rank, CI or Sobol test',
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in ('run_operator_mode_support.py','operator_mode_support.py'))).hexdigest())
    wave.json_file(output/'protocol.json',protocol)
    with np.load(original/'W_only_geometry.npz',allow_pickle=False) as saved:
        directions,old_membership=saved['directions'],saved['membership']
    # BR includes one audit frame already; only its actual case frames form
    # the common grid. New explicitly frozen stress frames are appended here.
    maps=[directions[axes].T for axes in old_membership[:len(data.paths)]]+list(extra_frames)
    directions,membership=conditional.direction_classes(maps)
    np.savez_compressed(output/'W_only_geometry.npz',directions=directions,membership=membership)
    operators=[np.load(original/f'X_witness_column{j}.npy',allow_pickle=False) for j in range(data.dx)]
    tasks,targets,discarded,audits=profiles.task_modes(bank.responses,operators,tolerance=1e-8)
    wave.json_file(output/'matrix_function_modes.json',dict(tasks=audits,maximum_discarded_relative_energy=float(np.max(discarded.sum(1)/data.n))))
    width=max(t.shape[1] for t in targets);values=np.zeros((data.n,len(tasks),width))
    for t,target in enumerate(targets):values[:,t,:target.shape[1]]=target
    target=torch.tensor(values.reshape(data.n,-1),dtype=torch.float64,device=device)
    z=data.u-data.u.mean(0);z/=np.sqrt(np.var(data.u,axis=0).mean())
    spectra={};feature_audits=[];maximum_trace_error=0.
    for i,v in enumerate(directions):
        features,audit=ui.features(z,v,seed=430,centers=32);feature_audits.append(dict(direction=i,**audit))
        for view,feature in features.items():
            cross=torch.tensor(feature,dtype=torch.float64,device=device).T@target/data.n
            if view not in spectra:spectra[view]=np.zeros((len(data.channels),len(directions),data.dx,width))
            for t,(c,j) in enumerate(tasks):
                matrix=cross[:,t*width:t*width+targets[t].shape[1]]
                singular=torch.linalg.svdvals(matrix);eigen=singular.square().cpu().numpy()
                expected=float(matrix.square().sum());observed=float(eigen.sum())
                relative=abs(observed-expected)/max(expected,1e-30);maximum_trace_error=max(maximum_trace_error,relative)
                if relative>1e-8:raise ArithmeticError('Direct operator SVD did not preserve its raw moment trace')
                spectra[view][c,i,j,:len(eigen)]=eigen
        if (i+1)%4==0 or i+1==len(directions):
            wave.json_file(output/'U_feature_audits.json',feature_audits)
            print(f'[OperatorSpectra] {data.index["model"]}: directions {i+1}/{len(directions)}',flush=True)
    np.savez_compressed(output/'operator_squared_singular_values.npz',**spectra)
    wave.json_file(output/'numerical_audit.json',dict(maximum_relative_SVD_trace_error=maximum_trace_error))
    rows=[]
    for view,values in spectra.items():
        for k,axes in enumerate(membership):
            for c,channel in enumerate(data.channels):
                for metric,value in mode_support.scores(values[c,axes]).items():
                    rows.append(dict(anonymous_case=f'case_{k:03d}' if k<len(data.paths) else f'audit_frame_{k-len(data.paths):03d}',
                        channel=channel,metric=view+'_'+metric,loss=value,
                        is_control=(view=='Ui_innovation' or not scope.primary(metric))))
    wave.csv_file(output/'blind_scores.csv',rows)
    digest=hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    reveal=json.loads((original/'reveal.json').read_text())['mapping']
    wave.json_file(output/'reveal.json',dict(scores_sha256=digest,mapping=reveal))
    grouped=defaultdict(dict);extra=[];controls={}
    for r in rows:
        key=r['channel'],r['metric'];controls[key]=r['is_control']
        if r['anonymous_case'] not in reveal:extra.append(r)
        else:grouped[key][reveal[r['anonymous_case']]]=r['loss']
    summary=[dict(channel=c,metric=m,**wave.st.evaluate_curve(losses,data.index['cases']),is_control=controls[c,m],
        baseline_loss=losses['baseline'],minimum_relative_grid_margin=(min(losses.values())-losses['baseline'])/max(abs(losses['baseline']),1e-30))
        for (c,m),losses in grouped.items()]
    wave.csv_file(output/'metric_summary.csv',summary)
    if extra:wave.csv_file(output/'frozen_frame_assessment.csv',[dict(**r,baseline_loss=grouped[r['channel'],r['metric']]['baseline'],
        relative_change_percent=100*(r['loss']-grouped[r['channel'],r['metric']]['baseline'])/
            max(abs(grouped[r['channel'],r['metric']]['baseline']),1e-30)) for r in extra])
    wave.json_file(output/'read_audit.json',dict(files=sorted(set(data.read_files)),forbidden_files_read=[],all_subjects=data.n,all_original_rows=data.m))
    protocol.update(status='COMPLETE',scores_sha256=digest);wave.json_file(output/'protocol.json',protocol)
    print(f'[OperatorSpectra] {data.index["model"]}: {sum(r["all_planes_pass"] and not r["is_control"] for r in summary)} primary grid combinations',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--polynomial-link',action='store_true');args=p.parse_args()
    with threadpool_limits(limits=4):
        torch.set_num_threads(4);tag='BT' if args.polynomial_link else 'BS';all_rows=[];common=None
        evidence=wave.ROOT/f'outputs/supervised_abs_rotation/wave{tag}_operator_modes_four_model_evidence';evidence.mkdir(parents=True,exist_ok=True)
        for model in ('CR3','C1','C2','C3'):
            source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
            original=parent/('waveBR_polynomial_link_kernel_seed430' if args.polynomial_link else 'waveBC2_conditional_kernel_centers32_seed430')
            bank_folder=parent/('matrix_response_polynomial_global_warp_seed438' if args.polynomial_link else 'matrix_response_table_seed383')
            output=parent/f'wave{tag}_operator_mode_support_seed430'
            if not output.exists():
                frames,descriptions=audit_frames(parent,model)
                analyze(source,original,bank_folder,output,bank_factory=learned.MatrixBank if args.polynomial_link else profiles.MatrixBank,
                        extra_frames=frames,extra_descriptions=descriptions)
            protocol=json.loads((output/'protocol.json').read_text())
            if protocol['status']!='COMPLETE' or hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()!=protocol['scores_sha256']:
                raise ValueError('Operator spectrum experiment incomplete or changed')
            with (output/'metric_summary.csv').open() as stream:rows=list(csv.DictReader(stream))
            passed={(r['channel'],r['metric']) for r in rows if r['all_planes_pass']=='True' and r['is_control']!='True'}
            common=passed if common is None else common&passed;all_rows.extend(dict(model=model,**r) for r in rows)
            wave.csv_file(evidence/'all_model_evidence.csv',all_rows)
            wave.json_file(evidence/'policy.json',dict(status='COMPLETE' if model=='C3' else 'RUNNING',
                completed_models=list(dict.fromkeys(r['model'] for r in all_rows)),
                common=[dict(channel=c,metric=m) for c,m in sorted(common)],no_MSE_or_position_veto=True))
            print(f'[Wave{tag}] {model}: {len(passed)} grid combinations, {len(common)} common',flush=True)
