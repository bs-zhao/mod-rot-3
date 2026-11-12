"""BG: contextual Ui innovations versus crossfitted conditional response residuals."""
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
import conditional_ui_witness as ui
import conditional_double_residual as double
import conditional_ux as conditional
import run_conditional_kernel_moments as kernel
import ux_metric_scope as scope


def analyze(source, original, folder, output, counterexample=None, capacities=((3,64),(5,128)),
            centers=32,device='cuda'):
    original,output = Path(original),Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain earlier double residual experiment: {output}')
    old = json.loads((original/'protocol.json').read_text())
    if old['status']!='COMPLETE' or hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest()!=old['scores_sha256']:
        raise ValueError('Frozen original conditional kernel source changed')
    bank = profiles.MatrixBank(source,folder);data=bank.data
    if bank.protocol['manifest_sha256']!=old['source_matrix_manifest_sha256']:
        raise ValueError('Actual response source changed')
    output.mkdir(parents=True)
    protocol = dict(status='RUNNING',model=data.index['model'],all_subjects=data.n,all_original_rows=data.m,
        source_scores_sha256=old['scores_sha256'],source_matrix_manifest_sha256=bank.protocol['manifest_sha256'],
        capacities=[list(v) for v in capacities],Ui_centers=centers,Ui_seed=430,response_seed=444,
        response_definition='actual UX function targets minus nested crossfitted prediction from U without Ui',
        conditional_feature_definition='unchanged BC Ui innovations and optional other-U context',
        task_normalization='unit original response energy per X/channel before regression; no per-direction residual renormalization',
        interpretation='exploratory conditional moment; not a calibrated KCI test or exact Sobol index',
        prediction_quality_policy='continuous, no fixed MSE veto',source_response_quality=bank.quality,
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_double_residual_kernel.py','conditional_double_residual.py','conditional_ui_witness.py'))).hexdigest())
    wave.json_file(output/'protocol.json',protocol)
    with np.load(original/'W_only_geometry.npz',allow_pickle=False) as saved:
        original_directions,original_membership=saved['directions'],saved['membership']
    maps=[original_directions[axes].T for axes in original_membership]
    if counterexample is not None:
        counterexample=Path(counterexample)
        frozen=json.loads((counterexample/'blind_freeze.json').read_text())
        digest=hashlib.sha256((counterexample/'blind_rotations.npz').read_bytes()).hexdigest()
        if digest!=frozen['rotations_sha256']:
            raise ValueError('Frozen counterexample changed')
        with np.load(counterexample/'blind_rotations.npz',allow_pickle=False) as saved:
            maps.append(saved['optimized'].copy())
        protocol['fixed_counterexample_sha256']=digest
    directions,membership=conditional.direction_classes(maps)
    np.savez_compressed(output/'W_only_geometry.npz',directions=directions,membership=membership)
    operators=[np.load(original/f'X_witness_column{j}.npy',allow_pickle=False) for j in range(data.dx)]
    tasks,targets,discarded,mode_audit=profiles.task_modes(bank.responses,operators,tolerance=1e-8)
    wave.json_file(output/'matrix_function_modes.json',dict(tasks=mode_audit,
        maximum_discarded_relative_energy=float(np.max(discarded.sum(1)/data.n))))
    width=max(t.shape[1] for t in targets);values=np.zeros((data.n,len(tasks),width))
    for t,target in enumerate(targets):values[:,t,:target.shape[1]]=target
    target=torch.tensor(values.reshape(data.n,-1),dtype=torch.float64,device=device)
    z=data.u-data.u.mean(0);z/=np.sqrt(np.var(data.u,axis=0).mean())
    graphs,audits={},[]
    for i,v in enumerate(directions):
        features,ua=ui.features(z,v,seed=430,centers=centers)
        for degree,radial_centers in capacities:
            residual,ya=double.residual_targets(z,v,target,ua['context_kernel']['lengthscale'],
                degree=degree,centers=radial_centers)
            norm=residual.reshape(data.n,len(tasks),width).square().sum(-1)
            for view,feature in features.items():
                raw,corrected=kernel.moments(torch.tensor(feature,dtype=torch.float64,device=device),residual,norm,width)
                for estimator,scores in (('raw',raw),('diagonal_removed',corrected)):
                    key=f'poly{degree}_rbf{radial_centers}_{view}_{estimator}'
                    if key not in graphs:graphs[key]=np.zeros((len(data.channels),len(directions),data.dx))
                    for t,(c,j) in enumerate(tasks):graphs[key][c,i,j]=scores[t]
            audits.append(dict(direction=i,capacity=[degree,radial_centers],Ui_audit=ua,response_audit=ya))
        if (i+1)%4==0 or i+1==len(directions):
            wave.json_file(output/'crossfit_audits.json',audits)
            print(f'[WaveBG] {data.index["model"]}: double residual directions {i+1}/{len(directions)}',flush=True)
    np.savez_compressed(output/'signed_conditional_moments.npz',**graphs)
    blind=[]
    for view,graph in graphs.items():
        for k,axes in enumerate(membership):
            for c,channel in enumerate(data.channels):
                blind.extend(dict(anonymous_case=f'case_{k:03d}' if k<len(data.paths) else 'audit_frame_000',
                    channel=channel,metric=view+'_'+m,loss=value) for m,value in
                    conditional.sparsity(np.maximum(graph[c,axes],0.)).items())
    wave.csv_file(output/'blind_scores.csv',blind)
    digest=hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    reveal=json.loads((original/'reveal.json').read_text())['mapping']
    wave.json_file(output/'reveal.json',dict(scores_sha256=digest,mapping=reveal))
    grouped=defaultdict(dict);extras={}
    for r in blind:
        if r['anonymous_case']=='audit_frame_000':extras[r['channel'],r['metric']]=r['loss']
        else:grouped[r['channel'],r['metric']][reveal[r['anonymous_case']]]=r['loss']
    rows=[dict(channel=c,metric=m,**wave.st.evaluate_curve(losses,data.index['cases']),
        is_control=not scope.primary(m),baseline_loss=losses['baseline'],
        minimum_relative_grid_margin=(min(losses.values())-losses['baseline'])/max(abs(losses['baseline']),1e-30))
        for (c,m),losses in grouped.items()]
    wave.csv_file(output/'metric_summary.csv',rows)
    if extras:
        wave.csv_file(output/'frozen_counter_assessment.csv',[dict(channel=c,metric=m,loss=value,
            baseline_loss=grouped[c,m]['baseline'],relative_margin=(value-grouped[c,m]['baseline'])/
            max(abs(grouped[c,m]['baseline']),1e-30)) for (c,m),value in extras.items()])
    wave.json_file(output/'read_audit.json',dict(files=sorted(set(data.read_files)),forbidden_files_read=[],
        all_subjects=data.n,all_original_rows=data.m,derived_source_run=str(original.resolve())))
    protocol.update(status='COMPLETE',scores_sha256=digest)
    wave.json_file(output/'protocol.json',protocol)
    print(f'[WaveBG] {data.index["model"]}: {sum(r["all_planes_pass"] and not r["is_control"] for r in rows)} primary grid combinations',flush=True)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',action='store_true')
    args=p.parse_args()
    with threadpool_limits(limits=4):
        torch.set_num_threads(4);all_rows=[];common=None
        evidence=wave.ROOT/'outputs/supervised_abs_rotation/waveBG_double_residual_four_model_evidence'
        evidence.mkdir(parents=True,exist_ok=True)
        for model in (('CR3','C1','C2','C3') if args.batch else ('CR3',)):
            source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
            output=parent/'waveBG_double_residual_kernel_seed444'
            if not output.exists():
                analyze(source,parent/'waveBC2_conditional_kernel_centers32_seed430',parent/'matrix_response_table_seed383',
                    output,counterexample=parent/'waveBC3_conditional_kernel_blind_seed439' if model=='CR3' else None)
            protocol=json.loads((output/'protocol.json').read_text())
            if protocol['status']!='COMPLETE' or hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()!=protocol['scores_sha256']:
                raise ValueError('Double residual experiment incomplete or changed')
            with (output/'metric_summary.csv').open() as stream:rows=list(csv.DictReader(stream))
            passed={(r['channel'],r['metric']) for r in rows if r['all_planes_pass']=='True' and r['is_control']!='True'}
            common=passed if common is None else common&passed
            all_rows.extend(dict(model=model,**r) for r in rows)
            wave.csv_file(evidence/'all_model_evidence.csv',all_rows)
            wave.json_file(evidence/'policy.json',dict(status='COMPLETE' if model==('C3' if args.batch else 'CR3') else 'RUNNING',
                completed_models=list(dict.fromkeys(r['model'] for r in all_rows)),
                common=[dict(channel=c,metric=m) for c,m in sorted(common)],fixed_MSE_veto=False))
            print(f'[WaveBG] {model}: {len(passed)} grid combinations, {len(common)} common',flush=True)
