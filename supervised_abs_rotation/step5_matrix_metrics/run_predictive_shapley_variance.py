"""CF: UX sparsity of PRESS predictive-skill Shapley allocations."""
import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as profiles
import conditional_ux as conditional
import ux_metric_scope as scope
from adaptive_coalition_models import AdaptiveCoalitionModels
from predictive_variance_game import PredictiveVarianceGame
from record_rotation_optima import WAVES
from run_operator_mode_support import audit_frames


def frozen_extra_frames(parent,model,additional_tags=()):
    frames,descriptions=audit_frames(parent,model)
    folder=parent/'waveBU6_local_Shapley_refinement_seed489'
    protocol=json.loads((folder/'protocol.json').read_text())
    digest=hashlib.sha256((folder/'refined_frames.npz').read_bytes()).hexdigest()
    if protocol['status']!='COMPLETE' or digest!=protocol['frames_sha256']:
        raise ValueError('Frozen refinement changed')
    if hashlib.sha256((folder/'blind_refined_scores.csv').read_bytes()).hexdigest()!=protocol['scores_sha256']:
        raise ValueError('Frozen refinement scores changed')
    with (folder/'blind_refined_scores.csv').open() as stream:rows=list(csv.DictReader(stream))
    with np.load(folder/'refined_frames.npz',allow_pickle=False) as saved:
        for row,q in zip(rows,saved['frames']):
            frames.append(q.copy());descriptions.append(dict(source='BU6 '+row['start'],frame_sha256=digest))
    for tag in additional_tags:
        tag,seed=tag if isinstance(tag,tuple) else (tag,478)
        folder=parent/f'wave{tag}_fresh_Shapley_blind_seed{seed}'
        if not folder.exists():continue
        protocol=json.loads((folder/'protocol.json').read_text())
        digest=hashlib.sha256((folder/'blind_rotations.npz').read_bytes()).hexdigest()
        if protocol['status']!='COMPLETE' or digest!=protocol['frames_sha256']:
            raise ValueError('Latest frozen blind frame changed or is incomplete')
        if hashlib.sha256((folder/'random_scores.csv').read_bytes()).hexdigest()!=protocol['random_scores_sha256']:
            raise ValueError('Latest frozen blind proposal scores changed')
        with np.load(folder/'blind_rotations.npz',allow_pickle=False) as saved:frames.append(saved['optimized'].copy())
        descriptions.append(dict(source=tag+' frozen blind optimized frame',frame_sha256=digest))
    return frames,descriptions


def analyze(source,output,device='cuda',bank_factory=profiles.MatrixBank,extra_frames=(),extra_descriptions=(),sources=None,
            original=None,bank_folder=None):
    parent=source.parent/'step5_matrix_metrics';original=original or parent/WAVES['BU']
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Retain earlier predictive game: {output}')
    old=json.loads((original/'protocol.json').read_text())
    if old['status']!='COMPLETE' or hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest()!=old['scores_sha256']:
        raise ValueError('Frozen conditional response source changed')
    bank=bank_factory(source,bank_folder or parent/'matrix_response_table_seed383');data=bank.data
    if bank.protocol['manifest_sha256']!=old['source_matrix_manifest_sha256']:raise ValueError('Actual matrix source changed')
    output.mkdir(parents=True)
    protocol=dict(status='RUNNING',model=data.index['model'],all_subjects=data.n,all_original_rows=data.m,
        source_matrix_manifest_sha256=bank.protocol['manifest_sha256'],response_transform='identity',
        source_response_basis_sha256=hashlib.sha256((original/'shared_response_basis.npz').read_bytes()).hexdigest(),
        source_response_quality=bank.quality,seed=471,
        interpretation='finite PRESS predictive skill Shapley allocation with an explicit unassigned-risk row; not calibrated population Sobol indices',
        value_function='max(0, 1-PRESS_MSE(S)/PRESS_MSE(empty)) and maximum over all subcoalitions; clipped to [0,1]',
        full_input_predictor_fixed_across_frames=True,unassigned_risk_included_in_primary_support=True,
        independence_of_inputs_assumed=False,all_subjects_used_in_conditional_fits=True,
        validation_scope='fixed W-only features, output compression, candidate family and ridge; model selection is not nested',
        envelope_selection_bias_independently_calibrated=False,prediction_or_position_veto=False,
        extra_frozen_frames=list(extra_descriptions),
        code_sha256=hashlib.sha256(b''.join((wave.HERE/name).read_bytes() for name in
            ('run_predictive_shapley_variance.py','predictive_variance_game.py','predictive_variance_effects.py','adaptive_coalition_models.py','conditional_model_quality.py'))).hexdigest())
    wave.json_file(output/'protocol.json',protocol)
    for name in ('shared_response_basis.npz','shared_target_compression.json','matrix_function_modes.json'):
        shutil.copyfile(original/name,output/name)
    with np.load(original/'shared_response_basis.npz',allow_pickle=False) as saved:
        latent=saved['latent'].copy();projection=saved['projection'].copy();slices=saved['task_slices'].copy();tasks=saved['tasks'].copy()
    with np.load(original/'W_only_geometry.npz',allow_pickle=False) as saved:
        maps=[saved['directions'][axes].T for axes in saved['membership'][:len(data.paths)]]+list(extra_frames)
    directions,membership=conditional.direction_classes(maps)
    keys=list(dict.fromkeys(tuple(sorted(axes)) for axes in membership))
    frame_index=np.array([keys.index(tuple(sorted(axes))) for axes in membership])
    np.savez_compressed(output/'W_only_geometry.npz',directions=directions,membership=membership,
        frame_axes=np.array(keys),case_frame_index=frame_index)
    y=torch.tensor(latent,dtype=torch.float64,device=device)
    p=[torch.tensor(projection[:,a:b],dtype=torch.float64,device=device) for a,b in slices]
    z=data.u-data.u.mean(0);z/=np.sqrt(np.var(data.u,axis=0).mean())
    sources=sources or {tag:parent/WAVES[tag] for tag in ('BU','BU3','CB','CC')}
    models=AdaptiveCoalitionModels(z,y,output/'conditional_models',sources,device=device)
    if models.protocol_info['source_matrix_manifest_sha256']!=bank.protocol['manifest_sha256']:
        raise ValueError('Adaptive predictors use different actual matrices')
    protocol.update(models.protocol_info);wave.json_file(output/'protocol.json',protocol)
    predictive=PredictiveVarianceGame(models,p,tasks,len(data.channels),data.dx);frame_scores={};numerical=[]
    wave.json_file(output/'empty_and_full_risks.json',dict(empty_PRESS_MSE=predictive.empty.tolist(),full_PRESS_MSE=predictive.full.tolist()))
    for f,axes in enumerate(keys):
        q=directions[list(axes)].T;effects,raw,game,audit=predictive.frame(q)
        np.savez_compressed(output/f'frame{f:03d}_predictive_variance.npz',UX_variance_allocation=effects,
            raw_predictive_skill=raw,monotone_predictive_skill=game)
        rows=[]
        for c,channel in enumerate(data.channels):
            rows.extend(dict(channel=channel,metric='PRESS_variance_'+m,loss=v,is_control=not scope.primary(m))
                for m,v in conditional.sparsity(effects[:,c,:]).items())
        frame_scores[f]=rows;numerical.append(dict(frame=f,cached_risk_subspaces=len(predictive.risks),**audit))
        wave.json_file(output/'predictive_game_audits.json',numerical)
        if (f+1)%4==0 or f+1==len(keys):
            wave.json_file(output/'conditional_model_audits.json',models.audits)
            wave.json_file(output/'fixed_fit_PRESS_audits.json',predictive.audits)
        print(f'[PredictiveVariance] {data.index["model"]}: frames {f+1}/{len(keys)}, {len(predictive.risks)} PRESS subspaces',flush=True)
    protocol.update(models.finalize());blind=[]
    for k,f in enumerate(frame_index):
        blind.extend(dict(anonymous_case=f'case_{k:03d}' if k<len(data.paths) else f'audit_frame_{k-len(data.paths):03d}',**r)
            for r in frame_scores[int(f)])
    wave.csv_file(output/'blind_scores.csv',blind);digest=hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    reveal=json.loads((original/'reveal.json').read_text())['mapping']
    wave.json_file(output/'reveal.json',dict(scores_sha256=digest,mapping=reveal))
    grouped=defaultdict(dict);controls={};extra=[]
    for row in blind:
        key=row['channel'],row['metric'];controls[key]=row['is_control']
        if row['anonymous_case'] in reveal:grouped[key][reveal[row['anonymous_case']]]=row['loss']
        else:extra.append(row)
    rows=[dict(channel=c,metric=m,**wave.st.evaluate_curve(losses,data.index['cases']),is_control=controls[c,m],
        baseline_loss=losses['baseline'],minimum_relative_grid_margin=(min(losses.values())-losses['baseline'])/max(abs(losses['baseline']),1e-30))
        for (c,m),losses in grouped.items()]
    wave.csv_file(output/'metric_summary.csv',rows)
    if extra:wave.csv_file(output/'frozen_frame_assessment.csv',[dict(**r,baseline_loss=grouped[r['channel'],r['metric']]['baseline'],
        relative_change_percent=100*(r['loss']-grouped[r['channel'],r['metric']]['baseline'])/max(abs(grouped[r['channel'],r['metric']]['baseline']),1e-30)) for r in extra])
    wave.json_file(output/'read_audit.json',dict(files=sorted(set(data.read_files)),forbidden_files_read=[],all_subjects=data.n,all_original_rows=data.m))
    protocol.update(status='COMPLETE',scores_sha256=digest,conditional_models=len(models.models),shared_target_rank=latent.shape[1])
    wave.json_file(output/'protocol.json',protocol)


def main(models):
    for model in models:
        source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
        output=parent/'waveCF_PRESS_Shapley_variance_seed471'
        if not output.exists():
            frames,descriptions=frozen_extra_frames(parent,model)
            analyze(source,output,extra_frames=frames,extra_descriptions=descriptions)
        protocol=json.loads((output/'protocol.json').read_text())
        if protocol['status']!='COMPLETE' or hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()!=protocol['scores_sha256']:
            raise ValueError('Predictive variance run incomplete or changed')
        print(f'[WaveCF] {model} COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',action='store_true')
    parser.add_argument('--model',choices=wave.io.DIMS,default='CR3');args=parser.parse_args()
    with threadpool_limits(4):
        torch.set_num_threads(4);main(('CR3','C1','C2','C3') if args.batch else (args.model,))
