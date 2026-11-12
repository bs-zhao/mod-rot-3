"""BY: scale-free UX detail/axis support in empirical orthogonal tree profiles."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as profiles
import conditional_ux as conditional
import tree_haar_profiles as haar
import ux_metric_scope as scope


def analyze(source,original,output,bank_folder=None,fresh_blind=None):
    original,output=Path(original),Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Retain previous Haar experiment: {output}')
    old=json.loads((original/'protocol.json').read_text())
    if old['status']!='COMPLETE' or hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest()!=old['scores_sha256']:
        raise ValueError('Frozen observed-profile source changed')
    bank=profiles.MatrixBank(source,bank_folder or original.parent/'matrix_response_table_seed383');data=bank.data
    if bank.protocol['manifest_sha256']!=old['source_matrix_manifest_sha256']:raise ValueError('Observed matrix source changed')
    with np.load(original/'shared_response_basis.npz',allow_pickle=False) as saved:
        target=saved['latent'].copy();projection=saved['projection'];slices=saved['task_slices'];tasks=saved['tasks']
    channel=list(data.channels).index('p_nn')
    projections=[(int(j),projection[:,a:b].copy()) for (c,j),(a,b) in zip(tasks,slices) if c==channel]
    with np.load(original/'W_only_geometry.npz',allow_pickle=False) as saved:
        maps=[saved['directions'][saved['frame_axes'][k]].T for k in saved['case_frame_index']]
    descriptions=list(old['extra_frozen_frames'])
    if fresh_blind is not None:
        folder=Path(fresh_blind);p=json.loads((folder/'protocol.json').read_text())
        digest=hashlib.sha256((folder/'blind_rotations.npz').read_bytes()).hexdigest()
        if p['status']!='COMPLETE' or digest!=p['frames_sha256']:raise ValueError('Frozen fresh frame changed')
        with np.load(folder/'blind_rotations.npz',allow_pickle=False) as saved:maps.append(saved['optimized'].copy())
        descriptions.append(dict(source='BU2 fresh blind',frame_sha256=digest))
    directions,membership=conditional.direction_classes(maps)
    keys=list(dict.fromkeys(tuple(sorted(axes)) for axes in membership))
    frame_index=np.array([keys.index(tuple(sorted(axes))) for axes in membership])
    output.mkdir(parents=True)
    np.savez_compressed(output/'W_only_geometry.npz',directions=directions,membership=membership,
        frame_axes=np.array(keys),case_frame_index=frame_index)
    protocol=dict(status='RUNNING',model=data.index['model'],all_subjects=data.n,all_original_rows=data.m,seed=494,
        channel='p_nn',primary_metrics=['haar_node_UX_columns_mean_soft_count_eps003','haar_axis_UX_columns_mean_soft_count_eps003'],
        source_matrix_manifest_sha256=bank.protocol['manifest_sha256'],source_scores_sha256=old['scores_sha256'],
        source_basis_sha256=hashlib.sha256((original/'shared_response_basis.npz').read_bytes()).hexdigest(),
        source_response_compression=json.loads((original/'shared_target_compression.json').read_text()),
        source='retained observed response functions, no conditional means or GP predictions used',
        approximation='full multi-output CART trained on every subject, with all normalized X/channel targets',
        coefficients='sqrt(n_left*n_right/n_parent) * (mean_left - mean_right)',
        unassigned_component='root mean plus leaf residual; included as last mode and last axis group',
        interpretation='finite empirical detail/UX-axis support; neither minimum tree program length nor independently validated smoothness',
        prediction_MSE_veto=False,independent_prediction_validation=False,extra_frozen_frames=descriptions)
    wave.json_file(output/'protocol.json',protocol);frame_scores={};audits=[]
    z=data.u-data.u.mean(0);z/=np.sqrt(np.var(data.u,axis=0).mean())
    for f,axes in enumerate(keys):
        q=directions[list(axes)].T;u=z@q
        estimator=DecisionTreeRegressor(random_state=494).fit(u,target)
        node,axis,audit=haar.energies(estimator,u,target,projections,data.d)
        tree=estimator.tree_
        np.savez_compressed(output/f'frame{f:03d}_partition_and_energies.npz',node_energy=node,axis_energy=axis,
            feature=tree.feature,threshold=tree.threshold,children_left=tree.children_left,children_right=tree.children_right,
            subject_count=tree.n_node_samples)
        scores={'haar_node_'+k:v for k,v in conditional.sparsity(node).items()}
        scores.update({'haar_axis_'+k:v for k,v in conditional.sparsity(axis).items()})
        frame_scores[f]=[dict(channel='p_nn',metric=m,loss=v,is_control=not scope.primary(m)) for m,v in scores.items()]
        audits.append(dict(frame=f,leaves=int(estimator.get_n_leaves()),depth=int(estimator.get_depth()),X_columns=audit))
        wave.json_file(output/'energy_and_fit_audits.json',audits)
        print(f'[TreeHaar] {data.index["model"]}: frame {f+1}/{len(keys)}, {estimator.get_n_leaves()} leaves',flush=True)
    blind=[]
    for k,f in enumerate(frame_index):
        blind.extend(dict(anonymous_case=f'case_{k:03d}' if k<len(data.paths) else f'audit_frame_{k-len(data.paths):03d}',**r)
            for r in frame_scores[int(f)])
    wave.csv_file(output/'blind_scores.csv',blind);digest=hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    reveal=json.loads((original/'reveal.json').read_text())['mapping']
    wave.json_file(output/'reveal.json',dict(mapping=reveal,scores_sha256=digest))
    grouped=defaultdict(dict);controls={};extra=[]
    for r in blind:
        key=r['channel'],r['metric'];controls[key]=r['is_control']
        if r['anonymous_case'] in reveal:grouped[key][reveal[r['anonymous_case']]]=r['loss']
        else:extra.append(r)
    summary=[dict(channel=c,metric=m,is_control=controls[c,m],baseline_loss=losses['baseline'],
        **wave.st.evaluate_curve(losses,data.index['cases'])) for (c,m),losses in grouped.items()]
    wave.csv_file(output/'metric_summary.csv',summary)
    if extra:wave.csv_file(output/'frozen_frame_assessment.csv',[dict(**r,relative_change_percent=
        100*(r['loss']-grouped[r['channel'],r['metric']]['baseline'])/max(abs(grouped[r['channel'],r['metric']]['baseline']),1e-30)) for r in extra])
    wave.json_file(output/'read_audit.json',dict(files=sorted(set(data.read_files)),forbidden_files_read=[]))
    protocol.update(status='COMPLETE',scores_sha256=digest);wave.json_file(output/'protocol.json',protocol)
    print(f'[TreeHaar] {data.index["model"]}: {sum(r["all_planes_pass"] and not r["is_control"] for r in summary)} primary grid combinations',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=wave.io.DIMS,default='CR3')
    p.add_argument('--batch',action='store_true');args=p.parse_args()
    with threadpool_limits(limits=4):
        common=None
        for model in (('CR3','C1','C2','C3') if args.batch else (args.model,)):
            source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
            output=parent/'waveBY_tree_Haar_profiles_seed494'
            if not output.exists():analyze(source,parent/'waveBU_conditional_Shapley_profiles_seed471',output,
                fresh_blind=parent/'waveBU2_fresh_Shapley_blind_seed478')
            protocol=json.loads((output/'protocol.json').read_text())
            if protocol['status']!='COMPLETE' or hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()!=protocol['scores_sha256']:
                raise ValueError('Existing Haar run incomplete or changed')
            with (output/'metric_summary.csv').open() as stream:rows=list(csv.DictReader(stream))
            passed={(r['channel'],r['metric']) for r in rows if r['all_planes_pass']=='True' and r['is_control']!='True'}
            common=passed if common is None else common&passed
            print(f'[WaveBY] {model}: {len(passed)} strict grid combinations, {len(common)} common',flush=True)
