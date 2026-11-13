"""BZ: UX sparsity of projections onto W-only conditional innovation spans."""
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
import conditional_ux as conditional
import conditional_projection_profiles as projection
import operator_mode_support as mode_support
import ux_metric_scope as scope


def analyze(source,original,output,bank_folder=None,fresh_blind=None,centers=32,context_centers=64,context_degree=3,
            device='cuda',press=False):
    original,output=Path(original),Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Retain previous projection experiment: {output}')
    old=json.loads((original/'protocol.json').read_text())
    if old['status']!='COMPLETE' or hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest()!=old['scores_sha256']:
        raise ValueError('Frozen observed-profile source changed')
    bank=profiles.MatrixBank(source,bank_folder or original.parent/'matrix_response_table_seed383');data=bank.data
    if bank.protocol['manifest_sha256']!=old['source_matrix_manifest_sha256']:raise ValueError('Observed matrix source changed')
    with np.load(original/'shared_response_basis.npz',allow_pickle=False) as saved:
        target=torch.tensor(saved['latent'],dtype=torch.float64,device=device)
        right=saved['projection'];slices=saved['task_slices'];tasks=saved['tasks']
    c=list(data.channels).index('p_nn');width=max(int(b-a) for a,b in slices)
    projections=[(int(j),torch.tensor(right[:,a:b],dtype=torch.float64,device=device))
        for (cc,j),(a,b) in zip(tasks,slices) if cc==c]
    with np.load(original/'W_only_geometry.npz',allow_pickle=False) as saved:
        maps=[saved['directions'][saved['frame_axes'][k]].T for k in saved['case_frame_index']]
    descriptions=list(old['extra_frozen_frames'])
    if fresh_blind is not None:
        folder=Path(fresh_blind);p=json.loads((folder/'protocol.json').read_text())
        digest=hashlib.sha256((folder/'blind_rotations.npz').read_bytes()).hexdigest()
        if p['status']!='COMPLETE' or digest!=p['frames_sha256']:raise ValueError('Frozen fresh frame changed')
        with np.load(folder/'blind_rotations.npz',allow_pickle=False) as saved:maps.append(saved['optimized'].copy())
        descriptions.append(dict(source='BU2 fresh blind',frame_sha256=digest))
    directions,membership=conditional.direction_classes(maps);output.mkdir(parents=True)
    np.savez_compressed(output/'W_only_geometry.npz',directions=directions,membership=membership)
    primary_view='conditional_PRESS' if press else 'conditional_span'
    protocol=dict(status='RUNNING',model=data.index['model'],all_subjects=data.n,all_original_rows=data.m,PRESS=press,
        source_matrix_manifest_sha256=bank.protocol['manifest_sha256'],source_scores_sha256=old['scores_sha256'],
        source_basis_sha256=hashlib.sha256((original/'shared_response_basis.npz').read_bytes()).hexdigest(),
        channel='p_nn',primary_metrics=[primary_view+'_projection_UX_operator_columns_mean_soft_count_eps003',
                                     primary_view+'_energy_UX_columns_mean_soft_count_eps003'],
        marginal_and_context_kernel_centers=centers,lower_context_degree=context_degree,lower_context_radial_centers=context_centers,
        numerical_singular_tolerance=1e-8,feature_construction_uses_Y=False,
        lower_span='all-subject generic other-U polynomial/radial span; affine included',
        estimator='orthogonal projection of observed response functions onto the feature innovation span',
        output_mode_whitening=False,full_response_energy_gauge=True,
        interpretation='finite-space conditional explained-profile support; not a population CI test or calibrated partial R-squared',
        prediction_or_position_veto=False,extra_frozen_frames=descriptions,
        PRESS_interpretation='positive part of fixed-W-space leave-one-out error improvement; all signed negative spectra audited' if press else None,
        PRESS_validation_scope='conditional on the all-subject observed response basis and W-only features; not independent original-Y validation' if press else None)
    wave.json_file(output/'protocol.json',protocol)
    z=data.u-data.u.mean(0);z/=np.sqrt(np.var(data.u,axis=0).mean())
    spectra={view:np.zeros((len(directions),data.dx,width)) for view in ('conditional_span','unresidualized_span')}
    if press:spectra['conditional_PRESS']=np.zeros((len(directions),data.dx,width))
    audits=[]
    for i,direction in enumerate(directions):
        features,feature_audit=ui.features(z,direction,seed=430,centers=centers)
        lower,lower_audit=conditional.complement_design(z,direction,degree=context_degree,centers=context_centers,
            lengthscale=feature_audit['context_kernel']['lengthscale'],seed=432)
        feature=torch.tensor(features['Ui_innovation_with_context'],dtype=torch.float64,device=device)
        context=torch.tensor(lower,dtype=torch.float64,device=device)
        innovation,linear_audit=projection.innovation_basis(feature,context)
        plain=projection.basis(feature);views=dict(conditional_span=innovation,unresidualized_span=plain)
        numerics={}
        for view,basis in views.items():
            values,checks=projection.mode_energies(basis,target,projections)
            for j,eigen in values:spectra[view][i,j,:len(eigen)]=eigen
            numerics[view]=dict(rank=basis.shape[1],X_columns=checks)
        if press:
            values,checks=projection.press_mode_gains(projection.basis(context),innovation,target,projections)
            for j,eigen in values:spectra['conditional_PRESS'][i,j,:len(eigen)]=eigen
            numerics['conditional_PRESS']=checks
        audits.append(dict(direction=i,W_only_feature_audit=feature_audit,lower_context_design=lower_audit,
            linear_algebra=linear_audit,projections=numerics))
        wave.json_file(output/'feature_and_projection_audits.json',audits)
        print(f'[ConditionalProjection] {data.index["model"]}: direction {i+1}/{len(directions)}, '
            f'rank {innovation.shape[1]}, context rank {linear_audit["context_rank"]}',flush=True)
    np.savez_compressed(output/'projection_squared_singular_values.npz',**spectra)
    blind=[]
    for view,values in spectra.items():
        for k,axes in enumerate(membership):
            scores={view+'_projection_'+m:v for m,v in mode_support.scores(values[axes]).items()}
            scores.update({view+'_energy_'+m:v for m,v in conditional.sparsity(values[axes].sum(-1)).items()})
            for metric,value in scores.items():
                blind.append(dict(anonymous_case=f'case_{k:03d}' if k<len(data.paths) else f'audit_frame_{k-len(data.paths):03d}',
                    channel='p_nn',metric=metric,loss=value,is_control=view!=primary_view or not scope.primary(metric)))
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
    print(f'[ConditionalProjection] {data.index["model"]}: {sum(r["all_planes_pass"] and not r["is_control"] for r in summary)} primary grid combinations',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=wave.io.DIMS,default='CR3')
    p.add_argument('--batch',action='store_true');p.add_argument('--press',action='store_true');args=p.parse_args()
    with threadpool_limits(limits=4):
        torch.set_num_threads(4);common=None
        for model in (('CR3','C1','C2','C3') if args.batch else (args.model,)):
            source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
            tag='CA' if args.press else 'BZ';output=parent/f'wave{tag}_conditional_projection_profiles_seed430'
            if not output.exists():analyze(source,parent/'waveBU_conditional_Shapley_profiles_seed471',output,
                fresh_blind=parent/'waveBU2_fresh_Shapley_blind_seed478',press=args.press)
            protocol=json.loads((output/'protocol.json').read_text())
            if protocol['status']!='COMPLETE' or hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()!=protocol['scores_sha256']:
                raise ValueError('Existing conditional projection run incomplete or changed')
            with (output/'metric_summary.csv').open() as stream:rows=list(csv.DictReader(stream))
            passed={(r['channel'],r['metric']) for r in rows if r['all_planes_pass']=='True' and r['is_control']!='True'}
            common=passed if common is None else common&passed
            print(f'[Wave{tag}] {model}: {len(passed)} strict grid combinations, {len(common)} common',flush=True)
