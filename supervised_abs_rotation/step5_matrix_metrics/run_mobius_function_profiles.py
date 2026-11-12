"""BX: sparse U-subset--X response profiles from frozen conditional coalitions.

All-order conditional Moebius dividends retain interactions as distinct
hyperedges. They sum to the observed response even for dependent U, but
are generally nonorthogonal and are not population Sobol variance effects.
"""
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
import conditional_ux as conditional
import operator_mode_support as mode_support
import mobius_function_profiles as mobius
import ux_metric_scope as scope
from shapley_profile_blind import Scorer


def analyze(source,original,output,device='cuda',bank_folder=None,fresh_blind=None):
    original,output=Path(original),Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Retain previous dividend run: {output}')
    output.mkdir(parents=True);scorer=Scorer(source,original,output,device=device,bank_folder=bank_folder)
    models,data=scorer.models,scorer.data
    with np.load(original/'W_only_geometry.npz',allow_pickle=False) as saved:
        maps=[saved['directions'][saved['frame_axes'][index]].T for index in saved['case_frame_index']]
    descriptions=list(scorer.protocol['extra_frozen_frames'])
    if fresh_blind is not None:
        fresh_blind=Path(fresh_blind);bp=json.loads((fresh_blind/'protocol.json').read_text())
        digest=hashlib.sha256((fresh_blind/'blind_rotations.npz').read_bytes()).hexdigest()
        if bp['status']!='COMPLETE' or digest!=bp['frames_sha256']:raise ValueError('Fresh frozen blind run changed')
        with np.load(fresh_blind/'blind_rotations.npz',allow_pickle=False) as saved:maps.append(saved['optimized'].copy())
        descriptions.append(dict(source='BU2 fresh blind',frame_sha256=digest))
    directions,membership=conditional.direction_classes(maps)
    keys=list(dict.fromkeys(tuple(sorted(axes)) for axes in membership))
    frame_index=np.array([keys.index(tuple(sorted(axes))) for axes in membership])
    np.savez_compressed(output/'W_only_geometry.npz',directions=directions,membership=membership,
        frame_axes=np.array(keys),case_frame_index=frame_index)
    protocol=dict(status='RUNNING',model=data.index['model'],channel='p_nn',all_subjects=data.n,all_original_rows=data.m,
        source_matrix_manifest_sha256=scorer.protocol['source_matrix_manifest_sha256'],source_scores_sha256=scorer.protocol['scores_sha256'],
        coalition_regressions='reuse frozen BU models; fit only new frozen audit-frame subspaces',
        subset_order_policy='all nonempty U subsets, no imposed interaction-order cutoff',
        degree=scorer.protocol['degree'],radial_centers=scorer.protocol['radial_centers'],
        decomposition='all-order conditional function Moebius transform; empty zero, full observed response',
        independence_of_U_assumed=False,orthogonality_assumed=False,population_Sobol_interpretation=False,
        extra_frozen_frames=descriptions,no_position_or_error_veto=True)
    wave.json_file(output/'protocol.json',protocol)
    spectra_rows={};audits=[];d=data.d
    for f,axes in enumerate(keys):
        q=directions[list(axes)].T;before=len(models.models)
        values=torch.zeros((2**d,*models.target.shape),dtype=models.target.dtype,device=device)
        values[-1]=models.target
        for mask in range(1,2**d-1):
            values[mask]=models.predict(q[:,[i for i in range(d) if mask&(1<<i)]])
        if f<len(keys)-int(fresh_blind is not None) and len(models.models)!=before:
            raise ArithmeticError('A source case unexpectedly needed new conditional subspace fits')
        parts=mobius.dividends(values);del values
        efficiency=float((parts.sum(0)-models.target).square().sum()/models.target.square().sum())
        if efficiency>1e-18:raise ArithmeticError('Conditional dividends did not reconstruct the full observed response')
        spectra=np.zeros((2**d-1,data.dx,scorer.width));ratios=[];full_fractions=[];trace_error=0.
        for j,p in scorer.projections:
            component=parts[1:]@p;energy=component.square().sum((1,2))/data.n
            eigen=torch.linalg.svdvals(component).square()/data.n
            relative=float(torch.max(torch.abs(eigen.sum(1)-energy)/energy.clamp_min(1e-30)))
            trace_error=max(trace_error,relative)
            if relative>1e-8:raise ArithmeticError('Dividend SVD lost hyperedge profile energy')
            spectra[:,j,:eigen.shape[1]]=eigen.cpu().numpy()
            total=float((models.target@p).square().sum()/data.n)
            ratios.append(float(energy.sum())/max(total,1e-30));full_fractions.append(float(energy[-1])/max(total,1e-30))
        np.savez_compressed(output/f'frame{f:03d}_dividend_spectra.npz',squared_singular_values=spectra)
        scores={'conditional_dividend_'+k:v for k,v in mode_support.scores(spectra).items()}
        scores.update({'conditional_dividend_energy_'+k:v for k,v in conditional.sparsity(spectra.sum(-1)).items()})
        spectra_rows[f]=[dict(channel='p_nn',metric=m,loss=v,is_control=not scope.primary(m)) for m,v in scores.items()]
        audits.append(dict(frame=f,relative_squared_efficiency_error=efficiency,maximum_SVD_trace_error=trace_error,
            sum_dividend_energy_over_observed_energy_by_X=ratios,full_subset_energy_over_observed_energy_by_X=full_fractions,
            new_conditional_models=len(models.models)-before))
        wave.json_file(output/'allocation_and_trace_audits.json',audits)
        print(f'[ConditionalDividends] {data.index["model"]}: frame {f+1}/{len(keys)}, allocation error {efficiency:.3g}',flush=True)
    blind=[]
    for k,f in enumerate(frame_index):
        blind.extend(dict(anonymous_case=f'case_{k:03d}' if k<len(data.paths) else f'audit_frame_{k-len(data.paths):03d}',**r)
            for r in spectra_rows[int(f)])
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
    wave.json_file(output/'new_conditional_model_audits.json',models.audits)
    wave.json_file(output/'read_audit.json',dict(files=sorted(set(data.read_files)),forbidden_files_read=[]))
    protocol.update(status='COMPLETE',scores_sha256=digest);wave.json_file(output/'protocol.json',protocol)
    print(f'[ConditionalDividends] {data.index["model"]}: {sum(r["all_planes_pass"] and not r["is_control"] for r in summary)} primary grid combinations',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=wave.io.DIMS,default='CR3')
    p.add_argument('--batch',action='store_true');args=p.parse_args()
    with threadpool_limits(limits=4):
        torch.set_num_threads(4);common=None
        for model in (('CR3','C1','C2','C3') if args.batch else (args.model,)):
            source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
            output=parent/'waveBX_conditional_dividend_profiles_seed471'
            if not output.exists():analyze(source,parent/'waveBU_conditional_Shapley_profiles_seed471',output,
                fresh_blind=parent/'waveBU2_fresh_Shapley_blind_seed478')
            protocol=json.loads((output/'protocol.json').read_text())
            if protocol['status']!='COMPLETE' or hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()!=protocol['scores_sha256']:
                raise ValueError('Existing dividend run incomplete or changed')
            with (output/'metric_summary.csv').open() as stream:rows=list(csv.DictReader(stream))
            passed={(r['channel'],r['metric']) for r in rows if r['all_planes_pass']=='True' and r['is_control']!='True'}
            common=passed if common is None else common&passed
            print(f'[WaveBX] {model}: {len(passed)} strict grid combinations, {len(common)} common',flush=True)
