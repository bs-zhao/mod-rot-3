"""Read frozen Shapley runs; report continuous gaps, full curves and audits."""
import csv
import hashlib
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import run as wave
from record_rotation_optima import positions
from record_blind_axis_positions import axis_angles

METRIC='shapley_profile_UX_operator_columns_mean_soft_count_eps003'


def main():
    output=wave.ROOT/'outputs/supervised_abs_rotation/waveBU_continuous_profile_evidence'
    output.mkdir(parents=True,exist_ok=True);rows=[];stress=[];blind_rows=[];audits=[];refits=[];initialization=[];start_positions=[];local=[]
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,model in zip(axes.flat,('CR3','C1','C2','C3')):
        source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
        cases=json.loads((source/'index.json').read_text())['cases'];metadata={r['id']:r for r in cases}
        for tag,style in (('BU','-'),('BU3','--')):
            folder=parent/f'wave{tag}_conditional_Shapley_profiles_seed471'
            if not (folder/'reveal.json').exists():continue
            protocol=json.loads((folder/'protocol.json').read_text())
            if protocol['status']!='COMPLETE':continue
            digest=hashlib.sha256((folder/'blind_scores.csv').read_bytes()).hexdigest()
            if digest!=protocol['scores_sha256']:raise ValueError('Frozen source changed')
            reveal=json.loads((folder/'reveal.json').read_text())['mapping']
            with (folder/'blind_scores.csv').open() as stream:
                selected=[r for r in csv.DictReader(stream) if r['channel']=='p_nn' and r['metric']==METRIC]
            losses={reveal[r['anonymous_case']]:float(r['loss']) for r in selected if r['anonymous_case'] in reveal}
            row=dict(wave=tag,model=model,channel='p_nn',metric=METRIC,**positions(losses,cases));rows.append(row)
            plane=defaultdict(list)
            for key,value in losses.items():
                if key=='baseline':continue
                info=metadata[key];pair=tuple(info['pair_1based'])
                plane[pair].append((info['angle_deg'],100*(value-losses['baseline'])/losses['baseline']))
            for i,(pair,points) in enumerate(sorted(plane.items())):
                x,y=zip(*sorted([(0,0.)]+points))
                ax.plot(x,y,style,color=plt.cm.tab20(i/20),alpha=.75,lw=1.3,
                    label=f'{pair}, {protocol["radial_centers"]} centers')
            for r in selected:
                if r['anonymous_case'] in reveal:continue
                k=int(r['anonymous_case'].split('_')[-1]);loss=float(r['loss'])
                stress.append(dict(wave=tag,model=model,frame=r['anonymous_case'],
                    source=protocol['extra_frozen_frames'][k]['source'],loss=loss,baseline_loss=losses['baseline'],
                    relative_change_percent=100*(loss-losses['baseline'])/losses['baseline']))
            numeric=json.loads((folder/'numerical_audits.json').read_text())
            compression=json.loads((folder/'shared_target_compression.json').read_text())
            audits.append(dict(wave=tag,model=model,all_subjects=protocol['all_subjects'],
                all_original_rows=protocol['all_original_rows'],conditional_models=protocol['conditional_models'],
                maximum_SVD_trace_relative_error=max(r['maximum_SVD_trace_relative_error'] for r in numeric),
                maximum_efficiency_error=max(r['relative_squared_efficiency_error'] for r in numeric),
                maximum_task_compression_error=compression['maximum_task_relative_squared_error'],scores_sha256=digest))
        blind=parent/'waveBU2_fresh_Shapley_blind_seed478'
        if (blind/'assessment.json').exists():
            protocol=json.loads((blind/'protocol.json').read_text())
            if protocol['status']=='COMPLETE':
                digest=hashlib.sha256((blind/'blind_rotations.npz').read_bytes()).hexdigest()
                if digest!=protocol['frames_sha256']:raise ValueError('Frozen blind frames changed')
                result=json.loads((blind/'assessment.json').read_text());blind_rows.append(dict(model=model,**result))
                original=parent/'waveBU_conditional_Shapley_profiles_seed471'
                mapping=json.loads((original/'reveal.json').read_text())['mapping']
                k=next(int(key.split('_')[1]) for key,value in mapping.items() if value=='baseline')
                with np.load(original/'W_only_geometry.npz',allow_pickle=False) as saved:
                    reference=saved['directions'][saved['frame_axes'][saved['case_frame_index'][k]]].T
                with (blind/'random_scores.csv').open() as stream:proposals=list(csv.DictReader(stream))
                selected=sorted(proposals,key=lambda r:float(r['loss']))[:2]
                with (blind/'optimization_history.csv').open() as stream:history=list(csv.DictReader(stream))
                with np.load(blind/'blind_rotations.npz',allow_pickle=False) as saved:start_frames=saved['optimizer_start_frames']
                for start,(proposal,frame) in enumerate(zip(selected,start_frames)):
                    value=min([float(proposal['loss'])]+[float(r['loss']) for r in history if int(r['start'])==start])
                    start_positions.append(dict(model=model,start=start,proposal=int(proposal['proposal']),
                        origin='anonymous observed W frame' if int(proposal['proposal'])==0 else 'independent random SO',
                        pre_Givens_score=value,relative_change_percent=100*(value-result['baseline_loss'])/result['baseline_loss'],
                        **axis_angles(reference,frame)))
                initialization.append(dict(model=model,anonymous_W_initial_score=float(proposals[0]['loss']),
                    anonymous_W_initial_maximum_axis_angle_deg=axis_angles(reference,np.eye(reference.shape[0]))['matched_maximum_angle_deg'],
                    independent_random_SO_minimum_score=min(float(r['loss']) for r in proposals[1:]),
                    selected_proposal_indices=[int(r['proposal']) for r in sorted(proposals,key=lambda r:float(r['loss']))[:2]]))
                ax.axhline(result['relative_change_percent'],color='black',ls=':',lw=1.5)
                ax.text(.02,.98,f'Fresh blind: {result["relative_change_percent"]:+.3f}%\n'
                    f'mean/max axis: {result["matched_mean_angle_deg"]:.1f}/{result["matched_maximum_angle_deg"]:.1f} deg',
                    transform=ax.transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
        refit=parent/'waveBU5_paired_Shapley_refits_seed482'
        if (refit/'summary.json').exists() and json.loads((refit/'protocol.json').read_text())['status']=='COMPLETE':
            refits.extend(dict(model=model,**r) for r in json.loads((refit/'summary.json').read_text()))
        refinement=parent/'waveBU6_local_Shapley_refinement_seed489'
        if (refinement/'assessment.json').exists():
            protocol=json.loads((refinement/'protocol.json').read_text())
            if protocol['status']=='COMPLETE':
                for name,key in (('blind_refined_scores.csv','scores_sha256'),('refined_frames.npz','frames_sha256')):
                    if hashlib.sha256((refinement/name).read_bytes()).hexdigest()!=protocol[key]:raise ValueError('Frozen refinement changed')
                local.extend(dict(model=model,**r) for r in json.loads((refinement/'assessment.json').read_text()))
        ax.axhline(0,color='gray',lw=.7);ax.set_title(model);ax.set_xlabel('Pairwise rotation angle (degrees)')
        ax.set_ylabel('Score change from baseline (%)');ax.set_xticks(range(0,91,15));ax.grid(alpha=.15)
    fig.suptitle('Same p_nn Shapley UX profile support: 3% relative L2 tail\nSolid: 32 centers; dashed: 64; dotted: frozen fresh blind score')
    fig.savefig(output/'full_plane_curves.png',dpi=160);plt.close(fig)
    wave.csv_file(output/'grid_positions.csv',rows);wave.csv_file(output/'previous_frozen_frames.csv',stress)
    wave.json_file(output/'fresh_blind_positions.json',blind_rows);wave.json_file(output/'numerical_audits.json',audits)
    wave.json_file(output/'paired_refit_summary.json',refits)
    wave.json_file(output/'initialization_assessment.json',initialization)
    wave.json_file(output/'individual_start_positions.json',start_positions)
    wave.json_file(output/'local_refinement_positions.json',local)
    wave.csv_file(output/'fresh_blind_positions.csv',[{k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in blind_rows])
    lines=['# Conditional Shapley profile support: retained continuous evidence','',
        'Same p_nn channel and 3% relative-L2 soft support in every model. No position or prediction-error veto.','',
        '| Wave | Model | Grid angle | Axis deviation | Score change | Strict planes (diagnostic) |',
        '|---|---|---:|---:|---:|---:|']
    for r in rows:
        a=r['minimum_grid_angle_deg'];lines.append(f'| {r["wave"]} | {r["model"]} | {a} | {min(a,90-a)} | '
            f'{r["minimum_grid_relative_change_percent"]:+.6g}% | {r["success_planes"]}/{r["n_planes"]} |')
    lines+=['','| Model | Fresh blind score change | Mean/max matched axis angle |','|---|---:|---:|']
    for r in blind_rows:lines.append(f'| {r["model"]} | {r["relative_change_percent"]:+.6g}% | '
        f'{r["matched_mean_angle_deg"]:.3f}/{r["matched_maximum_angle_deg"]:.3f} deg |')
    lines+=['','| Model | Smaller-step start | Score change | Mean/max matched axis angle |','|---|---|---:|---:|']
    for r in local:lines.append(f'| {r["model"]} | {r["start"]} | {r["relative_change_percent"]:+.6g}% | '
        f'{r["matched_mean_angle_deg"]:.3f}/{r["matched_maximum_angle_deg"]:.3f} deg |')
    lines+=['','| Model | Centers | Paired samples favoring frozen frame | 5% / median / 95% score changes |',
        '|---|---:|---:|---:|']
    for r in refits:
        values=' / '.join(f'{v:+.6g}%' for v in r['q05_median_q95_percent'])
        lines.append(f'| {r["model"]} | {r["centers"]} | {r["blind_lower_count"]}/{r["replicates"]} | {values} |')
    lines+=['','Blind scores and frames were frozen before label assessment. The finite search does not prove global optimality.',
        'All original responses are retained in the Shapley full coalition, up to the audited numerical compression.',
        'Conditional proper-coalition means are estimated. Generalization has not been independently verified; GCV and resampling have different roles.',
        'The anonymous observed-W starting frame is an explicitly separate proposal from the 32 random SO frames. It can already be close to reference axes by chance; initialization_assessment.json records this after freezing.',
        'See previous_frozen_frames.csv for all transferred counterexamples and numerical_audits.json for efficiency, trace and compression checks.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('[Shapley continuous report]',output)


if __name__=='__main__':main()
