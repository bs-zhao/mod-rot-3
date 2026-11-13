"""Evaluation-only full-plane plots of predictive variance UX criteria."""
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import run as wave
from record_rotation_optima import WAVES,positions
from record_blind_axis_positions import axis_angles


CRITERIA={
    'CF': [('p_nn','PRESS_variance_UX_columns_mean_soft_count_eps003','p_nn: mean-X predictive allocation support'),
           ('logit_nn','PRESS_variance_UX_columns_worst_soft_count_eps003','logit: worst-X predictive allocation support')],
    'CG': [('p_nn','nested_polynomial_variance_profile_UX_operator_columns_mean_soft_count_eps003','p_nn: joint U-by-mode variance support'),
           ('p_nn','nested_polynomial_variance_energy_UX_columns_mean_soft_count_eps003','p_nn: U variance allocation support')],
}


def blind_evidence(tag,model,original):
    folder=original.parent/f'wave{tag}2_fresh_Shapley_blind_seed478'
    if not (folder/'protocol.json').exists():return None
    protocol=json.loads((folder/'protocol.json').read_text())
    if protocol['status']!='COMPLETE':return None
    for name,key in (('blind_rotations.npz','frames_sha256'),('random_scores.csv','random_scores_sha256')):
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=protocol[key]:raise ValueError('Frozen blind result changed')
    assessment=json.loads((folder/'assessment.json').read_text())
    reveal=json.loads((original/'reveal.json').read_text())['mapping']
    k=next(int(key.split('_')[1]) for key,value in reveal.items() if value=='baseline')
    with np.load(original/'W_only_geometry.npz',allow_pickle=False) as saved:
        reference=saved['directions'][saved['frame_axes'][saved['case_frame_index'][k]]].T
    with (folder/'random_scores.csv').open() as stream:proposals=list(csv.DictReader(stream))
    selected=sorted(proposals,key=lambda r:float(r['loss']))[:protocol.get('optimizer_starts',2)]
    with np.load(folder/'blind_rotations.npz',allow_pickle=False) as saved:frames=saved['optimizer_start_frames'].copy()
    with (folder/'optimization_history.csv').open() as stream:history=list(csv.DictReader(stream))
    starts=[]
    for start,(proposal,frame) in enumerate(zip(selected,frames)):
        loss=min([float(proposal['loss'])]+[float(row['loss']) for row in history if int(row['start'])==start])
        starts.append(dict(start=start,proposal=int(proposal['proposal']),
            origin='anonymous observed W' if int(proposal['proposal'])==0 else 'random SO',
            score_change_percent=100*(loss-assessment['baseline_loss'])/assessment['baseline_loss'],**axis_angles(reference,frame)))
    return dict(wave=tag+'2',model=model,metric=protocol['metric'],channel=protocol['channel'],**assessment,
        anonymous_W_initial_maximum_axis_deviation_deg=axis_angles(reference,np.eye(reference.shape[0]))['matched_maximum_angle_deg'],
        selected_proposal_indices=[int(p['proposal']) for p in selected],optimizer_start_assessments=starts,
        random_only_proposal_minimum_score=min(float(p['loss']) for p in proposals[1:]))


def main():
    output=wave.ROOT/'outputs/supervised_abs_rotation/predictive_variance_evidence';output.mkdir(parents=True,exist_ok=True)
    all_rows=[];blind=[];stress=[];quality=[];policies=[]
    for tag,criteria in CRITERIA.items():
        fig,axes=plt.subplots(2,4,figsize=(15,6.6),squeeze=False)
        completed=[];common=None
        for col,model in enumerate(('CR3','C1','C2','C3')):
            source=wave.io.default_step2(wave.ROOT,model);folder=source.parent/'step5_matrix_metrics'/WAVES[tag]
            if not (folder/'protocol.json').exists():
                for ax in axes[:,col]:ax.text(.5,.5,'Not run',ha='center',transform=ax.transAxes)
                continue
            protocol=json.loads((folder/'protocol.json').read_text())
            if protocol['status']!='COMPLETE':
                for ax in axes[:,col]:ax.text(.5,.5,'Incomplete',ha='center',transform=ax.transAxes)
                continue
            digest=hashlib.sha256((folder/'blind_scores.csv').read_bytes()).hexdigest()
            if digest!=protocol['scores_sha256']:raise ValueError('Frozen predictive-variance scores changed')
            reveal=json.loads((folder/'reveal.json').read_text())['mapping']
            cases=json.loads((source/'index.json').read_text())['cases'];metadata={case['id']:case for case in cases}
            with (folder/'blind_scores.csv').open() as stream:scores=list(csv.DictReader(stream))
            completed.append(model)
            result=blind_evidence(tag,model,folder)
            if result is not None:blind.append(result)
            with (folder/'metric_summary.csv').open() as stream:
                passed={(r['channel'],r['metric']) for r in csv.DictReader(stream) if r['all_planes_pass']=='True' and r['is_control']!='True'}
            common=passed if common is None else common&passed
            if tag=='CF':
                audits=json.loads((folder/'predictive_game_audits.json').read_text())
                quality.append(dict(wave=tag,model=model,
                    maximum_monotone_envelope_increase=max(r['maximum_monotone_envelope_increase'] for r in audits),
                    maximum_conservation_error=max(r['maximum_conservation_error'] for r in audits)))
            else:
                quality.append(dict(wave=tag,model=model,degree=protocol['degree'],
                    response_quality=json.loads((folder/'full_model_response_audits.json').read_text()),
                    numerical=json.loads((folder/'numerical_audits.json').read_text())))
            for row,(channel,metric,label) in enumerate(criteria):
                ax=axes[row,col]
                losses={reveal[r['anonymous_case']]:float(r['loss']) for r in scores if r['channel']==channel and r['metric']==metric and r['anonymous_case'] in reveal}
                baseline=losses['baseline'];curves={}
                for key,value in losses.items():
                    if key=='baseline':continue
                    case=metadata[key];pair=tuple(case['pair_1based'])
                    curves.setdefault(pair,[(0,0.)]).append((case['angle_deg'],100*(value-baseline)/max(abs(baseline),1e-30)))
                for pair,curve in curves.items():
                    curve=sorted(curve);ax.plot([p[0] for p in curve],[p[1] for p in curve],marker='.',lw=1,alpha=.75)
                assessment=positions(losses,cases);all_rows.append(dict(wave=tag,model=model,channel=channel,metric=metric,**assessment))
                for r in scores:
                    if r['channel']==channel and r['metric']==metric and r['anonymous_case'] not in reveal:
                        index=int(r['anonymous_case'].split('_')[-1])
                        stress.append(dict(wave=tag,model=model,channel=channel,metric=metric,
                            source=protocol['extra_frozen_frames'][index]['source'],
                            score_change_percent=100*(float(r['loss'])-baseline)/max(abs(baseline),1e-30)))
                if result is not None and result['channel']==channel and result['metric']==metric:
                    ax.axhline(result['relative_change_percent'],ls=':',color='black',lw=1)
                ax.axhline(0,color='black',lw=.7,ls='--');ax.set_xticks([0,15,30,45,60,75,90]);ax.grid(alpha=.15)
                ax.set_title(f'{model} | best {assessment["minimum_grid_angle_deg"]} deg | {assessment["minimum_grid_relative_change_percent"]:+.3f}%',fontsize=9)
                if row==1:ax.set_xlabel('Plane rotation (degrees)')
                if col==0:ax.set_ylabel(label+'\nScore change from baseline (%)',fontsize=9)
        fig.suptitle(f'{tag}: full plane curves; all 5000 subjects; smaller score preferred',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.95));fig.savefig(output/f'{tag}_all_plane_curves.png',dpi=150);plt.close(fig)
        wave.json_file(output/f'{tag}_status.json',dict(completed_models=completed,plot_is_evaluation_only=True))
        policies.append(dict(wave=tag,completed_models=completed,
            common_strict_grid_metrics=[dict(channel=c,metric=m) for c,m in sorted(common or ())],
            no_position_or_MSE_veto=True))
    wave.csv_file(output/'grid_positions.csv',all_rows)
    wave.json_file(output/'blind_positions.json',blind);wave.json_file(output/'quality_and_numerical_audits.json',quality)
    wave.json_file(output/'four_model_policies.json',policies);wave.csv_file(output/'transferred_frames.csv',stress)
    lines=['# Predictive variance: full continuous evidence','',
        'Every main fit/score uses all 5000 subjects. Degrees and relative score percentages have separate meanings.',
        'No exact-position or MSE threshold automatically discards a candidate. Strict plane counts are diagnostics.','',
        '| Wave | Model | Channel / criterion | Coarse axis deviation | Score change | Planes |',
        '|---|---|---|---:|---:|---:|']
    for r in all_rows:
        label='worst-X allocation' if r['wave']=='CF' and r['channel']=='logit_nn' else ('mode support' if '_profile_' in r['metric'] else 'mean-X allocation')
        lines.append(f'| {r["wave"]} | {r["model"]} | {r["channel"]} / {label} | {r["minimum_grid_axis_deviation_deg"]} deg | {r["minimum_grid_relative_change_percent"]:+.6g}% | {r["success_planes"]}/{r["n_planes"]} |')
    lines+=['','| Search | Model | Score change | Mean/max matched axis deviation | Selected proposals |',
        '|---|---|---:|---:|---|']
    for r in blind:lines.append(f'| {r["wave"]} | {r["model"]} | {r["relative_change_percent"]:+.6g}% | {r["matched_mean_angle_deg"]:.3f}/{r["matched_maximum_angle_deg"]:.3f} deg | {r["selected_proposal_indices"]} |')
    lines+=['','Proposal 0 is the anonymous observed-W frame. Other proposals are random SO matrices.',
        'The observed-W frame already matches reference axes in C3; random-only starts are reported separately in blind_positions.json.',
        'Blind scores and frames are frozen before label assessment. Finite searches do not prove global optimality.',
        'CF uses fixed-fit PRESS with nonnested hyperparameter selection; CG allocates regularized explained covariance, not oracle conditional variance.',
        'Unassigned risk/covariance is retained. See transferred_frames.csv and quality_and_numerical_audits.json for complementary checks.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'[Predictive variance report] {len(all_rows)} model/criterion rows: {output}',flush=True)


if __name__=='__main__':main()
