"""Evaluation-only optima positions and continuous baseline gaps.

An interior minimum, including 15 degrees or its 75-degree permutation
equivalent, is recorded without discarding the criterion. Degrees and relative
score percentages are stored in separate fields to avoid unit ambiguity.
"""
import csv
import hashlib
import json
from collections import defaultdict

import run as wave


WAVES={
    'AU':'waveAU_conditional_UX_tree_complexity_seed421',
    'AY_poly5':'waveAY_approximate_UX_tree_seed421_poly5',
    'AY_rbf2':'waveAY_approximate_UX_tree_seed421_rbf2',
    'BA2':'waveBA2_rank_GP_posterior_UX_seed416',
    'BB':'waveBB_radial_GP_fixed_function_seed416',
    'BB2':'waveBB2_rank_radial_fixed_function_seed416',
    'BC2':'waveBC2_conditional_kernel_centers32_seed430',
    'BF':'waveBF_learned_global_link_kernel_seed430',
    'BG':'waveBG_double_residual_kernel_seed444',
    'BH':'waveBH_group_sparse_GP_seed448',
    'BJ':'waveBJ_hierarchical_group_GP_seed448',
    'BK':'waveBK_complete_response_kernel_seed430',
    'BL':'waveBL_complete_response_kernel_seed430',
    'BM':'waveBM_subset_kernel_group_GP_seed448',
    'BR':'waveBR_polynomial_link_kernel_seed430',
    'BS':'waveBS_operator_mode_support_seed430',
    'BT':'waveBT_operator_mode_support_seed430',
    'BU':'waveBU_conditional_Shapley_profiles_seed471',
    'BU3':'waveBU3_conditional_Shapley_profiles_seed471',
    'BV':'waveBV_conditional_Shapley_profiles_seed471',
}


def positions(losses,cases):
    metadata={case['id']:case for case in cases}
    baseline=losses['baseline'];scale=max(abs(baseline),1e-30)
    interior=[key for key in losses if key!='baseline' and 0<int(metadata[key].get('angle_deg',0))<90]
    if not interior:raise ValueError('No interior rotation observations')
    full=min(losses,key=losses.get);raw_minimum=losses[full];raw_minimum_case=full
    # A literal endpoint tie represents the same axes up to permutation.
    # Prefer the baseline label while retaining every tied case below.
    if baseline-raw_minimum<=1e-10*max(1.,abs(raw_minimum)):full='baseline'
    best=min(interior,key=losses.get)
    near=[key for key in interior if int(metadata[key]['angle_deg']) in (15,75)]
    def angle(key):return int(metadata[key].get('angle_deg',0))
    result=dict(baseline_loss=baseline,minimum_grid_loss=raw_minimum,minimum_grid_case=full,
        raw_minimum_grid_case=raw_minimum_case,
        raw_minimum_grid_relative_change_percent=100*(raw_minimum-baseline)/scale,
        minimum_position_numerical_tie_tolerance=1e-10,
        minimum_grid_angle_deg=angle(full),minimum_grid_pair=json.dumps(metadata[full].get('pair_1based',[])),
        minimum_grid_relative_change_percent=0. if full=='baseline' else 100*(raw_minimum-baseline)/scale,
        best_interior_case=best,best_interior_angle_deg=angle(best),
        best_interior_axis_deviation_deg=min(angle(best),90-angle(best)),
        best_interior_pair=json.dumps(metadata[best].get('pair_1based',[])),best_interior_loss=losses[best],
        best_interior_relative_change_percent=100*(losses[best]-baseline)/scale,
        complete_grid_span_relative_percent=100*(max(losses.values())-min(losses.values()))/scale,
        near_minimum_cases_json=json.dumps([dict(case=key,angle_deg=angle(key),pair=metadata[key].get('pair_1based',[]))
            for key in losses if losses[key]<=raw_minimum+1e-10*max(1,abs(raw_minimum))]),
        **wave.st.evaluate_curve(losses,cases))
    if near:
        key=min(near,key=losses.get)
        result.update(best_15_or_75_case=key,best_15_or_75_loss=losses[key],
                      best_15_or_75_relative_change_percent=100*(losses[key]-baseline)/scale)
    return result


def main():
    output=wave.ROOT/'outputs/supervised_abs_rotation/rotation_optima_including_nearby'
    output.mkdir(parents=True,exist_ok=True);rows=[];sources=[]
    for tag,name in WAVES.items():
        for model in ('CR3','C1','C2','C3'):
            source=wave.io.default_step2(wave.ROOT,model);folder=source.parent/'step5_matrix_metrics'/name
            if not (folder/'reveal.json').exists():continue
            protocol=json.loads((folder/'protocol.json').read_text())
            if protocol['status']!='COMPLETE':continue
            digest=hashlib.sha256((folder/'blind_scores.csv').read_bytes()).hexdigest()
            if digest!=protocol['scores_sha256']:raise ValueError('Frozen source scores changed')
            reveal=json.loads((folder/'reveal.json').read_text())['mapping']
            cases=json.loads((source/'index.json').read_text())['cases'];grouped=defaultdict(dict)
            with (folder/'blind_scores.csv').open() as stream:
                for r in csv.DictReader(stream):
                    if r['anonymous_case'] not in reveal:continue
                    grouped[r.get('estimator',''),r['channel'],r['metric']][reveal[r['anonymous_case']]]=float(r['loss'])
            with (folder/'metric_summary.csv').open() as stream:
                controls={(r.get('estimator',''),r['channel'],r['metric']):r.get('is_control','False') for r in csv.DictReader(stream)}
            for (estimator,channel,metric),losses in grouped.items():
                rows.append(dict(wave=tag,model=model,estimator=estimator,channel=channel,metric=metric,
                    is_control=controls[estimator,channel,metric],**positions(losses,cases)))
            sources.append(dict(wave=tag,model=model,source_scores_sha256=digest,source_run=str(folder.resolve())))
    wave.csv_file(output/'four_model_optima.csv',rows)
    overview=defaultdict(list)
    for row in rows:overview[row['wave'],row['estimator'],row['channel'],row['metric']].append(row)
    summary=[]
    for (tag,estimator,channel,metric),items in overview.items():
        summary.append(dict(wave=tag,estimator=estimator,channel=channel,metric=metric,models_completed=len(items),
            maximum_grid_improvement_over_baseline_percent=max(-r['minimum_grid_relative_change_percent'] for r in items),
            optimum_positions_by_model_json=json.dumps({r['model']:dict(angle_deg=r['minimum_grid_angle_deg'],
                pair=r['minimum_grid_pair'],relative_change_percent=r['minimum_grid_relative_change_percent']) for r in items}),
            best_interior_positions_by_model_json=json.dumps({r['model']:dict(angle_deg=r['best_interior_angle_deg'],
                axis_deviation_deg=r['best_interior_axis_deviation_deg'],pair=r['best_interior_pair'],
                relative_change_percent=r['best_interior_relative_change_percent']) for r in items})))
    wave.csv_file(output/'four_model_overview.csv',summary)
    wave.json_file(output/'protocol.json',dict(status='COMPLETE',evaluation_only=True,sources=sources,
        no_optimum_position_veto=True,no_prediction_error_veto=True,
        units='rotation positions in degrees; score differences in relative percent',
        report_interpretation='retain approximate minima and nearby cases; exact ranks are diagnostics'))
    print(f'[Rotation positions] {len(rows)} model/metric rows, {len(summary)} cross-model summaries')


if __name__=='__main__':main()
