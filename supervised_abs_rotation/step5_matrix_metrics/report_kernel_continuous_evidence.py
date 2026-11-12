"""Describe grid gaps and contrast continuously, alongside frozen blind evidence."""
import argparse
import csv
import json
from collections import defaultdict

import numpy as np

import run as wave
from record_rotation_optima import WAVES


METRICS = (
    'Ui_innovation_with_context_raw_UX_columns_worst_soft_count_eps010',
    'Ui_innovation_with_context_raw_UX_columns_worst_fractional_count_eps001',
)


def main(tag='BC2'):
    name = WAVES[tag]
    output = wave.ROOT/f'outputs/supervised_abs_rotation/wave{tag}_continuous_evidence'
    output.mkdir(parents=True, exist_ok=True)
    root = wave.io.default_step2(wave.ROOT,'CR3').parent/'step5_matrix_metrics'
    counter_file=(root/f'wave{tag}_frozen_counter_family_v2/assessment.csv' if tag in ('BC2','BF') else
                  root/name/'frozen_counter_assessment.csv')
    with counter_file.open() as stream:
        counters = {(r['channel'],r['metric']):r for r in csv.DictReader(stream)}
    for r in counters.values():
        if 'relative_blind_margin' not in r:r['relative_blind_margin']=r['relative_margin']
    rows, curves, counts = [], {}, []
    common = None
    for model in ('CR3','C1','C2','C3'):
        source = wave.io.default_step2(wave.ROOT,model); folder = source.parent/'step5_matrix_metrics'/name
        protocol = json.loads((folder/'protocol.json').read_text())
        if protocol['status']!='COMPLETE':
            raise ValueError(f'{model} source still incomplete')
        reveal = json.loads((folder/'reveal.json').read_text())['mapping']
        grouped = defaultdict(dict)
        with (folder/'blind_scores.csv').open() as stream:
            for r in csv.DictReader(stream):
                if r['anonymous_case'] not in reveal:continue
                grouped[r['channel'],r['metric']][reveal[r['anonymous_case']]] = float(r['loss'])
        with (folder/'metric_summary.csv').open() as stream:
            summaries = {(r['channel'],r['metric']):r for r in csv.DictReader(stream)}
        passed = {key for key,s in summaries.items() if s['is_control']=='False' and s['all_planes_pass']=='True'}
        common = passed if common is None else common & passed
        counts.append(dict(model=model, primary_grid_passes=len(passed)))
        for key, losses in grouped.items():
            c,m = key; baseline = losses['baseline']; summary = summaries[key]
            span = max(losses.values())-min(losses.values())
            grid_margin = (min(losses.values())-baseline)/max(abs(baseline),1e-30)
            counter = counters[key]
            rows.append(dict(model=model, channel=c, metric=m, baseline_loss=baseline,
                minimum_grid_loss=min(losses.values()), minimum_relative_grid_margin=grid_margin,
                grid_span=span, grid_relative_span=span/max(abs(baseline),1e-30),
                negative_grid_gap_in_span_units=(baseline-min(losses.values()))/span if span>0 else 0.,
                frozen_CR3_counter_relative_margin=counter['relative_blind_margin'] if model=='CR3' else '',
                frozen_CR3_counter_margin_in_span_units=(float(counter['loss'])-baseline)/span if model=='CR3' and span>0 else '',
                **{k:summary[k] for k in ('success_planes','n_planes','baseline_rank','endpoint_error','is_control')}))
        curves[model] = (json.loads((source/'index.json').read_text())['cases'], grouped)
    wave.csv_file(output/'continuous_evidence.csv', rows)
    wave.json_file(output/'summary.json', dict(status='COMPLETE', models=counts,
        common_strict_grid_metrics=[dict(channel=c,metric=m) for c,m in sorted(common)],
        interpretation='exploratory gaps and contrasts; no hard prediction-error or gap veto',
        CR3_counterexample_is_fixed_from_other_wave=True, other_models_blind_optimization_performed=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,4,figsize=(16,7.8))
    metrics=METRICS
    if tag=='BG':metrics=tuple('poly5_rbf128_'+m for m in METRICS)
    if tag in ('BH','BJ'):
        metrics=tuple('poly3_rbf32_posterior_mean_UX_columns_worst_'+suffix
                      for suffix in ('soft_count_eps010','fractional_count_eps001'))
    if tag=='BM':metrics=tuple('poly3_rbf8_posterior_mean_UX_columns_worst_'+suffix
                               for suffix in ('soft_count_eps010','fractional_count_eps001'))
    if tag=='BK':metrics=tuple('with_U_main_'+m for m in METRICS)
    if tag=='BL':metrics=tuple('with_nonaffine_U_main_'+m for m in METRICS)
    for col, model in enumerate(('CR3','C1','C2','C3')):
        cases, grouped = curves[model]
        for row, metric in enumerate(metrics):
            ax = axes[row,col]; losses = grouped['logit_nn',metric]; base = losses['baseline']
            planes = defaultdict(list)
            for case in cases:
                if case.get('pair_1based'):
                    planes[tuple(case['pair_1based'])].append((case['angle_deg'],losses[case['id']]))
            worst = min(planes,key=lambda p:min(v for a,v in planes[p] if 0<a<90))
            for plane, points in planes.items():
                xy = sorted([(0,base)]+points); angles = [a for a,v in xy]
                delta = [(v-base)/max(abs(base),1e-30)*100 for a,v in xy]
                ax.plot(angles,delta,color='#c06a1b' if plane==worst else '#6394b6',
                        alpha=1 if plane==worst else .48,lw=2 if plane==worst else 1)
            ax.axhline(0,color='#333333',lw=.8); ax.set_xticks([0,30,60,90]); ax.grid(alpha=.18)
            item = next(r for r in rows if r['model']==model and r['channel']=='logit_nn' and r['metric']==metric)
            ax.set_title(f"{model}: {item['success_planes']}/{item['n_planes']} planes",fontsize=11)
            if col==0:
                ax.set_ylabel(('10% soft support' if row==0 else '1% fractional support')+'\nchange from baseline (%)')
            if row==1:ax.set_xlabel('pair rotation (degrees)')
    fig.suptitle(f'{tag}: same logit response-support definitions across all four datasets\n'
                 'All planes shown; orange marks the lowest-scoring plane. Blind evidence is in the CSV.',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.92)); fig.savefig(output/'grid_gaps_and_contrast.png',dpi=170); plt.close(fig)
    print(json.dumps(dict(models=counts,common_strict=len(common),plot=str(output/'grid_gaps_and_contrast.png'))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',choices=('BC2','BF','BG','BH','BJ','BK','BL','BM','BR'),default='BC2')
    main(p.parse_args().tag)
