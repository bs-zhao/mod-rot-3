"""CL: predeclared independent Haar benchmark of the fixed BU NN-p criterion."""
import argparse
import csv
import hashlib
import inspect
import json

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run as wave
import tree_rotation_stability as rotations
from random_rotation_percentile import summarize
from shapley_profile_blind import Scorer
from validation_response_policy import response_fields,final_Y_eligible

MODELS=('CR3','C1','C2','C3')
METRIC='shapley_profile_UX_operator_columns_mean_soft_count_eps003'


def main(models=MODELS,draws=64,seed=541):
    root=wave.ROOT/f'outputs/supervised_abs_rotation/waveCL_independent_Haar_percentile_seed{seed}'
    root.mkdir(parents=True,exist_ok=True)
    plan=dict(candidate_wave='BU',channel='p_nn',metric=METRIC,models=list(MODELS),all_subjects=5000,
        random_draws_per_model=draws,root_seed=seed,model_seed_rule='root_seed + 1009 * fixed model index',
        random_distribution='Haar SO(d), Gaussian QR with positive diagonal correction and determinant correction',
        response_policy=response_fields('p_nn'),no_analytic_response_values_used=True,
        reference_labels_revealed_after_random_scores_and_frames_frozen=True,
        confidence_family=4,simultaneous_alpha=.05,numerical_relative_tie_tolerance=1e-7,
        reference_tail_threshold_recorded_percent=15,threshold_is_not_a_unique_global_optimum_requirement=True,
        initial_observed_W_is_not_a_random_draw=True,adaptive_stopping=False,
        code_sha256=hashlib.sha256(b''.join(path.read_bytes() for path in
            (wave.HERE/'run_independent_random_percentile.py',wave.HERE/'random_rotation_percentile.py',
             wave.HERE/'shapley_profile_blind.py',wave.HERE/'shapley_function_profiles.py'))).hexdigest())
    if not final_Y_eligible(plan['channel']):raise ValueError('Auxiliary Y cannot qualify for final validation')
    file=root/'plan_frozen_before_sampling.json'
    if file.exists():
        if json.loads(file.read_text())!=plan:raise ValueError('Predeclared random benchmark changed; retain the earlier run')
    else:wave.json_file(file,plan)
    plan_hash=hashlib.sha256(file.read_bytes()).hexdigest();results=[]
    for model in models:
        source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
        original=parent/'waveBU_conditional_Shapley_profiles_seed471';output=root/model
        if (output/'protocol.json').exists() and json.loads((output/'protocol.json').read_text())['status']=='COMPLETE':
            protocol=json.loads((output/'protocol.json').read_text())
            if protocol['plan_sha256']!=plan_hash:raise ValueError('Completed model used a different plan')
            for name,key in (('random_scores.csv','scores_sha256'),('random_frames.npz','frames_sha256')):
                if hashlib.sha256((output/name).read_bytes()).hexdigest()!=protocol[key]:raise ValueError('Frozen random benchmark changed')
            results.append(json.loads((output/'assessment.json').read_text()));continue
        if output.exists() and any(output.iterdir()):raise FileExistsError(f'Retain incomplete independent benchmark: {output}')
        output.mkdir(parents=True)
        protocol=dict(status='RUNNING',model=model,plan_sha256=plan_hash,**response_fields('p_nn'))
        wave.json_file(output/'protocol.json',protocol)
        scorer=Scorer(source,original,output)
        if scorer.data.n!=5000:raise ValueError('This real-data benchmark requires all 5000 subjects')
        with np.load(original/'W_only_geometry.npz',allow_pickle=False) as saved:
            directions=saved['directions'].copy();axes=saved['frame_axes'].copy();case_index=saved['case_frame_index'].copy()
        with (original/'blind_scores.csv').open() as stream:source_rows=list(csv.DictReader(stream))
        expected=next(float(r['loss']) for r in source_rows if r['anonymous_case']=='case_000' and r['channel']=='p_nn' and r['metric']==METRIC)
        if not np.isclose(scorer.score(directions[axes[case_index[0]]].T,force=True),expected,rtol=1e-8,atol=1e-10):
            raise ArithmeticError('Independent percentile scorer failed anonymous source replay')
        model_seed=seed+1009*MODELS.index(model);rng=np.random.default_rng(model_seed)
        frames=[];rows=[]
        for draw in range(draws):
            q=rotations.random_rotation(scorer.data.d,rng);loss=scorer.score(q)
            frames.append(q);rows.append(dict(draw=draw,loss=loss))
            wave.csv_file(output/'random_scores.csv',rows)
            if (draw+1)%8==0 or draw+1==draws:
                wave.json_file(output/'new_conditional_model_audits.json',scorer.models.audits)
                print(f'[CL independent Haar] {model}: {draw+1}/{draws}, score {loss:.8g}',flush=True)
        np.savez_compressed(output/'random_frames.npz',frames=np.asarray(frames))
        freeze=dict(scores_sha256=hashlib.sha256((output/'random_scores.csv').read_bytes()).hexdigest(),
            frames_sha256=hashlib.sha256((output/'random_frames.npz').read_bytes()).hexdigest())
        wave.json_file(output/'blind_freeze.json',dict(plan_sha256=plan_hash,**freeze))
        # The reference label is accessed only after all random scores freeze.
        reveal=json.loads((original/'reveal.json').read_text())['mapping']
        baseline=next(key for key,value in reveal.items() if value=='baseline');k=int(baseline.split('_')[1])
        reference=scorer.score(directions[axes[case_index[k]]].T,force=True)
        expected=next(float(r['loss']) for r in source_rows if r['anonymous_case']==baseline and r['channel']=='p_nn' and r['metric']==METRIC)
        if not np.isclose(reference,expected,rtol=1e-8,atol=1e-10):raise ArithmeticError('Reference replay changed after freezing random scores')
        assessment=dict(model=model,channel='p_nn',metric=METRIC,model_seed=model_seed,all_subjects=scorer.data.n,
            **response_fields('p_nn'),**summarize([r['loss'] for r in rows],reference))
        wave.json_file(output/'assessment.json',assessment);results.append(assessment)
        wave.json_file(output/'read_audit.json',dict(scoring_files=sorted(set(scorer.data.read_files)),
            evaluation_labels_read_after_freeze=True,original_NN_weights_or_generators_read=False))
        protocol.update(status='COMPLETE',**freeze,source_scores_sha256=scorer.protocol['scores_sha256'])
        wave.json_file(output/'protocol.json',protocol)
        wave.json_file(root/'four_model_assessment.json',dict(completed_models=[r['model'] for r in results],results=results))
        print('[CL assessment]',model,assessment['random_at_least_as_good_as_reference'],
            'of',draws,'upper %',assessment['random_favorable_tail_upper_percent'],flush=True)
    lines=['# Independent random-rotation percentile of the fixed BU NN-p criterion','',
        '64 fresh Haar rotations per model, all 5000 subjects; reference labels assessed only after score freezing.',
        'Numerical ties count as at least as good as reference. Confidence bounds are one-sided, simultaneous 95% via four-case Bonferroni.','',
        '| Model | Random frames at least as good as reference | Empirical fraction | Simultaneous upper bound |',
        '|---|---:|---:|---:|']
    for r in results:lines.append(f'| {r["model"]} | {r["random_at_least_as_good_as_reference"]}/{r["random_frames"]} | '
        f'{r["empirical_random_favorable_tail_percent"]:.3f}% | {r["random_favorable_tail_upper_percent"]:.3f}% |')
    lines+=['','These are conditional random-orientation ranks, not a claim of a unique global minimum, population sensitivity, or exact axis recovery.',
        'Optimized lower-scoring nearby and distant alternatives remain in the separate frozen-search reports.',
        'The response source is actual NN p, not an analytic probability. The metric and sample sizes were fixed before these new draws.']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('[CL independent percentile report]',root,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',choices=MODELS,default='CR3')
    parser.add_argument('--batch',action='store_true');args=parser.parse_args()
    with threadpool_limits(4):
        torch.set_num_threads(4);main(MODELS if args.batch else (args.model,))
