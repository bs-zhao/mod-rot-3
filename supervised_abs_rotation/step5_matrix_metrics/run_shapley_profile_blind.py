"""Fresh CR3 blind stress test of the common near-baseline Shapley profile score."""
import argparse
import csv
import hashlib
import inspect
import json

import numpy as np
from scipy.optimize import minimize
import torch
from threadpoolctl import threadpool_limits

import run as wave
import tree_rotation_stability as rotations
from record_blind_axis_positions import axis_angles
from shapley_profile_blind import Scorer


def main(model='CR3',source_tag='BU',run_tag='BU2',scorer_factory=Scorer,source_folder=None,
         metric='shapley_profile_UX_operator_columns_mean_soft_count_eps003',channel='p_nn',
         random_proposals=32,optimizer_starts=2,maxfev=80):
    source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
    original=source_folder or parent/f'wave{source_tag}_conditional_Shapley_profiles_seed471';output=parent/f'wave{run_tag}_fresh_Shapley_blind_seed478'
    if (output/'protocol.json').exists():
        previous=json.loads((output/'protocol.json').read_text())
        if previous['status']=='COMPLETE':
            freeze=json.loads((output/'blind_freeze.json').read_text())
            for name,key in (('random_scores.csv','random_scores_sha256'),('blind_rotations.npz','frames_sha256')):
                if hashlib.sha256((output/name).read_bytes()).hexdigest()!=freeze[key]:raise ValueError('Frozen blind run changed')
            if previous.get('source_tag','BU')!=source_tag:raise ValueError('Cannot reuse a different Shapley source')
            if previous['metric']!=metric or previous['channel']!=channel:raise ValueError('Cannot reuse a different blind criterion')
            requested=dict(random_proposals=random_proposals,optimizer_starts=optimizer_starts,maximum_evaluations_per_start=maxfev)
            defaults=dict(random_proposals=32,optimizer_starts=2,maximum_evaluations_per_start=80)
            if any(previous.get(key,defaults[key])!=value for key,value in requested.items()):
                raise ValueError('Cannot reuse a blind run with different search settings')
            print(f'[ShapleyBlind] {model}: reused complete frozen run',flush=True);return
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Retain earlier Shapley blind run: {output}')
    output.mkdir(parents=True)
    protocol=dict(status='RUNNING',model=model,source_tag=source_tag,seed=478,channel=channel,metric=metric,
        proposals=f'anonymous W frame and {random_proposals} random SO frames',
        optimizer=f'{optimizer_starts} best proposal Powell starts, {maxfev} evaluations each',
        random_proposals=random_proposals,optimizer_starts=optimizer_starts,maximum_evaluations_per_start=maxfev,
        exact_refinement='one Givens pass at +/- .1/.25 radians',all_subjects=5000,
        original_case_labels_used_by_search=False,source_profiles_and_coalition_policy_unchanged=True,
        frame_score_cache_rounding_decimals=8)
    wave.json_file(output/'protocol.json',protocol)
    scorer=scorer_factory(source,original,output)
    protocol.update(source_scores_sha256=hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest(),
        blind_runner_sha256=hashlib.sha256(wave.HERE.joinpath('run_shapley_profile_blind.py').read_bytes()).hexdigest(),
        scorer_code_sha256=hashlib.sha256(wave.HERE.joinpath(inspect.getfile(type(scorer))).read_bytes()).hexdigest())
    wave.json_file(output/'protocol.json',protocol)
    with np.load(original/'W_only_geometry.npz',allow_pickle=False) as saved:
        directions,frame_axes=saved['directions'],saved['frame_axes'];case_frames=saved['case_frame_index']
    qcheck=directions[frame_axes[case_frames[0]]].T
    with (original/'blind_scores.csv').open() as stream:
        expected=next(float(r['loss']) for r in csv.DictReader(stream) if r['anonymous_case']=='case_000'
            and r['channel']==channel and r['metric']==protocol['metric'])
    if not np.isclose(scorer.score(qcheck,force=True),expected,rtol=1e-8,atol=1e-10):
        raise ArithmeticError('Reloaded coalition scorer did not reproduce its anonymous source case')
    rng=np.random.default_rng(478);proposals=[np.eye(scorer.data.d)]
    proposals.extend(rotations.random_rotation(scorer.data.d,rng) for _ in range(random_proposals))
    rows=[]
    for k,q in enumerate(proposals):
        value=scorer.score(q);rows.append(dict(proposal=k,loss=value))
        wave.csv_file(output/'random_scores.csv',rows)
        print(f'[ShapleyBlind] proposal {k+1}/{len(proposals)}: {value:.9g}',flush=True)
    history=[];candidates=[]
    for start,index in enumerate(np.argsort([r['loss'] for r in rows])[:optimizer_starts]):
        q0=proposals[int(index)];best_value=rows[int(index)]['loss'];best_q=q0.copy()
        def objective(parameters):
            nonlocal best_value,best_q
            q=q0@rotations.increment(parameters,scorer.data.d);value=scorer.score(q)
            history.append(dict(start=start,parameters=np.asarray(parameters).tolist(),loss=value))
            if value<best_value:best_value,best_q=value,q.copy()
            if len(history)%10==0:
                wave.csv_file(output/'optimization_history.csv',history)
                print(f'[ShapleyBlind] start {start}, evaluations {sum(r["start"]==start for r in history)}, best {best_value:.9g}',flush=True)
            return value
        minimize(objective,np.zeros(scorer.data.d*(scorer.data.d-1)//2),method='Powell',
            bounds=[(-np.pi/3,np.pi/3)]*(scorer.data.d*(scorer.data.d-1)//2),options=dict(maxfev=maxfev,xtol=.005,ftol=0.))
        candidates.append((best_value,best_q))
    value,qbest=min(candidates,key=lambda r:r[0]);refinement=[]
    for i in range(scorer.data.d):
        for j in range(i+1,scorer.data.d):
            base=qbest.copy();trial_value,trial_q=value,qbest
            for angle in (-.25,-.1,.1,.25):
                change=np.eye(scorer.data.d);c,s=np.cos(angle),np.sin(angle)
                change[i,i]=change[j,j]=c;change[i,j]=-s;change[j,i]=s
                q=base@change;loss=scorer.score(q)
                refinement.append(dict(U_i=i+1,U_j=j+1,step_radians=angle,loss=loss))
                if loss<trial_value:trial_value,trial_q=loss,q.copy()
            value,qbest=trial_value,trial_q
            print(f'[ShapleyBlind] Givens {i}/{j}, best {value:.9g}',flush=True)
    wave.csv_file(output/'optimization_history.csv',history);wave.csv_file(output/'exact_refinement.csv',refinement)
    np.savez_compressed(output/'blind_rotations.npz',random=np.array(proposals),optimized=qbest,
                        optimized_loss=value,optimizer_start_frames=np.array([q for loss,q in candidates]))
    freeze=dict(random_scores_sha256=hashlib.sha256((output/'random_scores.csv').read_bytes()).hexdigest(),
        frames_sha256=hashlib.sha256((output/'blind_rotations.npz').read_bytes()).hexdigest())
    wave.json_file(output/'blind_freeze.json',freeze)
    # Labels are read only now, to assess the fully frozen search result.
    reveal=json.loads((original/'reveal.json').read_text())['mapping']
    k=next(int(k.split('_')[1]) for k,v in reveal.items() if v=='baseline')
    reference=directions[frame_axes[case_frames[k]]].T;base=scorer.score(reference)
    assessment=dict(baseline_loss=base,blind_minimum_loss=value,relative_change_percent=100*(value-base)/base,
        **axis_angles(reference,qbest),maximum_relative_squared_efficiency_error=scorer.maximum_efficiency_error,
        initial_coalition_models=scorer.initial_models,total_cached_coalition_models=len(scorer.models.models),
        no_case_labels_used_by_optimization=True,global_optimality_proven=False)
    wave.json_file(output/'assessment.json',assessment)
    wave.json_file(output/'new_conditional_model_audits.json',scorer.models.audits)
    if hasattr(scorer,'finalize'):protocol.update(scorer.finalize())
    wave.json_file(output/'read_audit.json',dict(files=sorted(set(scorer.data.read_files)),forbidden_files_read=[]))
    protocol.update(status='COMPLETE',**freeze);wave.json_file(output/'protocol.json',protocol)
    print('[ShapleyBlind assessment]',assessment,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',choices=wave.io.DIMS,default='CR3')
    parser.add_argument('--batch',action='store_true');parser.add_argument('--source-tag',default='BU');parser.add_argument('--run-tag',default='BU2')
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        for model in (('CR3','C1','C2','C3') if args.batch else (args.model,)):
            main(model,args.source_tag,args.run_tag)
