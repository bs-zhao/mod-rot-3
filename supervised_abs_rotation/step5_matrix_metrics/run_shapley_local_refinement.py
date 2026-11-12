"""BU6: smaller exact Givens steps from frozen best and random-only starts."""
import argparse
import csv
import hashlib
import json

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run as wave
from record_blind_axis_positions import axis_angles
from shapley_profile_blind import Scorer


def main(model):
    source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
    original=parent/'waveBU_conditional_Shapley_profiles_seed471';previous=parent/'waveBU2_fresh_Shapley_blind_seed478'
    output=parent/'waveBU6_local_Shapley_refinement_seed489'
    old=json.loads((previous/'protocol.json').read_text());digest=hashlib.sha256((previous/'blind_rotations.npz').read_bytes()).hexdigest()
    if old['status']!='COMPLETE' or digest!=old['frames_sha256']:raise ValueError('Source blind run changed')
    if (output/'protocol.json').exists():
        saved=json.loads((output/'protocol.json').read_text())
        if saved['status']=='COMPLETE' and saved['source_blind_frames_sha256']==digest:
            for file,key in (('blind_refined_scores.csv','scores_sha256'),('refined_frames.npz','frames_sha256')):
                if hashlib.sha256((output/file).read_bytes()).hexdigest()!=saved[key]:raise ValueError('Frozen refinement changed')
            print(f'[ShapleyLocal] {model}: reused complete frozen refinement',flush=True);return
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Retain previous local refinement: {output}')
    with (previous/'random_scores.csv').open() as stream:proposals=list(csv.DictReader(stream))
    selected=sorted(proposals,key=lambda r:float(r['loss']))[:2]
    with np.load(previous/'blind_rotations.npz',allow_pickle=False) as saved:
        frames=[('previous_best',saved['optimized'].copy())]
        for proposal,q in zip(selected,saved['optimizer_start_frames']):
            if int(proposal['proposal'])>0:frames.append(('independent_random_start',q.copy()))
    if len(frames)!=2:raise ValueError('This audit expects exactly one random-only selected proposal')
    output.mkdir(parents=True)
    protocol=dict(status='RUNNING',model=model,source_blind_frames_sha256=digest,all_subjects=5000,
        channel='p_nn',metric=old['metric'],source_function_and_regression_policy_unchanged=True,
        optimizer='one sequential Givens pass for each frozen frame, exact scores at +/- .02/.05/.1 radians',
        angle_labels_used_for_optimization=False,baseline_score_used_for_optimization=False,global_optimality_proven=False)
    wave.json_file(output/'protocol.json',protocol);scorer=Scorer(source,original,output)
    results=[];history=[];refined=[]
    for name,q in frames:
        start=scorer.score(q);value=start
        for i in range(scorer.data.d):
            for j in range(i+1,scorer.data.d):
                best_q=q.copy();best=value
                for angle in (-.1,-.05,-.02,.02,.05,.1):
                    change=np.eye(scorer.data.d);c,s=np.cos(angle),np.sin(angle)
                    change[i,i]=change[j,j]=c;change[i,j]=-s;change[j,i]=s
                    trial=q@change;loss=scorer.score(trial)
                    history.append(dict(start=name,U_i=i+1,U_j=j+1,angle_radians=angle,loss=loss))
                    if loss<best:best,best_q=loss,trial
                value,q=best,best_q
                wave.csv_file(output/'refinement_history.csv',history)
                print(f'[ShapleyLocal] {model} {name}, pair {i}/{j}: {value:.9g}',flush=True)
        if value>start+1e-12:raise ArithmeticError('Exact local refinement increased its objective')
        results.append(dict(start=name,initial_loss=start,final_loss=value));refined.append(q)
    wave.csv_file(output/'blind_refined_scores.csv',results)
    np.savez_compressed(output/'refined_frames.npz',frames=np.array(refined))
    freeze=dict(scores_sha256=hashlib.sha256((output/'blind_refined_scores.csv').read_bytes()).hexdigest(),
        frames_sha256=hashlib.sha256((output/'refined_frames.npz').read_bytes()).hexdigest())
    wave.json_file(output/'blind_freeze.json',freeze)
    # Assess only after every refined score/frame has frozen.
    reveal=json.loads((original/'reveal.json').read_text())['mapping']
    k=next(int(key.split('_')[1]) for key,value in reveal.items() if value=='baseline')
    with np.load(original/'W_only_geometry.npz',allow_pickle=False) as saved:
        reference=saved['directions'][saved['frame_axes'][saved['case_frame_index'][k]]].T
    baseline=scorer.score(reference)
    assessment=[dict(**r,baseline_loss=baseline,relative_change_percent=100*(r['final_loss']-baseline)/baseline,
        **axis_angles(reference,q)) for r,q in zip(results,refined)]
    wave.json_file(output/'assessment.json',assessment)
    wave.json_file(output/'new_conditional_model_audits.json',scorer.models.audits)
    wave.json_file(output/'read_audit.json',dict(files=sorted(set(scorer.data.read_files)),forbidden_files_read=[]))
    protocol.update(status='COMPLETE',**freeze);wave.json_file(output/'protocol.json',protocol)
    print('[ShapleyLocal assessment]',model,assessment,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=wave.io.DIMS,default='CR3')
    p.add_argument('--batch',action='store_true');args=p.parse_args()
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        for model in (('CR3','C1','C2','C3') if args.batch else (args.model,)):
            main(model)
