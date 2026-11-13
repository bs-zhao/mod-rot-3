"""Continuous exact-loss searches, separate from subsequent T evaluation."""
import argparse
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import torch
from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as profiles
import tree_rotation_stability as rotations
from write_approximate_shortlist import CANDIDATES
from gallery_exact_scorers import Scorer,SUPPORTED
from export_candidate_rotation_gallery import OUTPUT,MODELS,read_source,sha,evaluate_frozen,heatmap


def search(bank,info,output,tag,metric,channel,seed=557,random_starts=8,maxfev=48):
    """No access to T or original supervised R. Fixed budget; no global claim."""
    freeze=output/'search_freeze.json'
    if freeze.exists():
        result=json.loads(freeze.read_text())
        if result['status']!='FROZEN' or result['source_scores_sha256']!=info['scores_sha256']:
            raise ValueError('Existing search source changed')
        for name in ('R.npy','U.npy'):
            if sha(output/name)!=result[name[0]+'_sha256']:raise ValueError('Changed saved result')
        return result
    output.mkdir(parents=True,exist_ok=True)
    wave.json_file(output/'protocol.json',dict(status='RUNNING',metric=metric,channel=channel,seed=seed,
        random_starts=random_starts,maximum_Powell_evaluations_per_start=maxfev,optimizer_starts=2,
        Givens_passes=1,Givens_angles_radians=[-.1,.1],all_subjects=bank.data.n,
        initialization='minimum frozen W-grid loss, anonymous W identity and independent Haar proposals',
        source_scores_sha256=info['scores_sha256'],T_used_by_search=False,original_supervised_R_read=False,
        runner_sha256=sha(Path(__file__)),scorer_sha256=sha(Path(__file__).with_name('gallery_exact_scorers.py'))))
    scorer=Scorer(bank,info['folder'],output,tag,metric,channel)
    best_case=min(info['losses'],key=info['losses'].get)
    mapping=json.loads((info['folder']/'reveal.json').read_text())['mapping']
    k=next(int(k.split('_')[1]) for k,v in mapping.items() if v==best_case)
    with np.load(info['folder']/'W_only_geometry.npz',allow_pickle=False) as f:
        if 'frame_axes' in f:
            qgrid=f['directions'][f['frame_axes'][f['case_frame_index'][k]]].T
        else:qgrid=f['directions'][f['membership'][k]].T
    expected=info['losses'][best_case];replayed=scorer.score(qgrid)
    if not np.isclose(expected,replayed,rtol=1e-7,atol=1e-9):
        raise ArithmeticError(f'Exact score replay failed {tag} {metric}: {replayed} != {expected}')
    rng=np.random.default_rng(seed)
    proposals=[qgrid,np.eye(bank.data.d)]+[rotations.random_rotation(bank.data.d,rng) for _ in range(random_starts)]
    history=[];best_value=float('inf');best_q=None
    def evaluate(q,phase):
        nonlocal best_value,best_q
        value=scorer.score(q)
        history.append(dict(evaluation=len(history),phase=phase,loss=value))
        if value<best_value:best_value,best_q=value,q.copy()
        if len(history)%10==0:
            wave.csv_file(output/'search_history.csv',history)
            print(f'[RotationSearch] {bank.data.index["model"]}/{tag} eval {len(history)} best={best_value:.9g}',flush=True)
        return value
    proposal_losses=[evaluate(q,'proposal') for q in proposals]
    d=bank.data.d;p=d*(d-1)//2
    for start,index in enumerate(np.argsort(proposal_losses)[:2]):
        q0=proposals[int(index)]
        minimize(lambda x:evaluate(q0@rotations.increment(x,d),f'Powell_{start}'),np.zeros(p),
            method='Powell',bounds=[(-np.pi/3,np.pi/3)]*p,options=dict(maxfev=maxfev,xtol=.005,ftol=0.))
    for i in range(d):
        for j in range(i+1,d):
            center=best_q.copy()
            for angle in (-.1,.1):
                g=np.eye(d);g[i,i]=g[j,j]=np.cos(angle);g[i,j]=-np.sin(angle);g[j,i]=np.sin(angle)
                evaluate(center@g,f'Givens_{i}_{j}')
    wave.csv_file(output/'search_history.csv',history)
    np.save(output/'q_anonymous_W.npy',best_q)
    # A coordinate conversion only, performed after the loss-selected q exists.
    z=np.load(Path(bank.data.index['source_stage1'])/'Zstd.npy',allow_pickle=False)
    a,error=wave.io.registration(z,bank.data.u)
    r=(a@best_q).T;u=z@r.T
    if not np.allclose(r@r.T,np.eye(d),atol=1e-8):raise ArithmeticError('Nonorthogonal searched R')
    np.save(output/'R.npy',r);np.save(output/'U.npy',u)
    result=dict(status='FROZEN',search_scope='continuous exact metric: W grid + Haar + two Powell starts + Givens',
        global_optimality_proven=False,metric=metric,channel=channel,loss=best_value,
        baseline_loss=info['losses']['baseline'],grid_minimum_loss=expected,evaluations=len(history),
        source_scores_sha256=info['scores_sha256'],source_tag=info['source_tag'],
        all_subjects=bank.data.n,R_application='U = Zstd @ R.T; equivalently U_column = R @ Zstd_column',
        coordinate_registration_relative_error=error,T_used_by_search=False,original_saved_supervised_R_read=False,
        R_sha256=sha(output/'R.npy'),U_sha256=sha(output/'U.npy'),q_sha256=sha(output/'q_anonymous_W.npy'),
        exact_saved_grid_replay_loss=replayed,expected_saved_grid_loss=expected)
    wave.json_file(freeze,result)
    if best_value>expected+1e-7*max(abs(expected),1):raise ArithmeticError('Search lost its grid incumbent')
    # Keep the loss and searched coordinates; remove newly fitted scratch models
    # only after the frozen result is durable, to retain disk for all 44 metrics.
    import shutil
    scratch=(output/'conditional_models').resolve()
    if scratch.parent!=output.resolve():raise ValueError('Invalid scratch folder')
    if scratch.exists():shutil.rmtree(scratch)
    return result


def main(numbers=None):
    for model in MODELS:
        source=wave.io.default_step2(wave.ROOT,model)
        bank=profiles.MatrixBank(source,source.parent/'step5_matrix_metrics/matrix_response_table_seed383')
        for number,candidate in enumerate(CANDIDATES,1):
            if numbers and number not in numbers:continue
            label,tag,metric=candidate[:3];channel=candidate[3] if len(candidate)>3 else 'logit_nn'
            if tag not in SUPPORTED:continue
            folder=OUTPUT/f'{number:02d}_{tag}_{channel}'
            output=folder/model/'continuous_search';info=read_source(model,tag,metric,channel)
            if info is None:raise ValueError(f'Missing completed source: {model}/{tag}')
            result=search(bank,info,output,tag,metric,channel)
            evaluate_frozen(info,output)
            p=json.loads((output/'protocol.json').read_text());p['status']='COMPLETE';wave.json_file(output/'protocol.json',p)
            print(f'[RotationSearch COMPLETE] {model} candidate {number}/44 {tag}: {result["loss"]:.9g}',flush=True)
            gc.collect();torch.cuda.empty_cache()
        del bank;gc.collect();torch.cuda.empty_cache()
    # Refresh the deliverable after the full fixed-budget batch is frozen.
    from export_saved_search_heatmaps import main as refresh
    refresh()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--numbers',type=int,nargs='+')
    args=parser.parse_args()
    with threadpool_limits(4):
        torch.set_num_threads(4);main(args.numbers)
