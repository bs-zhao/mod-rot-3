"""Paired conditional-regression refits of frozen near-optimal Shapley frames.

The observed X response bank, output modes and per-task gauges remain frozen.
Within every subject sample the response functions and U are recentered. This
is a conditional-fit stability audit, not independent predictive validation.
"""
import argparse
import csv
import hashlib
import json

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as profiles
import operator_mode_support as mode_support
import shapley_function_profiles as shapley
from shapley_profile_blind import KEY


@torch.no_grad()
def paired_refit(u,latent,projection,slices,tasks,channel,frames,ids,output,centers,degree=3,device='cuda'):
    z=u[ids]-u[ids].mean(0);z/=np.sqrt(np.var(u[ids],axis=0).mean())
    response=latent[ids]-latent[ids].mean(0)
    model=shapley.CoalitionModels(z,torch.tensor(response,dtype=torch.float64,device=device),
        output/'conditional_models',centers=centers,degree=degree,device=device)
    selected=[(int(j),torch.tensor(projection[:,a:b],dtype=torch.float64,device=device))
        for (c,j),(a,b) in zip(tasks,slices) if c==channel]
    width=max(int(b-a) for a,b in slices);dx=max(j for j,p in selected)+1;rows=[]
    for name,q in frames.items():
        shares,error=model.frame_shares(q);spectra=np.zeros((u.shape[1],dx,width))
        for j,p in selected:
            eigen=torch.linalg.svdvals(shares@p).square()/len(ids)
            spectra[:,j,:eigen.shape[1]]=eigen.cpu().numpy()
        rows.append(dict(frame=name,loss=mode_support.scores(spectra)[KEY],
            relative_squared_efficiency_error=error))
    wave.json_file(output/'conditional_model_audits.json',model.audits)
    wave.json_file(output/'scores.json',rows)
    return rows


def main(model,replicates=12):
    source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
    original=parent/'waveBU_conditional_Shapley_profiles_seed471'
    blind=parent/'waveBU2_fresh_Shapley_blind_seed478';output=parent/'waveBU5_paired_Shapley_refits_seed482'
    old=json.loads((original/'protocol.json').read_text());bp=json.loads((blind/'protocol.json').read_text())
    if old['status']!='COMPLETE' or bp['status']!='COMPLETE':raise ValueError('Source and blind run must be complete')
    source_hash=hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest()
    frame_hash=hashlib.sha256((blind/'blind_rotations.npz').read_bytes()).hexdigest()
    if source_hash!=old['scores_sha256'] or frame_hash!=bp['frames_sha256']:raise ValueError('A frozen source changed')
    if (output/'protocol.json').exists():
        previous=json.loads((output/'protocol.json').read_text())
        if (previous['status']=='COMPLETE' and previous['source_scores_sha256']==source_hash
                and previous['frame_sha256']==frame_hash and previous['replicates']==replicates):
            print(f'[ShapleyRefits] {model}: reused complete paired audit',flush=True);return
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Retain paired refit audit: {output}')
    bank=profiles.MatrixBank(source,parent/'matrix_response_table_seed383');data=bank.data
    if bank.protocol['manifest_sha256']!=old['source_matrix_manifest_sha256']:raise ValueError('Actual matrix source changed')
    with np.load(original/'shared_response_basis.npz',allow_pickle=False) as saved:
        latent,projection,slices,tasks=(saved[k].copy() for k in ('latent','projection','task_slices','tasks'))
    reveal=json.loads((original/'reveal.json').read_text())['mapping']
    k=next(int(key.split('_')[1]) for key,value in reveal.items() if value=='baseline')
    with np.load(original/'W_only_geometry.npz',allow_pickle=False) as saved:
        reference=saved['directions'][saved['frame_axes'][saved['case_frame_index'][k]]].T
    with np.load(blind/'blind_rotations.npz',allow_pickle=False) as saved:optimized=saved['optimized'].copy()
    frames=dict(baseline=reference,frozen_fresh_blind=optimized)
    output.mkdir(parents=True);rng=np.random.default_rng(482)
    ids=[np.arange(data.n)]+[np.sort(rng.choice(data.n,int(.8*data.n),replace=False)) for _ in range(replicates)]
    np.savez_compressed(output/'subject_samples.npz',**{f'sample{i:02d}':sample for i,sample in enumerate(ids)})
    protocol=dict(status='RUNNING',model=model,seed=482,replicates=replicates,capacities=[32,64],
        main_fit_subjects=data.n,subset_fraction=.8,all_original_rows=data.m,
        source_scores_sha256=source_hash,frame_sha256=frame_hash,
        source_response_basis_frozen=True,per_subset_U_and_response_recentering=True,
        interpretation='paired conditional-fit stability with frozen observed X bank and output modes; not independent prediction validation',
        frames_selected_before_resampling=True,no_case_labels_used_for_optimization=True,no_error_or_angle_veto=True)
    wave.json_file(output/'protocol.json',protocol);rows=[]
    for sample_index,sample in enumerate(ids):
        for centers in (32,64):
            folder=output/f'sample{sample_index:02d}_centers{centers}'
            scores=paired_refit(data.u,latent,projection,slices,tasks,list(data.channels).index('p_nn'),
                frames,sample,folder,centers,degree=old['degree'])
            baseline=scores[0]['loss'];value=scores[1]['loss']
            row=dict(sample=sample_index,centers=centers,subjects=len(sample),baseline_loss=baseline,
                frozen_blind_loss=value,relative_change_percent=100*(value-baseline)/baseline,
                maximum_efficiency_error=max(r['relative_squared_efficiency_error'] for r in scores))
            if sample_index==0 and centers==32:
                expected=json.loads((blind/'assessment.json').read_text())
                if not np.allclose([baseline,value],[expected['baseline_loss'],expected['blind_minimum_loss']],rtol=1e-7,atol=1e-9):
                    raise ArithmeticError('Independent full-subject refit did not reproduce the frozen source comparison')
            rows.append(row);wave.csv_file(output/'paired_scores.csv',rows)
            print(f'[ShapleyRefits] {model} sample {sample_index}/{replicates}, centers {centers}: {row["relative_change_percent"]:+.6g}%',flush=True)
    summary=[]
    for centers in (32,64):
        subset=[r for r in rows if r['centers']==centers and r['sample']>0]
        gaps=[r['relative_change_percent'] for r in subset]
        summary.append(dict(centers=centers,replicates=len(gaps),blind_lower_count=sum(g<0 for g in gaps),
            q05_median_q95_percent=np.quantile(gaps,[.05,.5,.95]).tolist(),
            all_subject_relative_change_percent=next(r['relative_change_percent'] for r in rows if r['centers']==centers and r['sample']==0)))
    wave.json_file(output/'summary.json',summary)
    wave.json_file(output/'read_audit.json',dict(files=sorted(set(data.read_files)),forbidden_files_read=[]))
    protocol.update(status='COMPLETE');wave.json_file(output/'protocol.json',protocol)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=wave.io.DIMS,default='CR3')
    p.add_argument('--batch',action='store_true')
    p.add_argument('--replicates',type=int,default=12);args=p.parse_args()
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        for model in (('CR3','C1','C2','C3') if args.batch else (args.model,)):
            main(model,args.replicates)
