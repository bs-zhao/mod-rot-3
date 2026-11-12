"""Refit two fixed kernel-scoring frames on paired subject subsets.

This audits a labelled reference frame against an already frozen blind
counterexample; it is not another blind optimization. Quantiles describe
subsampling stability, not confidence intervals for a population parameter.
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
import matrix_function_profiles as profiles
import conditional_ui_witness as ui
import conditional_ux as conditional
import run_conditional_kernel_moments as kernel


def frame_scores(bank,ids,frames,operators,centers=32,device='cuda'):
    tasks,targets,discarded,_=profiles.task_modes(bank.responses[ids,:1],operators,tolerance=1e-8)
    width=max(t.shape[1] for t in targets);values=np.zeros((len(ids),len(tasks),width))
    for t,target in enumerate(targets):values[:,t,:target.shape[1]]=target
    flat=torch.tensor(values.reshape(len(ids),-1),dtype=torch.float64,device=device)
    norm=torch.tensor(np.sum(values*values,axis=-1),dtype=torch.float64,device=device)
    u=bank.data.u[ids];z=(u-u.mean(0))/np.sqrt(np.var(u,axis=0).mean())
    results=[];audits=[]
    for f,q in enumerate(frames):
        graphs={}
        for i,v in enumerate(q.T):
            features,audit=ui.features(z,v,seed=430,centers=centers)
            audits.append(dict(frame=f,axis=i,**audit))
            for view,feature in features.items():
                raw,corrected=kernel.moments(torch.tensor(feature,dtype=torch.float64,device=device),flat,norm,width)
                for estimator,scores in (('raw',raw),('diagonal_removed',corrected)):
                    key=view+'_'+estimator
                    if key not in graphs:graphs[key]=np.zeros((bank.data.d,bank.data.dx))
                    graphs[key][i]=scores
        for view,graph in graphs.items():
            results.extend(dict(frame=f,metric=view+'_'+metric,loss=value)
                for metric,value in conditional.sparsity(np.maximum(graph,0.)).items())
    return results,dict(subjects=len(ids),maximum_discarded_relative_energy=float(np.max(discarded.sum(1)/len(ids))),Ui_audits=audits)


def analyze(source,original,folder,counterexample,output,draws=24,fraction=.8,seed=451,device='cuda'):
    original,counterexample,output=map(Path,(original,counterexample,output))
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Retain earlier paired stability run: {output}')
    protocol=json.loads((original/'protocol.json').read_text())
    source_digest=hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest()
    if protocol['status']!='COMPLETE' or source_digest!=protocol['scores_sha256']:raise ValueError('Original kernel source changed')
    frozen=json.loads((counterexample/'blind_freeze.json').read_text())
    rotation_digest=hashlib.sha256((counterexample/'blind_rotations.npz').read_bytes()).hexdigest()
    if rotation_digest!=frozen['rotations_sha256']:raise ValueError('Frozen frame changed')
    bank=profiles.MatrixBank(source,folder);data=bank.data
    if bank.quality['mode']!='numerically verified shared observed X':
        raise ValueError('This paired audit requires exact shared actual X, without profile reinterpolation')
    if bank.protocol['manifest_sha256']!=protocol['source_matrix_manifest_sha256']:raise ValueError('Actual response cache changed')
    reveal=json.loads((original/'reveal.json').read_text())['mapping']
    k=next(int(k.split('_')[1]) for k,v in reveal.items() if v=='baseline')
    with np.load(original/'W_only_geometry.npz',allow_pickle=False) as saved:
        reference=saved['directions'][saved['membership'][k]].T
    with np.load(counterexample/'blind_rotations.npz',allow_pickle=False) as saved:counter=saved['optimized'].copy()
    frames=[reference,counter]
    operators=[np.load(original/f'X_witness_column{j}.npy',allow_pickle=False) for j in range(data.dx)]
    output.mkdir(parents=True);rng=np.random.default_rng(seed);rows=[];draw_ids=[]
    status=dict(status='RUNNING',draws=draws,fraction=fraction,seed=seed,original_subjects=data.n,
        original_rows=data.m,source_scores_sha256=source_digest,rotations_sha256=rotation_digest,
        frame_selection='labelled reference vs already frozen blind frame; no reoptimization',
        refit='new subject-specific response centering, direct target SVD and W-only conditional Ui features in each draw',
        fixed='actual observed shared X witness operators; kernel settings; compared frames',
        interpretation='paired sampling without replacement; stability quantiles, not population confidence intervals')
    wave.json_file(output/'protocol.json',status)
    for draw in range(draws):
        ids=np.sort(rng.choice(data.n,int(fraction*data.n),replace=False));draw_ids.append(ids)
        scores,audit=frame_scores(bank,ids,frames,operators,centers=protocol['Ui_kernel_centers'],device=device)
        rows.extend(dict(draw=draw,**r) for r in scores)
        wave.csv_file(output/'paired_scores.csv',rows)
        wave.json_file(output/f'draw{draw:03d}_refit_audit.json',audit)
        print(f'[PairedKernel] subject refits {draw+1}/{draws}',flush=True)
    np.savez_compressed(output/'subject_subsets.npz',subject_indices=np.array(draw_ids),frames=np.array(frames))
    digest=hashlib.sha256((output/'paired_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'scores_freeze.json',dict(paired_scores_sha256=digest))
    paired=defaultdict(dict)
    for r in rows:paired[r['metric'],r['draw']][r['frame']]=r['loss']
    margins=defaultdict(list)
    for (metric,draw),values in paired.items():margins[metric].append((values[1]-values[0])/max(abs(values[0]),1e-30))
    summary=[dict(metric=metric,draws=len(values),counter_lower_draws=int(np.sum(np.array(values)<0.)),
        margin_q05=float(np.quantile(values,.05)),margin_median=float(np.median(values)),margin_q95=float(np.quantile(values,.95)))
        for metric,values in margins.items()]
    wave.csv_file(output/'stability_summary.csv',summary)
    wave.json_file(output/'read_audit.json',dict(files=sorted(set(data.read_files)),forbidden_files_read=[]))
    status.update(status='COMPLETE',paired_scores_sha256=digest);wave.json_file(output/'protocol.json',status)


if __name__=='__main__':
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        source=wave.io.default_step2(wave.ROOT,'CR3');parent=source.parent/'step5_matrix_metrics'
        analyze(source,parent/'waveBC2_conditional_kernel_centers32_seed430',parent/'matrix_response_table_seed383',
            parent/'waveBC3_conditional_kernel_blind_seed439',parent/'waveBI_kernel_paired_subjects_seed451')
