"""Fixed-hyperparameter PRESS of conditional emulators in one anonymous W frame.

Proper coalitions intentionally omit U coordinates, so their prediction error
includes irreducible omitted-input variation. No absolute MSE veto is used.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as profiles
import conditional_ux as conditional
import conditional_model_quality as quality
from record_rotation_optima import WAVES


def key(basis):
    matrix=np.round(basis@basis.T,6);matrix[matrix==0.]=0.
    return basis.shape[1],hashlib.sha256(matrix.tobytes()).hexdigest()


def analyze(bank,original,output,device='cuda'):
    old=json.loads((original/'protocol.json').read_text());data=bank.data
    source_hash=hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest()
    if old['status']!='COMPLETE' or source_hash!=old['scores_sha256']:raise ValueError('Conditional source changed')
    if old['source_matrix_manifest_sha256']!=bank.protocol['manifest_sha256']:raise ValueError('Actual response source changed')
    if (output/'protocol.json').exists():
        previous=json.loads((output/'protocol.json').read_text())
        if previous['status']=='COMPLETE' and previous['source_scores_sha256']==source_hash:
            if hashlib.sha256((output/'task_quality.csv').read_bytes()).hexdigest()!=previous['audit_scores_sha256']:
                raise ValueError('Frozen conditional quality audit changed')
            if hashlib.sha256((original/'shared_response_basis.npz').read_bytes()).hexdigest()!=previous['source_basis_sha256']:
                raise ValueError('Frozen conditional response basis changed')
            for entry in previous['used_model_files']:
                if hashlib.sha256((original/'conditional_models'/entry['file']).read_bytes()).hexdigest()!=entry['sha256']:
                    raise ValueError('A frozen audited conditional model changed')
            with (output/'task_quality.csv').open() as stream:return list(csv.DictReader(stream))
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Retain earlier conditional quality audit: {output}')
    output.mkdir(parents=True)
    with np.load(original/'shared_response_basis.npz',allow_pickle=False) as saved:
        target=torch.tensor(saved['latent'],dtype=torch.float64,device=device)
        right=saved['projection'].copy();slices=saved['task_slices'].copy();tasks=saved['tasks'].copy()
    task_projection=[torch.tensor(right[:,a:b],dtype=torch.float64,device=device) for a,b in slices]
    protocol=dict(status='RUNNING',model=data.index['model'],all_subjects=data.n,all_original_rows=data.m,
        source_scores_sha256=source_hash,source_matrix_manifest_sha256=bank.protocol['manifest_sha256'],
        source_basis_sha256=hashlib.sha256((original/'shared_response_basis.npz').read_bytes()).hexdigest(),
        evaluation='all proper coalitions of the anonymous observed-W coordinate frame',
        hyperparameters_refitted=False,validation='exact fixed-design, fixed-ridge PRESS; model/ridge selection is not nested',
        absolute_error_veto=False,omitted_U_variation_included_in_error=True,case_angle_labels_used=False)
    wave.json_file(output/'protocol.json',protocol)
    masks={key(np.eye(data.d)[:,[i for i in range(data.d) if mask&(1<<i)]]):mask for mask in range(1,2**data.d-1)}
    remaining=set(masks);z=data.u-data.u.mean(0);z/=np.sqrt(np.var(data.u,axis=0).mean())
    rows=[];audits=[];files=[]
    for file in sorted((original/'conditional_models').glob('model*.npz')):
        with np.load(file,allow_pickle=False) as saved:
            basis=saved['basis'].copy();identifier=key(basis)
            if identifier not in remaining:continue
            coefficient=saved['coefficient'].copy();length=float(saved['lengthscale']);ridge=float(saved['selected_ridge'])
        mask=masks[identifier]
        design,design_audit=conditional.subspace_design(z,basis,old['degree'],old['radial_centers'],length,old['seed'])
        train,PRESS,audit=quality.audit(torch.tensor(design,dtype=torch.float64,device=device),
            torch.tensor(coefficient,dtype=torch.float64,device=device),target,design_audit['linear_count'],ridge)
        for t,((c,j),p) in enumerate(zip(tasks,task_projection)):
            total=float((target@p).square().sum());normalizer=max(total,1e-30)
            rows.append(dict(coalition_mask=mask,input_dimension=mask.bit_count(),channel=data.channels[int(c)],X_column=int(j),
                training_relative_MSE=float((train@p).square().sum())/normalizer,
                PRESS_relative_MSE=float((PRESS@p).square().sum())/normalizer,selected_ridge=ridge,
                effective_degrees_of_freedom=audit['effective_degrees_of_freedom']))
        audits.append(dict(coalition_mask=mask,input_dimension=mask.bit_count(),**audit))
        files.append(dict(file=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        remaining.remove(identifier);wave.json_file(output/'numerical_and_leverage_audits.json',audits)
        print(f'[ConditionalQuality] {data.index["model"]} {original.name}: {len(masks)-len(remaining)}/{len(masks)} coalitions',flush=True)
        if not remaining:break
    if remaining:raise ValueError('Frozen source lacks an anonymous-frame conditional model')
    wave.csv_file(output/'task_quality.csv',rows)
    protocol.update(status='COMPLETE',used_model_files=files,
        audit_scores_sha256=hashlib.sha256((output/'task_quality.csv').read_bytes()).hexdigest())
    wave.json_file(output/'protocol.json',protocol)
    return rows


def main(models,tags):
    output=wave.ROOT/'outputs/supervised_abs_rotation/conditional_emulator_quality';output.mkdir(parents=True,exist_ok=True)
    availability=[];all_rows=[]
    for model in models:
        source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
        bank=profiles.MatrixBank(source,parent/'matrix_response_table_seed383')
        for tag in tags:
            original=parent/WAVES[tag]
            if not (original/'protocol.json').exists():availability.append(dict(model=model,family=tag,status='NOT_RUN'));continue
            status=json.loads((original/'protocol.json').read_text())['status']
            if status!='COMPLETE':availability.append(dict(model=model,family=tag,status='SOURCE_'+status));continue
            rows=analyze(bank,original,output/model/tag)
            all_rows.extend(dict(model=model,family=tag,**r) for r in rows)
            availability.append(dict(model=model,family=tag,status='COMPLETE'))
        wave.json_file(output/model/'read_audit.json',dict(files=sorted(set(bank.data.read_files)),forbidden_files_read=[]))
    grouped=defaultdict(list)
    for r in all_rows:grouped[r['model'],r['family'],int(r['input_dimension']),r['channel']].append(r)
    summary=[]
    for (model,tag,dimension,channel),rows in grouped.items():
        errors=[float(r['PRESS_relative_MSE']) for r in rows]
        summary.append(dict(model=model,family=tag,input_dimension=dimension,channel=channel,
            mean_PRESS_relative_MSE=float(np.mean(errors)),minimum_PRESS_relative_MSE=min(errors),maximum_PRESS_relative_MSE=max(errors),
            mean_training_relative_MSE=float(np.mean([float(r['training_relative_MSE']) for r in rows]))))
    wave.csv_file(output/'family_dimension_quality.csv',summary);wave.json_file(output/'availability.json',availability)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=wave.io.DIMS,default='CR3')
    p.add_argument('--batch',action='store_true');p.add_argument('--tags',nargs='+',default=['BU','BU3','CB','CC']);args=p.parse_args()
    with threadpool_limits(limits=4):
        torch.set_num_threads(4);main(('CR3','C1','C2','C3') if args.batch else (args.model,),args.tags)
