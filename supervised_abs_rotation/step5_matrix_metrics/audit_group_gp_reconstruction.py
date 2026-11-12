"""Reconstruct saved subset GP posteriors and evaluate component functions directly."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run as wave
import group_sparse_gp as gp
import hierarchical_group_gp as hierarchy
import subset_kernel_group_gp as subset
import conditional_ui_witness as ui
import matrix_function_profiles as profiles


def main(tag):
    run_name={'BJ':'waveBJ_hierarchical_group_GP_seed448','BM':'waveBM_subset_kernel_group_GP_seed448'}[tag]
    factory=hierarchy.dictionary if tag=='BJ' else subset.dictionary
    output=wave.ROOT/f'outputs/supervised_abs_rotation/wave{tag}_posterior_reconstruction_audit'
    output.mkdir(parents=True,exist_ok=True);rows=[];details=[]
    for model in ('CR3','C1','C2','C3'):
        source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics';folder=parent/run_name
        protocol=json.loads((folder/'protocol.json').read_text())
        if protocol['status']!='COMPLETE':raise ValueError(f'{model} source is incomplete')
        digest=hashlib.sha256((folder/'blind_scores.csv').read_bytes()).hexdigest()
        if digest!=protocol['scores_sha256']:raise ValueError('Frozen source scores changed')
        if hashlib.sha256(Path(factory.__code__.co_filename).read_bytes()).hexdigest()!=protocol['dictionary_code_sha256']:
            raise ValueError('Saved posterior used a different W-only dictionary implementation')
        bank=profiles.MatrixBank(source,parent/'matrix_response_table_seed383');data=bank.data
        old=parent/'waveBC2_conditional_kernel_centers32_seed430'
        operators=[np.load(old/f'X_witness_column{j}.npy',allow_pickle=False) for j in range(data.dx)]
        tasks,targets,_,_=profiles.task_modes(bank.responses,operators,tolerance=1e-8)
        width=max(t.shape[1] for t in targets);values=np.zeros((len(tasks),data.n,width))
        for t,target in enumerate(targets):values[t,:,:target.shape[1]]=target
        y=torch.tensor(values,dtype=torch.float64,device='cuda')
        z=data.u-data.u.mean(0);z/=np.sqrt(np.var(data.u,axis=0).mean())
        radial,_=ui.nystrom(z,protocol['radial_centers'],448)
        reveal=json.loads((folder/'reveal.json').read_text())['mapping']
        k=next(int(k.split('_')[1]) for k,v in reveal.items() if v=='baseline')
        with np.load(folder/'W_only_geometry.npz',allow_pickle=False) as saved:
            directions=saved['directions'];frame_axes=saved['frame_axes'];indices=saved['case_frame_index']
        selected=[('reference',int(indices[k]))]
        if model=='CR3' and len(indices)>len(data.paths):selected.append(('fixed_counterexample',int(indices[-1])))
        for label,f in selected:
            rotated=z@directions[frame_axes[f]].T
            for degree in protocol['degrees']:
                phi,groups,masks,audit=factory(rotated,degree,radial)
                x=torch.tensor(phi,dtype=torch.float64,device='cuda');g=x.T@x;cross=x.T@y;yy=y.square().sum((1,2))
                with np.load(folder/f'frame{f:03d}_degree{degree}.npz',allow_pickle=False) as saved:
                    a=torch.tensor(saved['group_variances'],dtype=torch.float64,device='cuda')
                    noise=torch.tensor(saved['noise_variances'],dtype=torch.float64,device='cuda')
                    expected=saved['component_energies'];expected_loss=saved['evidence_per_subject']
                loss,coef,cov=gp.system(g,cross,yy,torch.tensor(groups,device='cuda'),a,noise,data.n)
                observed=[]
                for group in range(len(masks)):
                    ids=np.flatnonzero(groups==group)
                    function=x[:,ids]@coef[:,ids]
                    observed.append(function.square().sum((1,2)).cpu().numpy()/data.n)
                observed=np.stack(observed,1)
                component_error=float(np.max(np.abs(observed-expected)))
                density_error=float(np.max(np.abs(loss.cpu().numpy()/data.n-expected_loss)))
                if not np.allclose(observed,expected,rtol=1e-7,atol=1e-10) or density_error>1e-7:
                    raise ArithmeticError('Saved component functions or marginal density did not reconstruct')
                rows.append(dict(model=model,frame_label=label,frame=f,degree=degree,all_subjects=data.n,
                    original_rows=data.m,maximum_component_energy_absolute_error=component_error,
                    maximum_density_per_subject_absolute_error=density_error,
                    maximum_lower_span_cross_covariance=max((r['normalized_lower_span_cross_covariance'] for r in audit['hierarchy']),default=0.)))
                details.append(dict(model=model,frame_label=label,degree=degree,dictionary_audit=audit))
                print('[Group posterior reconstruction]',rows[-1],flush=True)
        wave.csv_file(output/'reconstruction_errors.csv',rows)
        wave.json_file(output/'W_only_dictionary_audits.json',details)
    wave.json_file(output/'protocol.json',dict(status='COMPLETE',source_wave=tag,all_subjects_per_model=5000,
        component_check='evaluate saved component functions at actual subject U and compare their empirical energies',
        source_dictionary_implementation_checked=True,no_refit_or_new_optimization=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',choices=('BJ','BM'),default='BJ');args=p.parse_args()
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        with torch.no_grad():main(args.tag)
