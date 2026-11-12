"""BW: trace-preserving, sign-sensitive complexity of fixed GP UX profiles."""
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
import isotropic_gp_rotation as radial
import conditional_ux as conditional
import operator_mode_support as mode_support
import norm_lift_profiles as lift
import ux_metric_scope as scope
from run_operator_mode_support import audit_frames


def analyze(source,original,bank_folder,output,extra_frames=(),extra_descriptions=(),device='cuda'):
    original,output=Path(original),Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Retain previous norm-lift run: {output}')
    bank=profiles.MatrixBank(source,bank_folder);data=bank.data
    old=json.loads((original/'protocol.json').read_text())
    if old['status']!='COMPLETE' or old['source_matrix_manifest_sha256']!=bank.protocol['manifest_sha256']:
        raise ValueError('Fixed GP source is incomplete or uses different matrices')
    if hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest()!=old['scores_sha256']:
        raise ValueError('Frozen source scores changed')
    output.mkdir(parents=True)
    protocol=dict(status='RUNNING',model=data.index['model'],all_subjects=data.n,all_original_rows=data.m,
        source_matrix_manifest_sha256=bank.protocol['manifest_sha256'],source_scores_sha256=old['scores_sha256'],
        source_GP_files_sha256={f:hashlib.sha256((original/f).read_bytes()).hexdigest() for f in
            ('source_geometry.npz','source_GP_parameters.npz','all_subject_gradient_covariances.npz')},
        source_fit='frozen all-subject isotropic GP, no refit or functional-form change',
        lift='Psi(g)=(g,||g||)/sqrt(2)',trace_preserving=True,axis_sign_invariant=True,output_mode_basis_invariant=True,
        interpretation='positive-ray derivative-profile complexity; not a general monotone-component count',
        prediction_error_veto=False,derivative_accuracy_independently_verified=False,
        source_quality=list(csv.DictReader((original/'source_fit_quality.csv').open())),extra_frozen_frames=list(extra_descriptions))
    wave.json_file(output/'protocol.json',protocol)
    maps=[wave.io.registration(data.u,data.coordinates(path))[0] for path in data.paths]+list(extra_frames)
    directions,membership=conditional.direction_classes(maps)
    np.savez_compressed(output/'W_only_geometry.npz',directions=directions,membership=membership)
    operators=[np.load(original/f'X_contrast_column{j}.npy',allow_pickle=False) for j in range(data.dx)]
    tasks,targets,discarded,audit=profiles.task_modes(bank.responses,operators,tolerance=1e-6)
    width=max(target.shape[1] for target in targets);values=np.zeros((len(tasks),data.n,width))
    for t,target in enumerate(targets):values[t,:,:target.shape[1]]=target
    with np.load(original/'source_geometry.npz',allow_pickle=False) as saved:
        centers=saved['center_subject_ids'];z=(data.u-saved['U_mean'])/saved['U_scale']
    convert=lambda value,dtype=torch.float64:torch.tensor(value,dtype=dtype,device=device)
    fitted=radial.model(convert(z[None]),convert(values),centers,starts=old['starts'])
    with np.load(original/'source_GP_parameters.npz',allow_pickle=False) as saved:
        selected=saved['selected_start'].copy()
        fitted.restore(dict(eta=convert(saved['inverse_lengthscales']),log_amplitude=convert(np.log(saved['signal_amplitude'])),
            log_noise=convert(np.log(saved['noise_sd']))))
    with np.load(original/'all_subject_gradient_covariances.npz',allow_pickle=False) as saved:expected=saved['covariance']
    spectra={key:np.zeros((len(data.channels),len(directions),data.dx,width+1)) for key in ('ordinary','norm_lift')}
    errors=[];rows=torch.arange(data.n,device=device)
    for t,(c,j) in enumerate(tasks):
        ids=convert([int(selected[t])*len(tasks)+t],torch.long)
        gradient,fit_error=radial.gradient_profiles(fitted,ids,rows);gradient=gradient[0,:,:,:targets[t].shape[1]]
        covariance=torch.einsum('nir,njr->nij',gradient,gradient).cpu().numpy()
        replay=float(np.linalg.norm(covariance-expected[:,c,j])/max(np.linalg.norm(expected[:,c,j]),1e-30))
        if replay>1e-7:raise ArithmeticError('Reconstructed derivative profiles did not replay frozen GP covariance')
        result,trace=lift.spectra(gradient,directions)
        for key,value in result.items():spectra[key][c,:,j,:value.shape[1]]=value
        errors.append(dict(channel=data.channels[c],X_column=j,covariance_replay_relative_L2_error=replay,
            maximum_relative_SVD_trace_error=trace,all_subject_retained_profile_fit_relative_MSE=float(fit_error[0])))
        wave.json_file(output/'gradient_replay_and_trace_audits.json',errors)
        print(f'[NormLift] {data.index["model"]}: task {t+1}/{len(tasks)}, covariance replay {replay:.3g}',flush=True)
    np.savez_compressed(output/'direction_squared_singular_values.npz',**spectra)
    blind=[]
    for view,values in spectra.items():
        for k,axes in enumerate(membership):
            for c,channel in enumerate(data.channels):
                for metric,value in mode_support.scores(values[c,axes]).items():
                    blind.append(dict(anonymous_case=f'case_{k:03d}' if k<len(data.paths) else f'audit_frame_{k-len(data.paths):03d}',
                        channel=channel,metric=view+'_'+metric,loss=value,is_control=view=='ordinary' or not scope.primary(metric)))
    wave.csv_file(output/'blind_scores.csv',blind)
    digest=hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    reveal={f'case_{k:03d}':path.parent.name for k,path in enumerate(data.paths)}
    wave.json_file(output/'reveal.json',dict(scores_sha256=digest,mapping=reveal))
    grouped=defaultdict(dict);controls={};extra=[]
    for r in blind:
        key=r['channel'],r['metric'];controls[key]=r['is_control']
        if r['anonymous_case'] in reveal:grouped[key][reveal[r['anonymous_case']]]=r['loss']
        else:extra.append(r)
    summary=[dict(channel=c,metric=m,**wave.st.evaluate_curve(losses,data.index['cases']),is_control=controls[c,m],
        baseline_loss=losses['baseline']) for (c,m),losses in grouped.items()]
    wave.csv_file(output/'metric_summary.csv',summary)
    if extra:wave.csv_file(output/'frozen_frame_assessment.csv',[dict(**r,
        relative_change_percent=100*(r['loss']-grouped[r['channel'],r['metric']]['baseline'])/
            max(abs(grouped[r['channel'],r['metric']]['baseline']),1e-30)) for r in extra])
    wave.json_file(output/'read_audit.json',dict(files=sorted(set(data.read_files)),forbidden_files_read=[]))
    protocol.update(status='COMPLETE',scores_sha256=digest);wave.json_file(output/'protocol.json',protocol)
    print(f'[NormLift] {data.index["model"]}: {sum(r["all_planes_pass"] and not r["is_control"] for r in summary)} primary grid combinations',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=wave.io.DIMS,default='CR3')
    p.add_argument('--batch',action='store_true');args=p.parse_args()
    with threadpool_limits(limits=4):
        torch.set_num_threads(4);common=None
        for model in (('CR3','C1','C2','C3') if args.batch else (args.model,)):
            source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
            output=parent/'waveBW_norm_lift_GP_profiles_seed416'
            if not output.exists():
                frames,descriptions=audit_frames(parent,model)
                analyze(source,parent/'waveBB_radial_GP_fixed_function_seed416',parent/'matrix_response_table_seed383',
                    output,extra_frames=frames,extra_descriptions=descriptions)
            protocol=json.loads((output/'protocol.json').read_text())
            if protocol['status']!='COMPLETE' or hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()!=protocol['scores_sha256']:
                raise ValueError('Existing norm-lift pilot is incomplete or changed')
            with (output/'metric_summary.csv').open() as stream:rows=list(csv.DictReader(stream))
            passed={(r['channel'],r['metric']) for r in rows if r['all_planes_pass']=='True' and r['is_control']!='True'}
            common=passed if common is None else common&passed
            print(f'[WaveBW] {model}: {len(passed)} strict grid combinations, {len(common)} common',flush=True)
