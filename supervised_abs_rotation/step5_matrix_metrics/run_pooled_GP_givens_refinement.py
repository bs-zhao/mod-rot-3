"""BP: exact coordinate-line refinement, with separate blind and provided-case arms."""
import argparse
import csv
import hashlib
import json

import numpy as np
from threadpoolctl import threadpool_limits

import run as wave
import pooled_support_certificate as certificate
import pooled_givens_refinement as givens
from record_blind_axis_positions import axis_angles


def main(best_decrease=False):
    matrices=wave.ROOT/'outputs/supervised_abs_rotation/waveBN_fixed_GP_support_certificate'
    previous=wave.ROOT/'outputs/supervised_abs_rotation/waveBO_stronger_pooled_GP_search_seed462'
    output=wave.ROOT/('outputs/supervised_abs_rotation/waveBP2_maximum_decrease_GP_refinement' if best_decrease else
                      'outputs/supervised_abs_rotation/waveBP_exact_Givens_GP_refinement')
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Retain earlier exact refinement: {output}')
    if json.loads((previous/'protocol.json').read_text())['status']!='COMPLETE':raise ValueError('Prior blind search is incomplete')
    output.mkdir(parents=True);assessments=[]
    protocol=dict(status='RUNNING',epsilons=[.01,.03,.1],passes=12,
        pair_selection='maximum available decrease, up to 60 steps' if best_decrease else '12 fixed-order cycles',
        line_search='complete 90-degree fundamental interval, .25-degree grid plus bounded scalar refinement',
        arms=['blind-frame refinement','provided anonymous W-case initialization'],
        provided_frame_source='our W-only case registrations; original saved R matrices are not read',
        provided_case_labels_used=False,reference_frame_used_only_after_freeze=True)
    wave.json_file(output/'protocol.json',protocol)
    for model in ('CR3','C1','C2','C3'):
        source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
        with np.load(matrices/f'{model}_normalized_matrices.npz',allow_pickle=False) as saved:
            h=saved['normalized_gradient_covariances'].copy()
        old=previous/model;frozen=json.loads((old/'blind_freeze.json').read_text())
        if hashlib.sha256((old/'blind_frames.npz').read_bytes()).hexdigest()!=frozen['frames_sha256']:
            raise ValueError('Previous blind frames changed')
        with np.load(old/'blind_frames.npz',allow_pickle=False) as saved:blind_frames=saved['optimized'].copy()
        with (old/'optimized_scores.csv').open() as stream:blind_rows=list(csv.DictReader(stream))
        geometry=parent/'waveBC2_conditional_kernel_centers32_seed430/W_only_geometry.npz'
        with np.load(geometry,allow_pickle=False) as saved:
            dirs,membership=saved['directions'],saved['membership']
        keys=list(dict.fromkeys(tuple(sorted(axes)) for axes in membership))
        provided=[dirs[list(axes)].T for axes in keys]
        model_output=output/model;model_output.mkdir();scores=[];frames=[];histories=[]
        for epsilon in protocol['epsilons']:
            original=[blind_frames[k] for k,r in enumerate(blind_rows) if float(r['epsilon'])==epsilon]
            best_provided=sorted(provided,key=lambda q:certificate.score(h,q,epsilon))[:4]
            for arm,starts in (('blind',original),('provided_cases',best_provided)):
                for start,initial in enumerate(starts):
                    initial_value=certificate.score(h,initial,epsilon)
                    q,value,history=(givens.refine_best_decrease(h,initial,epsilon,steps=60) if best_decrease else
                                     givens.refine(h,initial,epsilon,passes=12))
                    if value>initial_value+1e-10:raise ArithmeticError('Exact line refinement increased its fixed score')
                    frames.append(q);scores.append(dict(epsilon=epsilon,arm=arm,start=start,initial_loss=initial_value,loss=value))
                    histories.extend(dict(epsilon=epsilon,arm=arm,start=start,**r) for r in history)
                print(f'[WaveBP] {model} eps={epsilon:g} {arm}: {min(r["loss"] for r in scores if r["epsilon"]==epsilon and r["arm"]==arm):.9g}',flush=True)
            wave.csv_file(model_output/'optimization_history.csv',histories)
            wave.csv_file(model_output/'refined_scores.csv',scores)
        np.savez_compressed(model_output/'refined_frames.npz',frames=np.array(frames))
        wave.json_file(model_output/'blind_freeze.json',dict(scores_sha256=hashlib.sha256((model_output/'refined_scores.csv').read_bytes()).hexdigest(),
            frames_sha256=hashlib.sha256((model_output/'refined_frames.npz').read_bytes()).hexdigest(),
            provided_W_geometry_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest()))
        with np.load(matrices/f'{model}_normalized_matrices.npz',allow_pickle=False) as saved:
            reference=saved['reference_frame'].copy()
        for epsilon in protocol['epsilons']:
            for arm in ('blind','provided_cases'):
                eligible=[k for k,r in enumerate(scores) if r['epsilon']==epsilon and r['arm']==arm]
                k=min(eligible,key=lambda k:scores[k]['loss']);value=scores[k]['loss'];base=certificate.score(h,reference,epsilon)
                bound,_,_=certificate.lower_bound(h,epsilon)
                assessments.append(dict(model=model,epsilon=epsilon,arm=arm,baseline_loss=base,optimized_loss=value,
                    relative_change_percent=100*(value-base)/base,optimistic_lower_bound=bound,
                    remaining_bound_gap_percent=100*(value-bound)/max(abs(value),1e-30),
                    **axis_angles(reference,frames[k]),global_optimality_proven=False))
        wave.json_file(output/'assessment.json',assessments)
    protocol.update(status='COMPLETE');wave.json_file(output/'protocol.json',protocol)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--best-decrease',action='store_true');args=p.parse_args()
    with threadpool_limits(limits=4):main(args.best_decrease)
