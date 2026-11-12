"""BO: stronger label-blind optimization of the frozen pooled GP support statistic."""
import hashlib
import json

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

import run as wave
import pooled_support_certificate as certificate
import tree_rotation_stability as rotations
from record_blind_axis_positions import axis_angles


def main():
    source=wave.ROOT/'outputs/supervised_abs_rotation/waveBN_fixed_GP_support_certificate'
    output=wave.ROOT/'outputs/supervised_abs_rotation/waveBO_stronger_pooled_GP_search_seed462'
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Retain earlier stronger search: {output}')
    if json.loads((source/'protocol.json').read_text())['status']!='COMPLETE':raise ValueError('Matrix source is incomplete')
    output.mkdir(parents=True);assessments=[]
    protocol=dict(status='RUNNING',seed=462,order=['CR3','C1','C2','C3'],epsilons=[.01,.03,.1],
        random_starts_proposed=128,additional_proposals='anonymous W coordinate frame; eigenbasis of each X covariance and their mean',
        optimized_starts=12,maximum_evaluations_per_start=2000,optimizer='Powell in skew-matrix exponential coordinates',
        function_source='unchanged fixed BB2 GP function, fitted using all 5000 subjects',
        evaluation_source='all-subject pooled gradient covariances',labels_used_by_optimizer=False,
        code_sha256=hashlib.sha256((wave.HERE/'run_stronger_pooled_GP_search.py').read_bytes()).hexdigest())
    wave.json_file(output/'protocol.json',protocol)
    for model in protocol['order']:
        matrix_file=source/f'{model}_normalized_matrices.npz'
        with np.load(matrix_file,allow_pickle=False) as saved:
            # The labelled reference frame in this archive is not accessed
            # until all searches for this model have frozen below.
            h=saved['normalized_gradient_covariances'].copy()
        d=h.shape[-1];rng=np.random.default_rng(462)
        proposals=[np.eye(d)]+[np.linalg.eigh(a)[1] for a in [*h,h.mean(0)]]
        proposals.extend(rotations.random_rotation(d,rng) for _ in range(128))
        model_output=output/model;model_output.mkdir();scores=[];histories=[];optimized=[];best_parameters=[];start_frames=[]
        for epsilon in protocol['epsilons']:
            values=[certificate.score(h,q,epsilon) for q in proposals]
            selected=np.argsort(values)[:12]
            for start,index in enumerate(selected):
                q0=proposals[index];best_loss=float(values[index]);best_q=q0.copy();best_p=np.zeros(d*(d-1)//2);evaluations=0
                def objective(parameters):
                    nonlocal best_loss,best_q,best_p,evaluations
                    q=q0@rotations.increment(parameters,d);loss=certificate.score(h,q,epsilon);evaluations+=1
                    histories.append(dict(epsilon=epsilon,start=start,evaluation=evaluations,loss=loss))
                    if loss<best_loss:best_loss,best_q,best_p=loss,q.copy(),np.array(parameters).copy()
                    return loss
                result=minimize(objective,np.zeros(d*(d-1)//2),method='Powell',
                    bounds=[(-np.pi/3,np.pi/3)]*(d*(d-1)//2),
                    options=dict(maxfev=2000,xtol=1e-5,ftol=1e-10))
                verified=certificate.score(h,best_q,epsilon)
                if not np.isclose(verified,best_loss,rtol=1e-12,atol=1e-12):raise ArithmeticError('Frozen best score did not reproduce')
                scores.append(dict(epsilon=epsilon,start=start,proposal_index=int(index),initial_loss=values[index],
                    optimized_loss=best_loss,evaluations=evaluations,optimizer_converged=bool(result.success)))
                optimized.append(best_q);best_parameters.append(best_p);start_frames.append(q0)
                print(f'[WaveBO] {model} eps={epsilon:g} start {start+1}/12: {best_loss:.9g}',flush=True)
            wave.csv_file(model_output/'optimization_history.csv',histories)
            wave.csv_file(model_output/'optimized_scores.csv',scores)
        np.savez_compressed(model_output/'blind_frames.npz',proposals=np.array(proposals),optimized=np.array(optimized),
            best_parameters=np.array(best_parameters),start_frames=np.array(start_frames))
        freeze=dict(scores_sha256=hashlib.sha256((model_output/'optimized_scores.csv').read_bytes()).hexdigest(),
            frames_sha256=hashlib.sha256((model_output/'blind_frames.npz').read_bytes()).hexdigest(),
            source_matrices_sha256=hashlib.sha256(matrix_file.read_bytes()).hexdigest())
        wave.json_file(model_output/'blind_freeze.json',freeze)
        # Labelled assessment begins after every epsilon and score has frozen.
        with np.load(matrix_file,allow_pickle=False) as saved:reference=saved['reference_frame'].copy()
        for epsilon in protocol['epsilons']:
            eligible=[k for k,r in enumerate(scores) if r['epsilon']==epsilon]
            k=min(eligible,key=lambda k:scores[k]['optimized_loss']);row=scores[k];base=certificate.score(h,reference,epsilon)
            bound,_,_=certificate.lower_bound(h,epsilon)
            assessment=dict(model=model,epsilon=epsilon,baseline_loss=base,optimized_loss=row['optimized_loss'],
                relative_change_percent=100*(row['optimized_loss']-base)/base,optimistic_lower_bound=bound,
                remaining_bound_gap_percent=100*(row['optimized_loss']-bound)/max(abs(row['optimized_loss']),1e-30),
                **axis_angles(reference,optimized[k]),global_optimality_proven=False)
            assessments.append(assessment);print('[WaveBO assessment]',assessment,flush=True)
        wave.json_file(output/'assessment.json',assessments)
    protocol.update(status='COMPLETE');wave.json_file(output/'protocol.json',protocol)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
