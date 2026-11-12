"""BN: one-degree audits and matrix lower bounds for the fixed BB2 GP criterion."""
import csv
import hashlib
import json
from collections import defaultdict

import numpy as np
from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as profiles
import pooled_support_certificate as certificate


def main():
    output=wave.ROOT/'outputs/supervised_abs_rotation/waveBN_fixed_GP_support_certificate'
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Retain earlier support certificate: {output}')
    output.mkdir(parents=True);summary=[];all_curves=[]
    for model in ('CR3','C1','C2','C3'):
        source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics'
        folder=parent/'waveBB2_rank_radial_fixed_function_seed416'
        protocol=json.loads((folder/'protocol.json').read_text())
        digest=hashlib.sha256((folder/'blind_scores.csv').read_bytes()).hexdigest()
        if protocol['status']!='COMPLETE' or digest!=protocol['scores_sha256']:raise ValueError('Frozen radial GP source changed')
        bank=profiles.MatrixBank(source,parent/'matrix_response_table_seed383');data=bank.data
        baseline=next(path for path in data.paths if path.parent.name=='baseline')
        q,_=wave.io.registration(data.u,data.coordinates(baseline))
        with np.load(folder/'all_subject_gradient_covariances.npz',allow_pickle=False) as saved:
            h=saved['covariance'][:,0].mean(0)
        normalized=certificate.normalized_matrices(h)
        bound,eigen,by_x=certificate.lower_bound(normalized)
        base=certificate.score(normalized,q)
        assessment=json.loads((folder/'assessment.json').read_text())
        if protocol['primary_blind_metric']!='pooled_UX_columns_worst_soft_count_eps003' or not np.isclose(base,assessment['baseline_loss'],rtol=1e-10,atol=1e-10):
            raise ArithmeticError('Matrix certificate does not reproduce the frozen primary score')
        curves=certificate.fine_plane_grid(normalized,q)
        all_curves.extend(dict(model=model,**r,relative_change_percent=100*(r['loss']-base)/base) for r in curves)
        best=min(curves,key=lambda r:r['loss'])
        rows=defaultdict(list)
        for r in curves:rows[r['U_i'],r['U_j']].append(r)
        plane_best=[dict(U_i=i,U_j=j,**{k:min(v,key=lambda r:r['loss'])[k] for k in ('angle_deg','loss')}) for (i,j),v in rows.items()]
        row=dict(model=model,all_subjects=data.n,original_rows=data.m,baseline_loss=base,
            optimistic_global_lower_bound=bound,maximum_possible_score_improvement_percent=100*(base-bound)/base,
            best_one_degree_plane_loss=best['loss'],best_one_degree_plane_angle_deg=best['angle_deg'],
            best_plane=[best['U_i'],best['U_j']],fine_grid_change_percent=100*(best['loss']-base)/base,
            previous_blind_minimum_loss=assessment['blind_optimized_minimum_loss'],source_scores_sha256=digest)
        summary.append(row)
        np.savez_compressed(output/f'{model}_normalized_matrices.npz',normalized_gradient_covariances=normalized,
            reference_frame=q,eigenvalues=eigen,individual_X_bounds=by_x)
        wave.json_file(output/f'{model}_plane_optima.json',plane_best)
        wave.csv_file(output/'one_degree_curves.csv',all_curves)
        wave.json_file(output/'assessment.json',summary)
        print('[Pooled support certificate]',row,flush=True)
    wave.json_file(output/'protocol.json',dict(status='COMPLETE',metric='logit, fixed radial GP, pooled worst-X soft support eps=.03',
        scores_from_all_subject_covariances=True,fine_sweep_is_labelled_local_audit=True,no_new_function_fit=True,
        bound_interpretation='exact optimistic bound for this frozen GP matrix statistic; not a bound for the unknown response function',
        no_optimum_position_veto=True,angle_units='degrees',gap_units='relative percent'))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
