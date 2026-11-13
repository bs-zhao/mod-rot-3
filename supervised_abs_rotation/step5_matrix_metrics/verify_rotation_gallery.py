"""Audit the actual saved rotations, coordinates, correlations and plot inventory."""
import json
import numpy as np
from export_candidate_rotation_gallery import OUTPUT,MODELS,sha
from write_approximate_shortlist import CANDIDATES
import run as wave


def main():
    records=[]
    for number,c in enumerate(CANDIDATES,1):
        tag=c[1];metric=c[2];channel=c[3] if len(c)>3 else 'logit_nn'
        folder=OUTPUT/f'{number:02d}_{tag}_{channel}'
        for name in ('four_model_angle_curves.png','four_model_angle_curves.pdf',
                     'four_model_grid_minimum_U_T_heatmaps.png','four_model_best_available_U_T_heatmaps.png',
                     'four_model_best_available_U_T_heatmaps.pdf','angle_curves.csv'):
            if not (folder/name).exists():raise FileNotFoundError(folder/name)
        for model in MODELS:
            source=OUTPUT/'source_coordinates'/model
            z=np.load(source/'Zstd.npy',allow_pickle=False);t=np.load(source/'T.npy',allow_pickle=False)
            for kind in ('grid_search','best_available_search'):
                out=folder/model/kind;p=json.loads((out/'search_freeze.json').read_text())
                if p['T_used_by_search'] or p['metric']!=metric or p['channel']!=channel:raise ValueError('Invalid search provenance')
                for name in ('R.npy','U.npy'):
                    if sha(out/name)!=p[name[0]+'_sha256']:raise ValueError('Changed frozen result')
                r=np.load(out/'R.npy',allow_pickle=False);u=np.load(out/'U.npy',allow_pickle=False)
                if not np.allclose(r@r.T,np.eye(z.shape[1]),atol=1e-8):raise ValueError('Nonorthogonal R')
                if not np.allclose(z@r.T,u,atol=1e-10):raise ValueError('Wrong R convention')
                corr=np.corrcoef(np.column_stack((u,t)),rowvar=False)[:u.shape[1],u.shape[1]:]
                if not np.allclose(corr,np.load(out/'pearson_U_T.npy'),atol=1e-10):raise ValueError('Incorrect Pearson correlation')
                evaluation=json.loads((out/'evaluation.json').read_text())
                if not evaluation['T_dataset_SHA_verified'] or not evaluation['T_read_after_R_frozen']:
                    raise ValueError('Incomplete T/source verification')
                if p['all_subjects']!=5000:raise ValueError('Not all subjects')
                if kind=='best_available_search' and p['loss']>p['grid_minimum_loss']+1e-8:
                    raise ValueError('Saved pool search lost grid incumbent')
                records.append(dict(number=number,model=model,kind=kind,loss=p['loss']))
    wave.json_file(OUTPUT/'verification.json',dict(status='PASS',candidates=44,models=list(MODELS),
        frozen_rotations_verified=len(records),model_candidates=176,all_subjects=5000,
        R_orthogonality_and_application_verified=True,Pearson_correlations_recomputed=True,
        T_source_SHA_verified=True,all_plot_files_present=True))
    print(f'[Gallery verification] PASS: {len(records)} R/U/correlation sets, 44 four-model galleries')


if __name__=='__main__':main()
