import csv
import sys
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matrix_function_profiles as profiles
import run_conditional_kernel_moments as kernel
import kernel_paired_subject_stability as stability
from test_matrix_metrics import make_cases


def test_subject_refit_matches_frozen_full_sample_and_axis_symmetry(tmp_path):
    source=tmp_path/'source';make_cases(source)
    folder,original=tmp_path/'bank',tmp_path/'original'
    torch.set_num_threads(2)
    with threadpool_limits(2):
        profiles.prepare(source,folder)
        kernel.analyze(source,original,folder,centers=8,device='cpu')
        bank=profiles.MatrixBank(source,folder)
        with np.load(original/'W_only_geometry.npz',allow_pickle=False) as saved:
            q=saved['directions'][saved['membership'][0]].T
        operators=[np.load(original/f'X_witness_column{j}.npy',allow_pickle=False) for j in range(bank.data.dx)]
        rows,audit=stability.frame_scores(bank,np.arange(bank.data.n),
            [q,q[:,[2,0,3,1]]*np.array([-1.,1.,-1.,1.])],operators,centers=8,device='cpu')
    with (original/'blind_scores.csv').open() as stream:
        expected={r['metric']:float(r['loss']) for r in csv.DictReader(stream)
            if r['channel']=='logit_nn' and r['anonymous_case']=='case_000'}
    assert len(rows)==2*len(expected) and audit['subjects']==32
    for r in rows:np.testing.assert_allclose(r['loss'],expected[r['metric']],rtol=1e-8,atol=1e-10)
