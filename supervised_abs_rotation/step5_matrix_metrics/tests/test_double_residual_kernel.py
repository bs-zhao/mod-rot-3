import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import conditional_double_residual as double
import conditional_ux
import matrix_function_profiles as profiles
import run_conditional_kernel_moments as kernel
import run_double_residual_kernel as runner
import run as wave
from test_matrix_metrics import make_cases


def test_tensor_conditional_regression_matches_numpy_with_unpenalized_affine():
    rng=np.random.default_rng(741);x=rng.normal(size=(64,12));x[:,0]=1.
    y=rng.normal(size=(64,9));train=np.arange(47);query=np.arange(47,64)
    expected=conditional_ux.regressions(x,y,train,query,4,(1e-4,.1))
    actual=double.regressions(torch.tensor(x),torch.tensor(y),train,query,4,(1e-4,.1))
    for a,b in zip(actual,expected):np.testing.assert_allclose(a.numpy(),b,atol=1e-11)


def test_response_crossfit_removes_predictable_other_U_function_without_removing_omitted_U():
    rng=np.random.default_rng(742);z=rng.normal(size=(180,4));v=np.array([1.,0.,0.,0.])
    y=np.column_stack((2*z[:,1],2*z[:,1]+z[:,2]**2,z[:,0]))
    with threadpool_limits(2):
        r,audit=double.residual_targets(z,v,torch.tensor(y),2.,degree=2,centers=8)
    error=np.mean(r.numpy()**2,axis=0)
    assert error[0]<1e-20 and error[1]<.01*np.var(y[:,1]) and error[2]>.5
    assert sum(r['heldout_subjects'] for r in audit['history'])==180


def test_full_double_residual_pipeline_freezes_grid_and_fixed_counter_frame(tmp_path):
    source=tmp_path/'source';make_cases(source)
    folder,original,counter=tmp_path/'bank',tmp_path/'original',tmp_path/'counter';counter.mkdir()
    with threadpool_limits(2):
        profiles.prepare(source,folder)
        kernel.analyze(source,original,folder,centers=8,device='cpu')
        with np.load(original/'W_only_geometry.npz',allow_pickle=False) as saved:
            q=saved['directions'][saved['membership'][0]].T
        np.savez_compressed(counter/'blind_rotations.npz',optimized=q)
        wave.json_file(counter/'blind_freeze.json',dict(rotations_sha256=
            hashlib.sha256((counter/'blind_rotations.npz').read_bytes()).hexdigest()))
        rows=runner.analyze(source,original,folder,tmp_path/'out',counterexample=counter,
                            capacities=((2,8),),centers=8,device='cpu')
    assert rows and max(r['endpoint_error'] for r in rows)<1e-12
    with (tmp_path/'out/frozen_counter_assessment.csv').open() as stream:assess=list(csv.DictReader(stream))
    with (tmp_path/'out/blind_scores.csv').open() as stream:
        originals={(r['channel'],r['metric']):float(r['loss']) for r in csv.DictReader(stream)
                   if r['anonymous_case']=='case_000'}
    for r in assess:np.testing.assert_allclose(float(r['loss']),originals[r['channel'],r['metric']],atol=1e-12)
