import json
import sys
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matrix_function_profiles as profiles
import run_conditional_kernel_moments as kernel
from conditional_kernel_blind import Scorer
from test_matrix_metrics import make_cases


def test_constant_X_component_preserves_pure_U_signal_removed_by_UX_centering():
    rng=np.random.default_rng(791);u=rng.normal(size=(30,2));x=np.linspace(-1,1,12)
    responses=u[:,0,None,None]+x[None,None,:]**2
    f=responses-responses.mean(0,keepdims=True)-responses.mean(2,keepdims=True)+responses.mean((0,2),keepdims=True)
    np.testing.assert_allclose(f,0.,atol=1e-15)
    tasks,targets,tail,audit=profiles.append_main_modes(responses,[],[],np.empty((0,30)),[],1)
    np.testing.assert_allclose(np.mean(targets[0]**2),1.,atol=1e-12)
    assert tasks==[(0,1)] and tail.sum()==0.
    np.testing.assert_allclose(targets[0][:,0],(u[:,0]-u[:,0].mean())/u[:,0].std(),atol=1e-12)


def test_nonaffine_main_preserves_nonlinear_residual_and_is_invariant_to_U_rotation():
    rng=np.random.default_rng(792);u=rng.normal(size=(90,4));q=np.linalg.qr(rng.normal(size=(4,4)))[0]
    f=(2*u[:,0]+u[:,1]**2)[:,None,None]+np.linspace(-1,1,12)[None,None,:]
    _,a,_,audit=profiles.append_main_modes(f,[],[],np.empty((0,90)),[],1,affine_coordinates=u)
    _,b,_,_=profiles.append_main_modes(f,[],[],np.empty((0,90)),[],1,affine_coordinates=u@q)
    np.testing.assert_allclose(a[0],b[0],atol=1e-12)
    assert 0.<audit[0]['nonaffine_main_energy_fraction']<1.
    linear=(u@np.array([1.,2.,3.,4.]))[:,None,None]+np.linspace(-1,1,12)[None,None,:]
    _,zero,_,za=profiles.append_main_modes(linear,[],[],np.empty((0,90)),[],1,affine_coordinates=u)
    assert np.max(np.abs(zero[0]))==0. and za[0]['rank']==0


def test_complete_response_pipeline_marks_scope_and_freezes_extra_frame(tmp_path):
    source=tmp_path/'source';make_cases(source)
    folder,output=tmp_path/'bank',tmp_path/'out'
    torch.set_num_threads(2)
    with threadpool_limits(2):
        profiles.prepare(source,folder)
        rows=kernel.analyze(source,output,folder,centers=8,device='cpu',include_main=True,audit_frame=np.eye(4))
    assert rows and all(r['metric'].startswith('with_U_main_') for r in rows)
    assert max(r['endpoint_error'] for r in rows)<1e-12
    assert json.loads((output/'protocol.json').read_text())['include_pure_U_main_component'] is True
    assert (output/'frozen_counter_assessment.csv').exists()
    with threadpool_limits(2):
        scorer=Scorer(profiles.MatrixBank(source,folder),output,device='cpu')
        assert scorer.include_main and scorer.centers==8
        v,expected=scorer.exact_cache[0]
        np.testing.assert_allclose(scorer.direction(v,force=True),expected,atol=1e-12)
