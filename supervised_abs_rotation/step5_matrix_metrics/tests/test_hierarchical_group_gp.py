import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import group_sparse_gp as base
import hierarchical_group_gp as hierarchy


def test_W_only_hierarchy_preserves_complete_function_span_and_removes_lower_orders():
    rng=np.random.default_rng(781);z=rng.normal(size=(180,4));radial=np.exp(-.5*np.sum((z[:,None]-z[None,:8])**2,axis=-1))
    with threadpool_limits(2):
        x,groups,masks,_=base.dictionary(z,3,radial)
        h,_,_,audit=hierarchy.dictionary(z,3,radial,device='cpu')
        coefficient=np.linalg.lstsq(h,x,rcond=1e-12)[0]
    np.testing.assert_allclose(h@coefficient,x,atol=1e-8)
    order=np.array([int(mask).bit_count() for mask in masks])[groups]
    for level in np.unique(order):
        lower=x[:,order<level];current=h[:,order==level]
        if lower.shape[1]:
            assert np.linalg.norm(lower.T@current)/(np.linalg.norm(lower)*np.linalg.norm(current))<1e-8
    assert audit['projection_inputs']=='W only, all subjects'


def test_hierarchy_train_and_full_features_only_change_group_scales_and_intercepts():
    rng=np.random.default_rng(782);z=rng.normal(size=(120,4));radial=np.exp(-.5*np.sum((z[:,None]-z[None,:8])**2,axis=-1))
    with threadpool_limits(2):
        a,groups,_,ta=hierarchy.dictionary(z,3,radial,reference=np.arange(90),device='cpu')
        b,_,_,fa=hierarchy.dictionary(z,3,radial,device='cpu')
    ratio=np.array(ta['group_scales'])[groups]/np.array(fa['group_scales'])[groups]
    for group in np.unique(groups):
        np.testing.assert_allclose(np.mean(np.sum(a[:90,groups==group]**2,axis=1)),1.,atol=1e-12)
    shifted=a*ratio
    shifted-=shifted.mean(0)
    np.testing.assert_allclose(shifted,b,atol=1e-10)
