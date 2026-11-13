import sys
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quantile_conditional_variance import QuantileConditionalVariance,quantile_codes


def test_quantile_partitions_are_sign_equivalent_even_at_tied_boundaries():
    for values in (np.arange(10.),np.array([0,1,1,2,3,4,4,5.]),np.ones(8)):
        for bins in (2,3,4,5):
            first=quantile_codes(values,bins);other=quantile_codes(-values,bins)
            np.testing.assert_array_equal(first[:,None]==first[None],other[:,None]==other[None])


def test_coarsened_conditional_covariances_are_PSD_nested_and_handle_repeated_full_inputs(tmp_path):
    rng=np.random.default_rng(949);z=rng.normal(size=(100,3));z[1]=z[0]
    y=rng.normal(size=(100,5));y-=y.mean(0)
    with threadpool_limits(2):
        torch.set_num_threads(2)
        model=QuantileConditionalVariance(z,torch.tensor(y),tmp_path,'test',device='cpu')
        effects,audit=model.frame_covariances(np.eye(3))
        assert torch.linalg.eigvalsh(effects).min()>-1e-12
        assert float(torch.trace(effects[-1]))>0
        np.testing.assert_allclose(effects.sum(0),y.T@y/len(y),atol=1e-12)
        perm=[2,0,1];other,_=model.frame_covariances(np.eye(3)[:,perm]*np.array([-1,1,-1]))
        np.testing.assert_allclose(other,np.concatenate((effects[perm],effects[-1:])),atol=1e-12)
        assert audit['relative_squared_covariance_conservation_error']<1e-25


def test_fixed_partition_PRESS_matches_explicit_subject_deletion_and_singleton_fallback(tmp_path):
    rng=np.random.default_rng(950);z=rng.normal(size=(20,2));y=rng.normal(size=(20,3));y-=y.mean(0)
    labels=np.array([0]+[1]*2+[2]*5+[3]*12)
    with threadpool_limits(2):
        model=QuantileConditionalVariance(z,torch.tensor(y),tmp_path,'test',device='cpu')
        _,audit=model.partition_covariance(labels)
    residual=[]
    for i in range(len(y)):
        keep=(labels==labels[i]);keep[i]=False
        if not keep.any():keep=np.arange(len(y))!=i
        residual.append(y[i]-y[keep].mean(0))
    expected=np.sum(np.asarray(residual)**2)/np.sum(y**2)
    np.testing.assert_allclose(audit['fixed_partition_PRESS_relative_MSE'],expected,rtol=1e-14)
    assert audit['singleton_subject_fraction']==1/20
