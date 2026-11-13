import sys
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nested_polynomial_variance import NestedPolynomialVariance


def test_nested_covariance_increments_are_PSD_and_conserve_observed_response_covariance(tmp_path):
    rng=np.random.default_rng(940);z=rng.normal(size=(110,3));z-=z.mean(0)
    y=rng.normal(size=(110,7));y-=y.mean(0)
    with threadpool_limits(2):
        torch.set_num_threads(2)
        model=NestedPolynomialVariance(z,torch.tensor(y),tmp_path,'test',device='cpu',degrees=(2,3))
        q=np.linalg.qr(rng.normal(size=(3,3)))[0]
        effects,audit=model.frame_covariances(q)
        assert torch.linalg.eigvalsh(effects).min()>-1e-10
        np.testing.assert_allclose(effects.sum(0),y.T@y/len(y),atol=1e-12)
        perm=[2,0,1];other,_=model.frame_covariances(q[:,perm]*np.array([-1,1,-1]))
        np.testing.assert_allclose(other,np.concatenate((effects[perm],effects[-1:])),atol=1e-11)
        assert audit['relative_squared_covariance_conservation_error']<1e-25


def test_full_kernel_degree_scales_and_regularization_are_rotation_equivariant(tmp_path):
    rng=np.random.default_rng(941);z=rng.normal(size=(100,3));z-=z.mean(0)
    y=np.column_stack((np.sin(z[:,0]*z[:,1]),z[:,2]**3,rng.normal(size=100)));y-=y.mean(0)
    q=np.linalg.qr(rng.normal(size=(3,3)))[0]
    with threadpool_limits(2):
        first=NestedPolynomialVariance(z,torch.tensor(y),tmp_path/'one','test',device='cpu',degrees=(2,3))
        rotated=NestedPolynomialVariance(z@q,torch.tensor(y),tmp_path/'two','test',device='cpu',degrees=(2,3))
        assert first.degree==rotated.degree
        np.testing.assert_allclose(first.ridge,rotated.ridge,rtol=1e-12)
        np.testing.assert_allclose(first.full,rotated.full,atol=1e-11)
        design=first.design(q,first.degree)
        original=first.design(np.eye(3),first.degree)
        np.testing.assert_allclose(design@design.T,original@original.T,atol=1e-12)
        partial=first.covariance(q[:,:2]);same=rotated.covariance(np.eye(3)[:,:2])
        np.testing.assert_allclose(partial,same,atol=1e-10)
