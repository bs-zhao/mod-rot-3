import sys
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neighbor_conditional_variance import NeighborConditionalVariance


def test_all_anchor_neighbor_covariance_matches_explicit_sample_covariances(tmp_path):
    rng=np.random.default_rng(957);z=rng.normal(size=(80,3));y=rng.normal(size=(80,5));y-=y.mean(0)
    with threadpool_limits(2):
        torch.set_num_threads(2)
        model=NeighborConditionalVariance(z,torch.tensor(y),tmp_path,'test',device='cpu')
        basis=np.eye(3)[:,:2];actual=model.observed-model.covariance(basis)
        points=z@basis;distance=np.sum((points[:,None]-points[None])**2,axis=2)
        ids=np.argsort(distance,axis=1)[:,:3]
        expected=sum(np.cov(y[row].T,ddof=1) for row in ids)/len(y)
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        assert torch.linalg.eigvalsh(actual).min()>-1e-12
        effects,_=model.frame_covariances(np.eye(3))
        np.testing.assert_allclose(effects.sum(0),y.T@y/len(y),atol=1e-12)


def test_neighbor_conditioning_depends_on_subspace_and_not_its_choice_of_basis(tmp_path):
    rng=np.random.default_rng(958);z=rng.normal(size=(70,3));y=rng.normal(size=(70,4));y-=y.mean(0)
    basis=np.linalg.qr(rng.normal(size=(3,2)))[0];rotation=np.linalg.qr(rng.normal(size=(2,2)))[0]
    with threadpool_limits(2):
        a=NeighborConditionalVariance(z,torch.tensor(y),tmp_path/'a','test',device='cpu')
        b=NeighborConditionalVariance(z,torch.tensor(y),tmp_path/'b','test',device='cpu')
        np.testing.assert_allclose(a.covariance(basis),b.covariance(basis@rotation),atol=1e-12)


def test_local_affine_residual_covariance_matches_explicit_local_regressions(tmp_path):
    rng=np.random.default_rng(960);z=rng.normal(size=(80,3));y=rng.normal(size=(80,4));y-=y.mean(0)
    with threadpool_limits(2):
        model=NeighborConditionalVariance(z,torch.tensor(y),tmp_path,'test',device='cpu',neighbors=12,local_linear=True,stable_ties=True)
        basis=np.eye(3)[:,:2];actual=model.observed-model.covariance(basis)
        points=z@basis;distance=np.sum((points[:,None]-points[None])**2,axis=2)
        ids=np.argsort(distance,axis=1)[:,:12];expected=np.zeros((4,4))
        for i,row in enumerate(ids):
            design=np.column_stack((np.ones(12),points[row]-points[i]))
            coefficients=np.linalg.lstsq(design,y[row],rcond=1e-10)[0]
            residual=y[row]-design@coefficients
            expected+=residual.T@residual/(12-np.linalg.matrix_rank(design))/len(y)
        np.testing.assert_allclose(actual,expected,atol=1e-12)
    from run_neighbor_conditional_variance import parse_arguments
    args=parse_arguments(['--neighbors','32','--local-linear','--stable-ties','--run-tag','CK'])
    assert args.local_linear and args.stable_ties and args.neighbors==32


def test_seeded_boundary_ties_are_reproducible_under_sign_changes(tmp_path):
    rng=np.random.default_rng(961);z=rng.normal(size=(60,3));z[:,0]=np.repeat([-1.,0.,1.],20)
    y=rng.normal(size=(60,4));y-=y.mean(0);basis=np.eye(3)[:,:1]
    with threadpool_limits(2):
        a=NeighborConditionalVariance(z,torch.tensor(y),tmp_path/'a','test',device='cpu',stable_ties=True)
        b=NeighborConditionalVariance(z,torch.tensor(y),tmp_path/'b','test',device='cpu',stable_ties=True)
        np.testing.assert_allclose(a.covariance(basis),b.covariance(-basis),atol=1e-12)
        assert a.audits[-1]['boundary_distance_tie_anchor_fraction']==1
