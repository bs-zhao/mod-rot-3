"""Analytic toy Hessian quadrature, independent SO(d) sparse basis checks."""
import numpy as np
import pytest
import torch
from rotations.gaussian_integrated_rotation.moments import (
    accumulate,save_moments,load_moments)
from rotations.gaussian_integrated_rotation.loss import GaussianHessianLoss
from rotations.gaussian_integrated_rotation.optimizer import (
    givens,pairs,optimize,plane_landscapes)
from rotations.gaussian_integrated_rotation.oracle import (
    procrustes,corr_oracle,corr)


class Toy(torch.nn.Module):
    def decode(self,x,z):
        y=(z[:,0,None]**2*x[...,0]+
           z[:,1,None]*z[:,2,None]*x[...,1])
        return torch.stack((torch.zeros_like(y),y),dim=-1)


def toy_data():
    x=np.array([[1.,2.,.5,0,0,.5],[3.,4.,.2,0,0,.8],
                [5.,6.,.5,0,0,.5],[7.,8.,.7,0,0,.3]])
    q=np.array([[0.,1.,2.],[2.,3.,0.],[-1.,2.,1.]])
    wq=np.array([.2,.3,.5])
    wx=np.array([.1,.2,.3,.4])
    source=dict(mu=np.zeros(3),B=np.eye(3),
                x_mu=np.zeros(6),x_sd=np.ones(6))
    return source,q,wq,x,wx


def test_integrated_UU_tensor_equals_analytic_weighted_hessian(tmp_path):
    source,q,wq,x,wx=toy_data()
    model=Toy()
    fields=accumulate(model,torch.device("cpu"),source,q,wq,x,wx,
                      groups=np.array([0,0,1,1]),grouping="pooled",
                      derivatives="uu_only",z_batch=2,x_batch=2)
    assert len(fields)==1
    H=np.zeros((len(x),3,3))
    H[:,0,0]=2*x[:,0];H[:,1,2]=x[:,1];H[:,2,1]=x[:,1]
    expected=np.einsum("t,tij,tkl->ijkl",wx,H,H)
    np.testing.assert_allclose(fields[0]["UU"],expected,rtol=1e-5,atol=1e-5)
    assert fields[0]["UX"] is None
    path=tmp_path/"moments.npz"
    save_moments(path,fields)
    loaded=load_moments(path)
    np.testing.assert_allclose(loaded[0]["UU"],expected)
    value=GaussianHessianLoss(loaded,mode="uu_only").value(np.eye(3))
    assert np.isfinite(value)


def test_balanced_mixture_has_same_per_group_formula_with_equal_aggregate():
    source,q,wq,x,wx=toy_data()
    gr=accumulate(Toy(),torch.device("cpu"),source,q,wq,x,wx,
                  np.array([0,0,1,1]),grouping="balanced",
                  derivatives="joint",z_batch=2,x_batch=2)
    assert len(gr)==2
    assert gr[0]["x_points"]==2 and gr[1]["x_points"]==2
    assert gr[0]["UX"].shape==(3,3,6)
    l=GaussianHessianLoss(gr,mode="joint")
    scored=l.scalar(np.eye(3))
    assert np.isclose(scored["total"],np.mean(
        [x["total"] for x in scored["per_group"]]))


@pytest.mark.parametrize("dim",[3,4])
def test_SO_d_sparsity_loss_and_plane_symmetry(dim):
    rng=np.random.default_rng(17+dim)
    H=np.zeros((50,dim,dim))
    for i in range(dim):H[:,i,i]=rng.normal(scale=i+1,size=50)
    Q,_=np.linalg.qr(rng.normal(size=(dim,dim)))
    if np.linalg.det(Q)<0:Q[:,0]*=-1
    rotated=np.einsum("ia,nab,jb->nij",Q,H,Q,optimize=True)
    C=np.einsum("nab,ncd->abcd",rotated,rotated)/len(rotated)
    norm=float(np.einsum("ijij->",C))
    field=[dict(name="all",x_points=50,probability_mass=1.,
                UU=C,UX=None,UU_norm=norm,UX_norm=None)]
    l=GaussianHessianLoss(field,mode="uu_only")
    oracle=Q.T
    assert l.value(oracle)<l.value(np.eye(dim))-1e-3
    for i in range(dim):
        for j in range(i+1,dim):
            G=np.eye(dim);G[i,i]=0;G[j,j]=0;G[i,j]=1;G[j,i]=-1
            assert np.isclose(l.value(G@oracle),l.value(oracle),atol=1e-9)
    R,report=optimize(l,l,dict(seed=23,restarts=3,maxiter=100))
    assert l.value(R)<=l.value(np.eye(dim))+1e-9
    assert np.isclose(np.linalg.det(R),1.,atol=1e-9)
    rec=plane_landscapes(R,l,l,count=7)
    assert len(rec)==len(pairs(dim))*7


def test_full_subject_oracle_is_separate_from_gaussian_loss():
    rng=np.random.default_rng(7)
    q=rng.normal(size=(320,3))
    T=q@np.array([[.8,.6,0],[-.6,.8,0],[0,0,1]]).T
    Rpro=procrustes(q,T)
    R,diag=corr_oracle(q,T,Rpro,restarts=2,maxiter=65,seed=23)
    assert diag["hessian_used_for_oracle"] is False
    assert diag["no_held_out_subjects"] is True
    assert np.mean(np.diag(corr(T,q@R.T))**2)>.999
