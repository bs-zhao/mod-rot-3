"""Meaningful mathematical contracts, independent of trained-model artifacts."""
import numpy as np
import torch
from .optimization import RowObjective, jvol_score, rotation, optimize
from .entropy_reference import entropy_and_coverage
from .fields import fisher_rows, representation
from .shape_reference import curvature_refinement_rotation


def test_binary_fisher_equals_normalized_score_geometry():
    rng=np.random.default_rng(1);g=rng.normal(size=(25,4));p=rng.uniform(.01,.99,25)
    a=np.linalg.qr(rng.normal(size=(4,4)))[0]
    rows,w,_=fisher_rows(dict(logit=g,p=p),np.full(25,1/25),1e-10)
    fisher=p[:,None,None]*(1-p[:,None,None])*g[:,:,None]*g[:,None,:]
    fisher/=np.trace(fisher,axis1=-2,axis2=-1)[:,None,None]
    diagonal=np.diagonal(a[None]@fisher@a.T[None],axis1=-2,axis2=-1)
    np.testing.assert_allclose(diagonal,(rows@a.T)**2,atol=1e-13)
    eps=.01
    expected=np.sum(w*np.exp(np.log(diagonal+eps).mean(-1)))
    actual=float(RowObjective([rows],eps=eps,weights=[w])(torch.tensor(a)))
    np.testing.assert_allclose(actual,expected,atol=1e-13)


def test_entropy_and_coverage_port_matches_reference():
    rng=np.random.default_rng(2);b=rng.normal(size=(30,4));w=rng.uniform(.1,1,30);w/=w.sum()
    a=np.linalg.qr(rng.normal(size=(4,4)))[0]
    ent,cov,*_=entropy_and_coverage(b,w,a,.3)
    score=RowObjective([b],kind='entropy',weights=[w],coverage_lambda=.1,coverage_tau=.3)
    np.testing.assert_allclose(float(score(torch.tensor(a))),ent+.1*cov,rtol=1e-12)


def test_jvol_scale_invariance_and_full_residual_inverse():
    rng=np.random.default_rng(3);rows=torch.tensor(rng.normal(size=(80,4)))
    w=torch.tensor(rng.normal(size=(4,4))+3*np.eye(4))
    np.testing.assert_allclose(float(jvol_score(w,rows)),float(jvol_score(7*w,rows)),atol=1e-12)
    basis=np.linalg.qr(rng.normal(size=(8,4)))[0];z=rng.normal(size=(20,8));mu=rng.normal(size=8)
    q=(z-mu)@basis;residual=z-mu-q@basis.T
    a=np.linalg.inv(w.numpy()).T;u=q@a.T
    back=mu+(u@np.linalg.inv(a).T)@basis.T+residual
    np.testing.assert_allclose(back,z,atol=1e-12)


def test_row_simple_structure_recovers_mixed_axes_without_theta():
    angle=.4;r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    signatures=np.tile(np.eye(2),(10,1))@r
    score=RowObjective([signatures],kind='entropy')
    a,fit,_=optimize(score,2,dict(starts=2,seed=31,maxiter=100))
    assert fit['after'] < 1e-8
    assert np.min(np.max(np.abs(signatures[:2]@a.T),axis=1)) > .99999


def test_curvature_only_rotates_detected_shared_group():
    angle=.45;r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    c=np.zeros((4,4));c[1:3,1:3]=r@np.diag([0.,5.])@r.T
    fit={'per_x':{0:{'curvature':c}}}
    s,info,error=curvature_refinement_rotation(4,{0:[1,2]},fit)
    np.testing.assert_allclose((s@c@s.T)[1:3,1:3],np.diag([0.,5.]),atol=1e-12)
    np.testing.assert_allclose(s[[0,3]],np.eye(4)[[0,3]],atol=1e-12)
    assert error<1e-12
    noop,_,_=curvature_refinement_rotation(4,{},fit)
    np.testing.assert_array_equal(noop,np.eye(4))


def test_rotation_objective_autodiff_matches_finite_difference():
    rng=np.random.default_rng(7);score=RowObjective([rng.normal(size=(50,4))],eps=.01)
    v=torch.tensor(rng.normal(0,.2,6),requires_grad=True)
    value=score(rotation(v,4));value.backward()
    step=1e-6;fd=[]
    for i in range(6):
        d=np.zeros(6);d[i]=step
        fd.append((float(score(rotation(v.detach()+torch.tensor(d),4)))-float(score(rotation(v.detach()-torch.tensor(d),4))))/(2*step))
    np.testing.assert_allclose(v.grad.numpy(),fd,rtol=1e-5,atol=1e-8)


def test_raw_coordinate_selection_and_scaling_use_train_only():
    rng=np.random.default_rng(19);z=rng.normal(size=(23,8))*.01
    z[:17,2]*=400;z[:17,5]*=200
    z[17:,0]*=1000000  # TEST variance must not change selected axes or TRAIN scales.
    tr=np.arange(17)
    s=representation(dict(z=z,tr=tr), 'raw_top_standardized',2,1e-5)
    assert set(s['representation']['selected_indices'])=={2,5}
    assert s['q'].shape==(23,2)
    np.testing.assert_allclose(s['q'][tr].mean(0),0.,atol=1e-12)
    np.testing.assert_allclose(s['q'][tr].std(0),1.,atol=1e-12)
    np.testing.assert_allclose(s['W']@s['B'],np.eye(2),atol=1e-12)
