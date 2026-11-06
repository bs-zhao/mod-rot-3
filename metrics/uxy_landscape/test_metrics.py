"""Estimator, tensor-transform and no-refit contracts on independent synthetic data."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import block_diag
from .dependence import ucenter,double_center,xi,codec,nearest,cmi_knn,Dependence
from .regression import poly,poly_derivatives,symmetric_basis,Regressions


def test_u_centering_is_zero_diagonal_and_idempotent():
    x=np.random.default_rng(1).normal(size=(30,3));a=ucenter(cdist(x,x))
    np.testing.assert_allclose(np.diag(a),0,atol=1e-13)
    np.testing.assert_allclose(a.sum(0),0,atol=1e-12)
    np.testing.assert_allclose(ucenter(a),a,atol=1e-12)


def test_xi_handles_binary_response_ties_without_jitter():
    x=np.linspace(-1,1,400);y=(x>0).astype(float)
    assert xi(x,y)>.98
    shuffled=np.random.default_rng(2).permutation(y)
    assert abs(xi(x,shuffled))<.2


def test_codec_detects_added_variable_and_reports_zero_denominator():
    rng=np.random.default_rng(2);x=rng.normal(size=(500,1));z=rng.normal(size=(500,1))
    y=z[:,0];from scipy.stats import rankdata
    ranks=rankdata(y,method='max')[:,None]
    score=codec(ranks,nearest(cdist(x,x)**2),nearest(cdist(np.c_[x,z],np.c_[x,z])**2))[0]
    assert score>.65
    assert np.isnan(codec(np.ones((20,1)),np.roll(np.arange(20),1),np.roll(np.arange(20),2))[0])


def test_knn_cmi_detects_conditional_nonlinear_information():
    rng=np.random.default_rng(3);x=rng.normal(size=400);c=rng.normal(size=(400,1))
    y=x*x+.2*c[:,0]+rng.normal(0,.15,400)
    assert cmi_knn(x,y,c,5)>.3


def test_complete_polynomial_features_transform_orthogonally():
    rng=np.random.default_rng(4);u=rng.normal(size=(30,4));a=np.linalg.qr(rng.normal(size=(4,4)))[0]
    b=symmetric_basis(4);power=b.T@np.kron(a,a)@b
    t=block_diag(np.ones((1,1)),a,power)
    np.testing.assert_allclose(poly(u@a.T),poly(u)@t.T,atol=1e-12)
    np.testing.assert_allclose(t@t.T,np.eye(len(t)),atol=1e-12)


def test_polynomial_mixed_derivatives_match_finite_differences():
    rng=np.random.default_rng(5);u=rng.normal(size=(8,3));d,h=poly_derivatives(u);step=1e-4
    for i in range(3):
        du=np.eye(3)[i]*step
        np.testing.assert_allclose((poly(u+du)-poly(u-du))/(2*step),d[:,i],atol=1e-9)
        for j in range(3):
            dv=np.eye(3)[j]*step
            fd=(poly(u+du+dv)-poly(u+du-dv)-poly(u-du+dv)+poly(u-du-dv))/(4*step*step)
            np.testing.assert_allclose(fd,np.broadcast_to(h[i,j],fd.shape),atol=1e-7)


def test_regression_scan_transforms_same_function_and_preserves_controls():
    rng=np.random.default_rng(6);u=rng.normal(size=(100,3));x=rng.normal(size=(100,2))
    y=np.column_stack([u[:,0]*x[:,0]+u[:,1]**2,(u[:,0]>0).astype(float),u[:,0]+x[:,1],u[:,2]*x[:,1],rng.normal(size=100)])
    split=[(np.arange(50),np.arange(50,100)),(np.arange(50,100),np.arange(50))]
    cfg=dict(ridge=1.,spline_knots=3,seed=31,profile_x_points=6,module_floor=.02)
    fitted=Regressions(u,x,y,split,cfg);a=np.linalg.qr(rng.normal(size=(3,3)))[0]
    before,_=fitted.compute(np.eye(3),u);after,_=fitted.compute(a,u@a.T)
    for key in before:
        if key.endswith('_control'):np.testing.assert_allclose(before[key],after[key],atol=1e-10)
    b=symmetric_basis(3);t=block_diag(np.ones((1,1)),a,b.T@np.kron(a,a)@b)
    c=np.einsum('ip,pqy->iqy',t,fitted.coefficient)
    old=np.einsum('np,pqy,nq->ny',poly(u),fitted.coefficient,poly(x))
    new=np.einsum('np,pqy,nq->ny',poly(u@a.T),c,poly(x))
    np.testing.assert_allclose(old,new,atol=1e-10)


def test_dependence_full_distance_control_and_binary_cmi_status():
    rng=np.random.default_rng(7);u=rng.normal(size=(60,3));x=rng.normal(size=(60,2))
    y=np.column_stack([(u[:,0]>0).astype(float),(x[:,0]>0).astype(float),u[:,0]+x[:,0],u[:,1]**2,u[:,2]+rng.normal(size=60)])
    cfg=dict(seed=31,association_bins=4,knn_k=4,rdc_features=5)
    engine=Dependence(x,y,cfg);first,axis=engine.compute(u)
    a=np.linalg.qr(rng.normal(size=(3,3)))[0];second,_=engine.compute(u@a.T)
    np.testing.assert_allclose(first['full_U_dcor_squared_control'],second['full_U_dcor_squared_control'],atol=1e-12)
    assert np.isnan(axis['cmi_knn_given_X'][:,:2]).all()
    assert np.isfinite(axis['cmi_knn_given_X'][:,2:]).all()
