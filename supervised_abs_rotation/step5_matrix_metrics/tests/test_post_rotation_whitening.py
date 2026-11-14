"""Verify unit covariance and rotation equivariance of post-rotation ZCA."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from export_whitened_rotation_gallery import zca_whiten


def test_zca_all_subject_covariance_and_explicit_application():
    rng=np.random.default_rng(36)
    x=rng.normal(size=(5000,4))@np.array([[2.,.4,0.,0.],[0.,1.,.8,0.],[0.,0.,.3,.2],[0.,0.,0.,3.]])+7
    white,mean,a,audit=zca_whiten(x)
    np.testing.assert_allclose(white,(x-mean)@a,atol=1e-12)
    np.testing.assert_allclose(a,a.T,atol=1e-12)
    np.testing.assert_allclose(white.T@white/len(x),np.eye(4),atol=1e-12)
    np.testing.assert_allclose(white.mean(0),0.,atol=1e-12)
    assert audit['rank']==4


def test_zca_preserves_rotation_equivariance_instead_of_choosing_PCA_axes():
    rng=np.random.default_rng(39);z=rng.normal(size=(5000,6))@rng.normal(size=(6,6))
    r=np.linalg.qr(rng.normal(size=(6,6)))[0]
    first,_,_,_=zca_whiten(z)
    second,_,_,_=zca_whiten(z@r.T)
    np.testing.assert_allclose(second,first@r.T,atol=1e-10)


def test_rank_deficient_U_uses_declared_projector_without_ridge():
    rng=np.random.default_rng(40);x=rng.normal(size=(5000,2));x=np.column_stack((x,x[:,0]+x[:,1]))
    white,_,_,audit=zca_whiten(x)
    covariance=white.T@white/len(white)
    np.testing.assert_allclose(covariance@covariance,covariance,atol=1e-12)
    assert audit['rank']==2 and audit['dimension']==3
