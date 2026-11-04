import numpy as np
from modrot3.analysis import ridge_probe, corr
from modrot3.rotation import skew_matrix, optimize, OBJECTIVES
from scipy.linalg import expm


def test_ridge_recovery_heldout():
    rng=np.random.default_rng(10)
    z=rng.normal(size=(150,4))
    theta=z@rng.normal(size=(4,4))
    r=ridge_probe(z,theta,np.arange(100),np.arange(100,125),np.arange(125,150))
    assert min(r['test_r2'])>.9999
    assert corr(theta,z).shape==(4,4)


def test_orthogonal_rotation_and_method_registry():
    rng=np.random.default_rng(13)
    z=rng.laplace(size=(90,4))
    R,info=optimize(z,None,None,None,'kurtosis',starts=1,maxiter=3,seed=13)
    assert np.allclose(R@R.T,np.eye(4),atol=1e-9)
    assert np.all(np.isfinite(R))
    assert set(OBJECTIVES)=={'identity','kurtosis','quartic'}
    assert np.allclose(skew_matrix(np.zeros(6),4),np.zeros((4,4)))