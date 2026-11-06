import numpy as np
from .method import matrices, loss, from_angles, optimize
from .toy_audit import exact_linear_contrast


def test_recovery_on_distinct_diagonal_contexts():
    rng = np.random.default_rng(9)
    a = from_angles([.4,-.3,.2],3)
    m = np.array([np.diag(row) for row in rng.normal(size=(12,3))])
    mixed = np.einsum('ia,xab,jb->xij',a,m,a)
    r, _, _ = optimize(mixed, starts=3)
    assert loss(r,mixed) < 1e-10
    assert loss(np.eye(3),mixed) > .01
    assert np.max(np.abs(r@r.T-np.eye(3))) < 1e-12


def test_common_covariance_removed_and_rotation_equivariant():
    rng = np.random.default_rng(3)
    g = rng.normal(size=(80,3,10))
    a = from_angles([.2,.3,-.1],3)
    m = matrices(g)
    assert np.max(np.abs(m.mean(0))) < 1e-14
    rotated = matrices(np.einsum('ia,nax->nix',a,g))
    assert np.allclose(rotated,np.einsum('ia,xab,jb->xij',a,m,a))
    assert np.isclose(np.sum(rotated**2),np.sum(m**2))


def test_bilinear_counterexample_is_unidentifiable():
    # f=t1*t2*x; isotropic independent t. Covariances are scalar identity.
    # No covariance-JD method can identify its coordinate axes.
    m = np.array([np.eye(2)*v for v in [1.,2.,3.]])
    assert loss(np.eye(2),m) == 0
    assert abs(loss(from_angles([np.pi/4],2),m)) < 1e-14


def test_exact_linear_decoder_prefers_mixed_axes():
    h=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],float)/2
    true_contrast,true_pooled=exact_linear_contrast(np.eye(4))
    mixed_contrast,mixed_pooled=exact_linear_contrast(h)
    assert np.isclose(true_contrast,5/6)
    assert np.isclose(mixed_contrast,13/24)
    assert mixed_contrast < true_contrast
    assert mixed_pooled < true_pooled


def test_weight_audit_reconstructs_historical_primitives():
    from types import SimpleNamespace
    from .weight_audit import primitives
    from rotations.mlp_teacher_rotation.metrics import fingerprint_and_drift, contextual_jointdiag
    rng=np.random.default_rng(77);g=rng.normal(size=(40,3,7))
    a=from_angles([.2,-.4,.1],3)
    proxy=SimpleNamespace(gradients=lambda q:g)
    j,f=primitives(g,a)
    q=np.zeros((len(g),3))
    assert np.isclose(j,contextual_jointdiag(proxy,a,q),atol=1e-12)
    assert np.isclose(f,fingerprint_and_drift(proxy,a,q)[0],atol=1e-12)
