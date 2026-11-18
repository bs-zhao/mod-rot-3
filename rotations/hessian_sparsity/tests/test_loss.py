"""No roles, no teacher parameter labels, no analytic CPT used to fit R."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from rotations.hessian_sparsity.loss import summarize_fields,HessianLoss
from rotations.hessian_sparsity.optimizer import pairs,givens,optimize
from rotations.hessian_sparsity.planes import plane_rotate


def so(d,seed):
    rng=np.random.default_rng(seed)
    M,_=np.linalg.qr(rng.normal(size=(d,d)))
    if np.linalg.det(M)<0:M[0]*=-1
    return M


def synthetic(d,seed=42,n=8,t=20):
    rng=np.random.default_rng(seed)
    Q=so(d,seed+4)
    diag=rng.normal(size=(n,t,d))*np.linspace(.3,1.7,d)
    H0=np.zeros((n,t,d,d))
    for i in range(d):H0[...,i,i]=diag[...,i]
    J0=np.zeros((n,t,d,6))
    for j in range(6):
        J0[...,j%d,j]=rng.normal(size=(n,t))
    # Original q coordinates were rotated by Q.T.
    H=np.einsum("ai,...ab,bj->...ij",Q.T,H0,Q,optimize=True)
    J=np.einsum("ia,...aj->...ij",Q.T,J0,optimize=True)
    return dict(eta=np.zeros((n,t)),UU=H,UX=J,
                group_codes=np.arange(t)%4),Q


def test_so3_so4_givens_are_proper_rotations():
    for d in (3,4):
        x=torch.linspace(-.9,1.1,len(pairs(d)),dtype=torch.float64,
                         requires_grad=True)
        R=givens(x,d)
        assert np.allclose((R@R.T).detach().numpy(),np.eye(d),atol=1e-12)
        assert np.isclose(float(torch.det(R).detach()),1,atol=1e-12)
        loss=(R*torch.arange(d*d,dtype=R.dtype).reshape(d,d)).sum()
        assert torch.isfinite(torch.autograd.grad(loss,x)[0]).all()


def test_true_sparse_basis_better_than_random_mixing_and_90deg_invariance():
    for d in (3,4):
        fields,Q=synthetic(d)
        groups=summarize_fields(fields,grouping="pooled",include_ux=True)
        for mode in ("joint","uu_only","ux_only"):
            objective=HessianLoss(groups,mode=mode)
            target=objective.value(Q)
            original=objective.value(np.eye(d))
            assert target<original-1e-3,(d,mode,target,original)
            for i in range(d):
                for j in range(i+1,d):
                    assert np.isclose(
                        target,objective.value(plane_rotate(Q,i,j,90)),atol=1e-9)


def test_balanced_condition_loss_is_mean_of_same_universal_function():
    fields,_=synthetic(3,t=28)
    all_groups=summarize_fields(fields,grouping="balanced",include_ux=True)
    assert len(all_groups)==4
    objective=HessianLoss(all_groups,mode="joint")
    R=so(3,11)
    metric=objective.scalar(R)
    assert np.isclose(metric["total"],np.mean(
        [part["loss"] for part in metric["by_group"]]),atol=1e-12)
    unstratified=HessianLoss(summarize_fields(fields,"pooled"),mode="joint")
    assert np.isfinite(unstratified.value(R))


def test_hessian_loss_autograd_gradient_matches_finite_differences():
    fields,_=synthetic(3,n=5,t=11)
    loss=HessianLoss(summarize_fields(fields),mode="joint")
    x=torch.tensor([.27,-.31,.43],dtype=torch.float64,requires_grad=True)
    val=loss.torch_terms(givens(x,3))[0]
    analytic=torch.autograd.grad(val,x)[0].detach().numpy()
    base=x.detach().numpy()
    step=1e-5
    finite=np.array([(loss.value(givens(torch.tensor(base+np.eye(3)[i]*step),
                                        3).numpy())-
                      loss.value(givens(torch.tensor(base-np.eye(3)[i]*step),
                                        3).numpy()))/(2*step) for i in range(3)])
    np.testing.assert_allclose(analytic,finite,rtol=1e-4,atol=2e-5)


def test_multistart_fits_only_on_fit_not_validation():
    fit_fields,Q=synthetic(3,seed=7)
    valid_fields,_=synthetic(3,seed=8)
    fit=HessianLoss(summarize_fields(fit_fields),mode="uu_only")
    valid=HessianLoss(summarize_fields(valid_fields),mode="uu_only")
    R,report=optimize(fit,valid,dict(seed=7,restarts=3,maxiter=70))
    assert report["number_of_angles"]==3
    assert report["best_fit_start"]==min(
        report["all_starts"],key=lambda t:t["fit"])["start"]
    assert fit.value(R)<=fit.value(np.eye(3))+1e-8
    assert np.allclose(R@R.T,np.eye(3),atol=1e-9)
    assert np.isclose(np.linalg.det(R),1,atol=1e-9)


def test_all_ux_columns_invariant_under_orthogonal_u_rotation():
    fields,Q=synthetic(4)
    J=fields["UX"]
    rotated=np.einsum("ia,staj->stij",Q,J,optimize=True)
    np.testing.assert_allclose(
        np.sum(np.square(rotated),axis=(0,1,2)),
        np.sum(np.square(J),axis=(0,1,2)),rtol=1e-12,atol=1e-12)
