import numpy as np
import torch
import math
from experiments.derivative_fidelity.estimators import value_difference,hybrid_ad_fd,stencil_weights,fit_value_surrogate


def fun(u,x):
    return u[0]*x[0]+u[0]*u[1]*x[1]+2*x[0]**3


def test_stencils():
    for n in [1,2,3]:
        nodes,weights=stencil_weights(n,3,6,"fd")
        assert abs(sum(w*x**n for x,w in zip(nodes,weights))-math.factorial(n))<1e-9


def test_finite_difference_all_orders():
    u=torch.tensor([2.,3.],dtype=torch.float64)
    x=torch.tensor([4.,5.],dtype=torch.float64)
    for kind in ("fd2","fd4","local_poly","spectral_chebyshev"):
        f=value_difference(fun,"UX",2,2,.05,kind)
        assert np.allclose(f(u,x).numpy(),[[1,3],[0,2]],atol=1e-6),kind
    g=hybrid_ad_fd(fun,"UUX",2,2,.01)
    assert abs(g(u,x).detach().numpy()[0,1,1]-1)<1e-6


def test_value_only_surrogate():
    rng=np.random.default_rng(6)
    U=rng.normal(size=(100,2))
    X=rng.normal(size=(100,2))
    s,meta=fit_value_surrogate(fun,U,X,kind="krr",n_train=200,n_centers=40,seed=6)
    assert meta["validation_mse"]>=0
    assert torch.isfinite(s(torch.tensor([.2,.3],dtype=torch.float64),
                            torch.tensor([.2,.1],dtype=torch.float64))).item()
