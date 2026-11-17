"""Run: python -m pytest experiments/derivative_fidelity/tests -q"""
import numpy as np
import torch

from experiments.derivative_fidelity.derivatives import derivative_function, tensor_symmetry_error, chain_rule_sigmoid
from experiments.derivative_fidelity.metrics import block_metrics, assemble_energy, field_accuracy


def f(u,x):
    return u[0] * x[0] + u[0]*u[1]*x[1] + 2*x[0]**3


def test_exact_derivative_blocks():
    u=torch.tensor([2.,3.],dtype=torch.float64)
    x=torch.tensor([4.,5.],dtype=torch.float64)
    assert np.allclose(derivative_function(f,"U")(u,x).numpy(),[19,10])
    assert np.allclose(derivative_function(f,"UX")(u,x).numpy(),[[1,3],[0,2]])
    assert np.allclose(derivative_function(f,"UU")(u,x).numpy(),[[0,5],[5,0]])
    assert abs(derivative_function(f,"XXX")(u,x).numpy()[0,0,0]-12)<1e-9
    assert abs(derivative_function(f,"UUX")(u,x).numpy()[0,1,1]-1)<1e-9


def test_sigmoid_third_rule():
    z=np.random.default_rng(8).normal(size=(2,3))
    g=np.ones((2,3,2))
    H=np.zeros((2,3,2,2))
    T=np.zeros((2,3,2,2,2))
    gp,hp,tp=chain_rule_sigmoid(z,g,H,T)
    q=1/(1+np.exp(-z))
    assert np.allclose(gp[...,0],q*(1-q))
    assert np.allclose(tp[...,0,1,1],q*(1-q)*(1-6*q+6*q*q))


def test_metrics_nonzero_and_fullw_count():
    a=np.ones((2,3,2,2))
    m=block_metrics(a,"UU")
    assert m["C03.frob_energy"]==4
    assert m["C12.diag_share"]==.5
    fields={"U":np.ones((2,3,2)),"X":np.ones((2,3,1)),
            "UU":a,"UX":np.ones((2,3,2,1)),"XX":np.ones((2,3,1,1))}
    r=assemble_energy(fields)
    assert r["E.order2.fullW_energy"]==4+2*2+1
    assert field_accuracy(a,a)["rmse"]==0
