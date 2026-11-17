import numpy as np
import torch
from experiments.derivative_fidelity.audits import audit_derivatives,audit_probability_chain

def f(u,x):
    return u[0]*x[0]+u[0]*u[1]*x[1]+x[0]**3

def test_derivative_and_chain_audit():
    U=np.array([[.2,.4],[.5,.6]])
    X=np.array([[.1,.3],[.2,.4]])
    report=audit_derivatives(f,U,X,points=2)
    assert report["max_errors"]["symmetry_H"]<1e-10
    assert report["max_errors"]["symmetry_T"]<1e-10
    assert report["max_errors"]["H_fd_gradient"]<1e-5
    assert report["max_errors"]["T_fd_hessian"]<1e-5
    p=lambda u,x:torch.sigmoid(f(u,x))
    chain=audit_probability_chain(f,p,U,X,points=2)
    assert chain["max_chain_errors"]["third"]<1e-9
