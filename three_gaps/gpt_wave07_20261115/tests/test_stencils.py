import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'exp21'))
import core as c
from stencils import Field
from scales import chord_design
import numpy as np
import torch

def test_quadratic():
    field=Field.__new__(Field);field.d=2;field.p=2
    field.C=torch.tensor([[2.,.4],[.4,1.]],dtype=torch.float64)
    field.V=torch.tensor([[.2,.1],[-.4,.7]],dtype=torch.float64)
    field.X=torch.tensor([[.3,-.2],[.1,.8]],dtype=torch.float64)
    def f(v,x):return v[...,0]**2+3*v[...,0]*v[...,1]+v[...,0]*x[...,0]+2*v[...,1]*x[...,1]
    field.f=f;field.scalar=f
    for theta in [0,.3,.8]:
        R=torch.tensor([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]],dtype=torch.float64)
        u0,t0=field.energies(R,0,'autograd')
        for h in [.1,.25,.5]:
            u,t=field.energies(R,h,'ambient',np.array([.2,.3]))
            torch.testing.assert_close(u,u0,atol=1e-12,rtol=1e-12)
            torch.testing.assert_close(t,t0,atol=1e-12,rtol=1e-12)
    try:chord_design(np.array([[0.,0],[1.,1],[2.,2],[3.,3]]),k=2)
    except ValueError:pass
    else:raise AssertionError('rank-deficient chord must be rejected')

def test_high_frequency():
    eps=.01;omega=20.;x=.3
    def f(t):return t*t+eps*np.sin(omega*t)
    for h in [.01,.05,.1,.25,.5,1.]:
        D=(f(x+h)-2*f(x)+f(x-h))/h**2
        exact=2-4*eps*np.sin(omega*x)*np.sin(omega*h/2)**2/h**2
        np.testing.assert_allclose(D,exact,atol=1e-11)
        assert abs(D-2)<=4*eps/h**2+1e-11

if __name__=='__main__':test_quadratic();test_high_frequency();print('FD quadratic / bilinear / R-dependent steps / high-frequency bound / chord rank tests PASS')
