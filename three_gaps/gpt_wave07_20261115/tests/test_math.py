import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as c
import nodes
import numpy as np
import torch
from torch.func import jacfwd,jacrev

def test_gaussian():
    for d in (4,6):
        e,w,_=nodes.gaussian(0,d,1,'gh',order=3)
        assert abs(w.sum()-1)<1e-12
        np.testing.assert_allclose(w@e,0,atol=1e-12)
        np.testing.assert_allclose(np.einsum('n,na,nb->ab',w,e,e),np.eye(d),atol=1e-12)
        np.testing.assert_allclose(w@(e**4),3,atol=1e-12)
    C=np.array([[2.,.7],[.7,1.]])
    e,w,_=nodes.gaussian(0,2,1,'gh',order=3);v=nodes.physical(e,C,np.zeros(2))
    np.testing.assert_allclose(np.einsum('n,na,nb->ab',w,v,v),C,atol=1e-12)
    for method in ('mc','qmc'):
        e,w,_=nodes.gaussian(65536,4,23,method)
        np.testing.assert_allclose(w@e,0,atol=.02)
        np.testing.assert_allclose(w@(e**2),1,atol=.025)
        np.testing.assert_allclose(w@(e**4),3,atol=.12)
    e,_,_=nodes.gaussian(256,4,23)
    np.testing.assert_array_equal(e,nodes.gaussian(512,4,23)[0][:256])
    a=nodes.antithetic(32,4,23);np.testing.assert_array_equal(a[:16],nodes.antithetic(16,4,23))
    np.testing.assert_allclose(a.mean(0),0,atol=1e-16)
    v=nodes.physical(e,np.eye(4),np.arange(4.),'P3');np.testing.assert_allclose(v+np.arange(4.),e)
    Q=c.haar_set(4,1,17)[0].astype(float);Q=np.linalg.qr(Q)[0];mu=np.arange(4.)
    v=nodes.physical(e,np.eye(4),mu,'P3',basis=Q)
    np.testing.assert_allclose(v@Q.T+mu,e@Q.T,atol=1e-12)
    w,rate=nodes.truncated(e,np.ones(len(e))/len(e));assert abs(w.sum()-1)<1e-12 and 0<rate<.15

def test_contraction():
    rng=np.random.default_rng(1);tt=torch.tensor(rng.normal(size=(7,5,4,4)),dtype=torch.float64)
    tt=(tt+tt.transpose(-1,-2))/2;tx=torch.tensor(rng.normal(size=(7,5,4,6)),dtype=torch.float64)
    sx,t4=c.moments(tt,tx);C=c.cov(rng.normal(size=(50,4)));ml=c.Loss(sx.mean(0),t4.mean(0),C)
    R=torch.tensor(c.haar_set(4,3,4),dtype=torch.float32);ux,uu=ml.energies(R)
    for k,q in enumerate(R):
        A=c.effective_A(q,torch.tensor(C,dtype=torch.float32)).double()
        a=torch.einsum('ia,nkaj->nkij',A,tx).square().mean((0,1))
        b=torch.einsum('ia,nkab,jb->nkij',A,tt,A).square().mean((0,1))
        torch.testing.assert_close(ux[k].double(),a,atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(uu[k].double(),b,atol=2e-6,rtol=2e-6)
        D=torch.diag(torch.tensor([-10.,.1,2.,-3.]));torch.testing.assert_close(ml.total(D@q),ml.total(q),atol=2e-5,rtol=2e-6)
    s1,t1=c.moments(tt[:3],tx[:3]);s2,t2=c.moments(tt[3:],tx[3:])
    np.testing.assert_allclose(np.concatenate([s1,s2]),sx,atol=1e-14)
    np.testing.assert_allclose(np.concatenate([t1,t2]),t4,atol=1e-14)

def test_smooth_toy():
    # Average Hessians = Hessian of average function; A vs B cross terms.
    e=nodes.antithetic(32,2,7)*.3
    def f(v):return v[0]**2+3*v[0]*v[1]+torch.sin(v[1]*20)*.01
    v=torch.tensor([.2,.1],dtype=torch.float64)
    hs=torch.stack([jacfwd(jacrev(f))(v+torch.tensor(x)) for x in e])
    def smooth(w):return torch.stack([f(w+torch.tensor(x)) for x in e]).mean()
    torch.testing.assert_close(jacfwd(jacrev(smooth))(v),hs.mean(0))
    np.testing.assert_allclose(hs[:,0,0],2,atol=1e-12)
    assert not torch.allclose(hs.square().mean(0),hs.mean(0).square())
    h1=torch.tensor([1.,-2.]);h2=torch.tensor([-1.,4.]);w=.5
    torch.testing.assert_close((w*h1+w*h2)**2,w*w*h1**2+w*w*h2**2+2*w*w*h1*h2)

if __name__=='__main__':
    test_gaussian();test_contraction();test_smooth_toy();print('ALL MATH SELFTESTS PASS')
