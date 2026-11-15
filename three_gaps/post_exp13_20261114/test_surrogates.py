"""Check analytic sieve / RBF derivatives against independent torch autograd."""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.func import jacrev,hessian,vmap
sys.path.insert(0,str(Path(__file__).resolve().parent))
import core as C
sys.path.insert(0,str(C.HERE/'exp18_matrix_only'))
from surrogates import Polynomial,Kernel

class SurrogateDerivatives(unittest.TestCase):
    def test_polynomial_and_kernel(self):
        rng=np.random.default_rng(81); W=rng.normal(size=(256,5)); V=rng.normal(size=(32,5))
        func=lambda w:w[:,0]*w[:,2]+w[:,1]**2+.2*w[:,3]**3
        for klass in (Polynomial,Kernel):
            m=klass().fit(W,func(W),V,func(V)); p=V[:8]; g,uu,ux=m.derivatives(p,2)
            mu=C.tensor(m.mu); sd=C.tensor(m.sd); co=C.tensor(m.coef)
            if klass is Polynomial:
                powers=torch.tensor(m.powers)
                h=lambda w:((((w-mu)/sd)[None]**powers).prod(-1)*co).sum()
            else:
                centers=C.tensor(m.centers)
                h=lambda w:((-( ((w-mu)/sd-centers).square().sum(-1))/(2*m.ell**2)).exp()*co).sum()+m.intercept
            refg=vmap(jacrev(h))(C.tensor(p)).numpy()[:,:2]
            refh=vmap(hessian(h))(C.tensor(p)).numpy()[:,:2]
            np.testing.assert_allclose(g,refg,atol=1e-8,rtol=1e-8)
            np.testing.assert_allclose(uu,refh[:,:,:2],atol=1e-8,rtol=1e-8)
            np.testing.assert_allclose(ux,refh[:,:,2:],atol=1e-8,rtol=1e-8)

if __name__=='__main__': unittest.main()
