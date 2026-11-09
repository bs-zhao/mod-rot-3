import unittest
import numpy as np
from local_wy import LocalPolynomial


class LocalWYTests(unittest.TestCase):
    def test_unknown_cubic_hessian_and_hidden_rotation(self):
        rng=np.random.default_rng(37);w=rng.normal(size=(700,3));q=rng.normal(size=(5,3))*.3
        y=w[:,0]*w[:,1]**2+.2*w[:,2]**3+3*w[:,0]-2*w[:,1]
        estimator=LocalPolynomial(3,500,1.,1e-12).fit(w,y)
        f=estimator.fields(q,True)
        expected=np.zeros((len(q),3,3));expected[:,0,1]=expected[:,1,0]=2*q[:,1]
        expected[:,1,1]=2*q[:,0];expected[:,2,2]=1.2*q[:,2]
        np.testing.assert_allclose(f['h'],expected,atol=2e-8)
        r,_=np.linalg.qr(rng.normal(size=(3,3)))
        other=LocalPolynomial(3,500,1.,1e-12).fit(w@r.T,y).fields(q@r.T,True)
        back=np.einsum('ai,nab,bj->nij',r,other['h'],r)
        np.testing.assert_allclose(back,f['h'],atol=2e-8)
        np.testing.assert_allclose(f['third'][:,0,1,1],2,atol=2e-8)


if __name__=='__main__':unittest.main()
