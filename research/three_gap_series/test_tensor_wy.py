import unittest
import numpy as np
import torch
from tensor_wy import TensorWY


class TensorWYTests(unittest.TestCase):
    def test_generic_dictionary_recovers_polynomial_interactions(self):
        rng=np.random.default_rng(99);z=rng.normal(size=(90,2));bank=rng.uniform(-1,1,(150,2))
        x=[bank.copy() for _ in z]
        y=[zz[0]*bank[:,0]**2+zz[1]**2*bank[:,0]*bank[:,1]+.3*zz[0]*zz[1] for zz in z]
        fitted=TensorWY(3,1e-10,1e-4,12,1.,31).fit(z,x,y)
        q=torch.tensor([.2,-.3,.4,.1],dtype=torch.float64)
        def exact(w):return w[0]*w[2]**2+w[1]**2*w[2]*w[3]+.3*w[0]*w[1]
        got=torch.func.hessian(fitted.value)(q).detach().numpy()
        expected=torch.func.hessian(exact)(q).detach().numpy()
        np.testing.assert_allclose(got,expected,atol=2e-6)
        self.assertAlmostEqual(fitted.predict_subject(q.numpy()[:2],q.numpy()[None,2:])[0],float(fitted.value(q)),places=10)
        self.assertFalse(fitted.info['model_identity_used'])


if __name__=='__main__':unittest.main()
