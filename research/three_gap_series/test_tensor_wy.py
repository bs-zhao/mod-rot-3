import unittest
import numpy as np
import torch
from tensor_wy import TensorWY


class TensorWYTests(unittest.TestCase):
    def test_anova_kernel_value_and_derivatives_use_same_generic_dictionary(self):
        rng=np.random.default_rng(202);z=rng.normal(size=(50,2));x=rng.normal(size=(70,3))
        tables=[x for _ in z]
        ys=[zz[0]*np.sin(x[:,0])*np.cos(x[:,2])+zz[1]**2*x[:,1] for zz in z]
        for kernel in ['anova2','anova3']:
            fit=TensorWY(2,1e-8,1e-5,24,.8,3,kernel).fit(z,tables,ys)
            q=torch.tensor([.2,-.1,.3,.4,-.2],dtype=torch.float64)
            self.assertAlmostEqual(float(fit.value(q)),fit.predict_subject(q.numpy()[:2],q.numpy()[None,2:])[0],places=10)
            h=torch.func.hessian(fit.value)(q).numpy();step=1e-4
            approximate=np.empty_like(h)
            for j in range(len(q)):
                delta=torch.eye(len(q),dtype=q.dtype)[j]*step
                approximate[:,j]=((torch.func.jacrev(fit.value)(q+delta)-torch.func.jacrev(fit.value)(q-delta))/(2*step)).numpy()
            np.testing.assert_allclose(h,approximate,atol=1e-6,rtol=1e-5)
            np.testing.assert_allclose(fit.latent_fields(q.numpy()[None,:2],q.numpy()[None,2:])['h'][0],h[:2,:2],atol=1e-10)
    def test_generic_dictionary_recovers_polynomial_interactions(self):
        rng=np.random.default_rng(99);z=rng.normal(size=(90,2));bank=rng.uniform(-1,1,(150,2))
        x=[bank.copy() for _ in z]
        y=[zz[0]*bank[:,0]**2+zz[1]**2*bank[:,0]*bank[:,1]+.3*zz[0]*zz[1]+.2*zz[0]**3*bank[:,1] for zz in z]
        fitted=TensorWY(3,1e-10,1e-4,12,1.,31).fit(z,x,y)
        q=torch.tensor([.2,-.3,.4,.1],dtype=torch.float64)
        def exact(w):return w[0]*w[2]**2+w[1]**2*w[2]*w[3]+.3*w[0]*w[1]+.2*w[0]**3*w[3]
        got=torch.func.hessian(fitted.value)(q).detach().numpy()
        expected=torch.func.hessian(exact)(q).detach().numpy()
        np.testing.assert_allclose(got,expected,atol=2e-6)
        self.assertAlmostEqual(fitted.predict_subject(q.numpy()[:2],q.numpy()[None,2:])[0],float(fitted.value(q)),places=10)
        self.assertFalse(fitted.info['model_identity_used'])
        latent=fitted.latent_fields(q.numpy()[None,:2],q.numpy()[None,2:],True)
        np.testing.assert_allclose(latent['h'][0],got[:2,:2],atol=1e-10)
        t=torch.func.jacfwd(torch.func.hessian(fitted.value))(q).detach().numpy()
        np.testing.assert_allclose(latent['third'][0],t[:2,:2,:2],atol=1e-10)


if __name__=='__main__':unittest.main()
