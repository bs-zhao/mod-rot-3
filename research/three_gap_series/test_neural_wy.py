import unittest
import numpy as np
import torch
from neural_wy import NeuralWY


class NeuralWYTests(unittest.TestCase):
    def test_generic_cubic_branch_recovers_true_hessian_without_neural_fit(self):
        rng=np.random.default_rng(119);w=rng.normal(size=(1000,4));v=rng.normal(size=(180,4))
        def numpy_target(a):return a[:,0]*a[:,2]**2+a[:,1]**2*a[:,3]+.2*a[:,0]*a[:,1]*a[:,2]
        fit=NeuralWY(2,width=8,epochs=1,seed=11,device='cpu').fit(w,numpy_target(w),v,numpy_target(v))
        self.assertTrue(fit.info['skipped_neural_residual'])
        q=torch.tensor([.2,-.4,.5,.6],dtype=torch.float64)
        def exact(a):return a[0]*a[2]**2+a[1]**2*a[3]+.2*a[0]*a[1]*a[2]
        np.testing.assert_allclose(torch.func.hessian(fit.value)(q).numpy(),torch.func.hessian(exact)(q).numpy(),atol=2e-6)
        self.assertFalse(fit.info['teacher_derivatives_used_for_fit'])

    def test_neural_residual_is_smooth_and_its_hessian_matches_value_differences(self):
        rng=np.random.default_rng(991);w=rng.normal(size=(220,4));v=rng.normal(size=(80,4))
        target=lambda a:np.sin(a[:,0])*np.cos(a[:,2])+a[:,1]*a[:,3]
        fit=NeuralWY(2,width=8,epochs=1,seed=11,device='cpu').fit(w,target(w),v,target(v))
        self.assertFalse(fit.info['skipped_neural_residual'])
        q=torch.tensor([.2,-.4,.5,.6],dtype=torch.float64);h=torch.func.hessian(fit.value)(q).numpy()
        approx=np.empty_like(h);step=1e-4
        for j in range(len(q)):
            delta=torch.eye(len(q),dtype=q.dtype)[j]*step
            approx[:,j]=((torch.func.jacrev(fit.value)(q+delta)-torch.func.jacrev(fit.value)(q-delta))/(2*step)).numpy()
        np.testing.assert_allclose(h,approx,atol=1e-6)


if __name__=='__main__':unittest.main()
