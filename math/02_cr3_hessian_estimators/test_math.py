import unittest
import numpy as np
from scipy.special import expit
from estimators import split_subjects,fit_krr,fit_spline,second_order_score,fit_stein_mean,SplineDesign
from reference import prior_quadrature,logits_and_hess_tt,truth

class EstimatorTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(33)
        self.T=rng.uniform([.1,.5,.1,.7],[.9,3.,.9,3.3],(180,4))
        self.y=self.T[:,0]**2+2*self.T[:,1]*self.T[:,2] + .3*self.T[:,3]
        self.analytic=np.zeros((4,4));self.analytic[0,0]=2
        self.analytic[1,2]=self.analytic[2,1]=2
        self.split=split_subjects(len(self.T),11)

    def test_spline_quadratic_hessian(self):
        # Verify the analytic B-spline derivative map independently of
        # smoothing bias / underdetermined regression in a tiny sample.
        from estimators import SplineModel
        rng=np.random.default_rng(44)
        T=rng.uniform([.1,.5,.1,.7],[.9,3.,.9,3.3],(400,4))
        y=T[:,0]**2+2*T[:,1]*T[:,2]+.3*T[:,3]
        design=SplineDesign(n_basis=4)
        c=np.linalg.lstsq(design.features(T),y,rcond=1e-12)[0]
        H=SplineModel(design,c).hessian(T[:40])
        np.testing.assert_allclose(H,np.broadcast_to(self.analytic,H.shape),atol=1e-7,rtol=1e-7)
        models,_=fit_spline(T,{'logit':y},split_subjects(len(T),11),n_basis=4,lambdas=(1e-10,1e-7))
        self.assertLess(float(np.sqrt(np.mean((models['logit'].predict(T)-y)**2))),.01)

    def test_kernel_quadratic_mixed_derivatives(self):
        models,_=fit_krr(self.T,{'logit':self.y},self.split,
                         rank=110,lengthscales=(1.,2.,3.),lambdas=(1e-8,1e-6),seed=42)
        H=models['logit'].hessian(self.T[self.split['test']])
        self.assertTrue(np.isfinite(H).all())
        self.assertLess(float(np.mean(abs(H[:,0,0]-2))),.35)
        self.assertLess(float(np.mean(abs(H[:,1,2]-2))),.35)
        np.testing.assert_allclose(H,H.transpose(0,2,1),atol=1e-10)

    def test_prior_quadrature_mass(self):
        points,weights,_=prior_quadrature(8)
        self.assertEqual(points.shape,(8**4,4))
        self.assertAlmostEqual(weights.sum(),1.,places=10)

    def test_weighted_stein_identity_polynomial(self):
        T,wq,_=prior_quadrature(12)
        W,psi=second_order_score(T)
        f=T[:,0]**2+2*T[:,1]*T[:,2]
        expected=np.zeros((4,4));expected[0,0]=2;expected[1,2]=expected[2,1]=2
        est=np.einsum('n,n,nij->ij',wq,f,psi)/(wq@W)
        np.testing.assert_allclose(est,expected,atol=.01,rtol=.01)

    def test_stein_only_weighted_target(self):
        est,se,normalizer=fit_stein_mean(self.T,self.y)
        self.assertEqual(est.shape,(4,4))
        self.assertEqual(se.shape,(4,4))
        self.assertGreater(normalizer,0)
        self.assertTrue(np.isfinite(est).all())

    def test_cr3_hessian_tt_matches_finite_differences(self):
        T=np.array([[.8,2.,.7,2.2]])
        X=np.array([[100.,60.,.6,80.,20.,.4],
                    [0.,30.,0.,90.,0.,1.]])
        eta,p,H_eta,H_p=logits_and_hess_tt(T,X)
        for key,H in ((0,H_eta),(1,H_p)):
            def f(t):
                a,b,_,_=logits_and_hess_tt(t[None,:],X)
                return float((a if key==0 else b)[0])
            for i in range(4):
                for j in range(4):
                    delta=1e-4;ei=np.eye(4)[i]*delta;ej=np.eye(4)[j]*delta
                    if i==j:
                        num=(-f(T[0]+2*ei)+16*f(T[0]+ei)-30*f(T[0])+16*f(T[0]-ei)-f(T[0]-2*ei))/(12*delta**2)
                    else:
                        num=(f(T[0]+ei+ej)-f(T[0]+ei-ej)-f(T[0]-ei+ej)+f(T[0]-ei-ej))/(4*delta**2)
                    self.assertAlmostEqual(num,H[0,i,j],delta=2e-5)

if __name__=='__main__':unittest.main()
