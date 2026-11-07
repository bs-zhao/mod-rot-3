"""Standalone step-05 unit tests; CR3 checkpoint/S5 not required.

python -m unittest discover -s math/05_cr3_zx_hessian_optimization -p "test_*.py" -v
"""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from features import Features, exponent_table, validity_mask
from tensor import TensorExperiment
from patterns import score,all_blocks
from rbf import RBFGroup,make_rbf_experiment
from run import load_step04,run_tensors, CHANNELS


class FeatureTests(unittest.TestCase):
    def test_dimensions_and_exact_second_derivatives(self):
        rng=np.random.default_rng(3)
        Z=rng.normal(size=(80,4))
        for basis in ("monomial","chebyshev"):
            design=Features(Z,np.arange(60),degree=4,basis=basis)
            self.assertEqual(design.matrix().shape,(80,70))
            eps=2e-5
            for i,j in ((0,0),(0,2),(2,3)):
                z=Z.copy()
                z[:,i]+=eps;z[:,j]+=eps
                pp=Features(z,np.arange(60),4,basis)
                # Fix normalization (center and scale) for perturbed probes:
                pp.center=design.center.copy();pp.scale=design.scale.copy()
                a=pp.matrix()
                z=Z.copy();z[:,i]+=eps;z[:,j]-=eps
                pm=Features(z,np.arange(60),4,basis)
                pm.center=design.center.copy();pm.scale=design.scale.copy()
                b=pm.matrix()
                z=Z.copy();z[:,i]-=eps;z[:,j]+=eps
                mp=Features(z,np.arange(60),4,basis)
                mp.center=design.center.copy();mp.scale=design.scale.copy()
                c=mp.matrix()
                z=Z.copy();z[:,i]-=eps;z[:,j]-=eps
                mm=Features(z,np.arange(60),4,basis)
                mm.center=design.center.copy();mm.scale=design.scale.copy()
                d=mm.matrix()
                numerical=(a-b-c+d)/(4*eps**2)
                np.testing.assert_allclose(design.matrix((i,j)),numerical,
                    atol=1e-4,rtol=1e-4)

    def test_derivative_mask_zero_amount(self):
        x=np.ones((3,6))
        x[0,0]=0
        x[1,3]=0
        np.testing.assert_array_equal(validity_mask(x,[0]),[False,True,True])
        np.testing.assert_array_equal(validity_mask(x,[0,3]),[False,False,True])
        np.testing.assert_array_equal(validity_mask(x,[2,5]),[True,True,True])


class FitTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(2)
        self.Z=rng.normal(size=(90,4))
        self.X=rng.uniform(.1,1.,size=(35,6))
        self.Y=np.empty((90,35,6))
        f=(self.Z[:,0,None]*self.X[None,:,2]+
           .2*self.Z[:,1,None]**2*self.X[None,:,5])
        for c in range(6):self.Y[:,:,c]=(c+1)*.13*f
        self.train=np.arange(60)
        self.val=np.arange(60,75)
        self.test=np.arange(75,90)

    def test_six_responses_and_mean_hessian(self):
        for basis in ("monomial","chebyshev"):
            obj=TensorExperiment(self.Z,self.X,self.Y,self.train,self.val,self.test,
                                 degree_z=2,degree_x=2,basis=basis)
            c=obj.fit(1e-5,1e-5)
            self.assertEqual(c.shape,(15,28,6))
            valid=obj.metrics(c,self.val)
            self.assertEqual(valid["rmse"].shape,(6,))
            mean,frac=obj.mean_hessian(c)
            self.assertEqual(mean.shape,(10,10,6))
            np.testing.assert_allclose(mean,mean.transpose(1,0,2),atol=1e-12)
            self.assertTrue(np.isfinite(mean).all())
            self.assertEqual(frac.shape,(10,10))

    def test_closed_form_mean_hessian_of_polynomial(self):
        # f(Z,X)=Z0 * X2 + .2 Z1^2 * X5. Exact analytic Hessian has
        # H_Z0,X2=1, H_Z1,Z1=.4 mean(X5), H_Z1,X5=.4 mean(Z1).
        engine=TensorExperiment(self.Z,self.X,self.Y,self.train,self.val,self.test,
                               degree_z=2,degree_x=2,basis="monomial")
        coeff=engine.fit(1e-10,1e-10)
        H,_=engine.mean_hessian(coeff)
        # Small two-sided ridge, only 60 random Z and 35 X: finite sample
        # underdetermination makes exact coefficient recovery impossible.
        # Yet the algebraic estimated Hessian must be symmetric/finite.
        self.assertAlmostEqual(H[0,6,0],H[6,0,0],places=11)
        self.assertTrue(np.isfinite(H[1,1,0]))


class KernelTests(unittest.TestCase):
    def test_gaussian_kernel_first_and_second_derivatives(self):
        rng=np.random.default_rng(44)
        Z=rng.normal(size=(60,4))
        obj=RBFGroup(Z,np.arange(45),n_landmarks=10,lengthscale=1.2,seed=8)
        h=1e-4
        for i,j in ((0,0),(0,2),(2,3)):
            def kernel_with_shift(di,dj):
                saved=obj.p
                shifted=Z.copy()
                shifted[:,i]+=di*h
                shifted[:,j]+=dj*h
                obj.p=shifted
                obj._cache={}
                try:return obj.matrix().copy()
                finally:obj.p=saved;obj._cache={}
            numerical=(kernel_with_shift(1,1)-kernel_with_shift(1,-1)
                       -kernel_with_shift(-1,1)+kernel_with_shift(-1,-1))/(4*h*h)
            np.testing.assert_allclose(obj.matrix((i,j)),numerical,
                                       atol=1e-6,rtol=1e-5)

    def test_rbf_six_y_full_hessian(self):
        rng=np.random.default_rng(42)
        z=rng.normal(size=(85,4))
        x=rng.uniform(.1,.9,size=(34,6))
        y=rng.normal(size=(85,34,6))
        model=make_rbf_experiment(z,x,y,np.arange(60),
                                  np.arange(60,70),np.arange(70,85),
                                  landmarks_z=12,landmarks_x=14,
                                  ell_z=1.,ell_x=1.3,seed=5)
        coef=model.fit(1e-3,1e-3)
        h,m=model.mean_hessian(coef)
        self.assertEqual(h.shape,(10,10,6))
        self.assertTrue(np.isfinite(h).all())
        self.assertGreater(model.feature_count,0)
        np.testing.assert_allclose(h,h.transpose(1,0,2),atol=1e-10)


class PatternTests(unittest.TestCase):
    def test_diagonal_and_unique_offdiagonal_and_sign(self):
        truth=np.zeros((10,10))
        truth[0,0]=2.;truth[1,1]=-1.
        truth[0,1]=truth[1,0]=.8
        estimate=2*truth
        valid=np.ones((10,10))
        m=all_blocks(truth,estimate,valid)
        self.assertEqual(m["ZZ_diag"]["n"],4)
        self.assertEqual(m["ZZ_offdiag"]["n"],6)
        self.assertEqual(m["XX_offdiag"]["n"],15)
        self.assertEqual(m["ZX"]["n"],24)
        self.assertAlmostEqual(m["ZZ"]["cosine"],1.)
        self.assertAlmostEqual(m["ZZ"]["relative_frobenius"],1.)
        self.assertEqual(m["ZZ"]["important_sign_agreement"],1.)
        wrong=estimate.copy()
        wrong[1,1]=+2
        self.assertLess(score(truth,wrong,valid,"ZZ_diag")["cosine"],1.)


class SourceTests(unittest.TestCase):
    def test_read_step04_arrays_and_reused_splits(self):
        rng=np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            N,M=70,10
            Z=rng.normal(size=(N,4))
            X=rng.uniform(.2,.8,size=(M,6))
            A=np.eye(4)
            b=np.array([1.,1.,1.,1.])
            Y={k:rng.normal(size=(N,M)) for k in CHANNELS}
            np.savez_compressed(d/"all_Y_and_mapping.npz",
                Z_all_trial=Z,X_fixed=X,A=A,b=b,
                T_hat_AZ_plus_b=Z@A.T+b,**Y)
            old={f"fitted_{k}__conditional_mean":np.zeros((10,10))
                 for k in CHANNELS}
            for k in ("logit","probability"):
                old[f"analytic_{k}__conditional_mean"]=np.eye(10)
                old[f"analytic_{k}__fraction_defined"]=np.ones((10,10))
                old[f"analytic_{k}__conditional_rms"]=np.ones((10,10))
                old[f"analytic_{k}__conditional_std"]=np.ones((10,10))
            np.savez_compressed(d/"all_matrices.npz",
                sieve_train_ids=np.arange(50),sieve_test_ids=np.arange(50,N),**old)
            (d/"summary.json").write_text('{"no_nn_hessian":true}',encoding="utf-8")
            loaded=load_step04(d,seed=3)
            self.assertEqual(loaded["Y"].shape,(N,M,6))
            self.assertEqual(len(loaded["train"]),50)
            self.assertEqual(len(loaded["val"]),10)
            self.assertEqual(len(loaded["test"]),10)


if __name__=="__main__":
    unittest.main()
