"""Regression, chain-rule, and six-channel matrix-estimator correctness tests.

Does not require an actual saved CR3 checkpoint or S5 file.
Run: python -m unittest discover -s math/cr3_zx_hessian_threeway -p "test_*.py" -v
"""
import unittest
import numpy as np
import torch

from run import (fit_affine,pullback_hessian,make_six_Y,
                 decoder_y,mapping_metrics,analytic_zx_moments)
from data_only import TensorPolynomial
from oracle import oracle_hessians,oracle_outputs
from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig,DiscreteNeuralProcess


class TestAffineAndTransformation(unittest.TestCase):
    def test_train_regression_and_affine_mapping(self):
        rng=np.random.default_rng(11)
        Z=rng.normal(size=(100,4))
        A=np.array([[.5,.1,0.,0.],[-.2,1.2,.1,0.],
                    [.1,.15,.9,.2],[0.,.1,.05,1.]])
        b=np.array([.8,2.,.55,1.6])
        T=Z@A.T+b
        fitA,fitb,pred,singular,condition=fit_affine(Z,T,np.arange(65))
        np.testing.assert_allclose(fitA,A,atol=1e-12)
        np.testing.assert_allclose(fitb,b,atol=1e-12)
        np.testing.assert_allclose(pred,T,atol=1e-12)
        self.assertGreater(singular.min(),0)
        self.assertLess(condition,10)
        summary=mapping_metrics(Z,T,pred,{"train":np.arange(65),"test":np.arange(65,100)})
        np.testing.assert_allclose(summary["test"]["per_theta_R2"],1.,atol=1e-12)

    def test_train_only_does_not_peek_at_test_subjects(self):
        rng=np.random.default_rng(29)
        Z=rng.normal(size=(100,4))
        T=Z@np.eye(4)+np.array([.5,1.,1.2,2.])
        T[70:]+=rng.normal(size=(30,4))*8
        A,b,_,_,_=fit_affine(Z,T,np.arange(70))
        np.testing.assert_allclose(A,np.eye(4),atol=1e-12)
        np.testing.assert_allclose(b,[.5,1.,1.2,2.],atol=1e-12)

    def test_pullback_known_full_matrix_and_nan_isolation(self):
        rng=np.random.default_rng(8)
        A=rng.normal(size=(4,4))
        H=rng.normal(size=(2,3,10,10))
        H=(H+H.swapaxes(-1,-2))/2
        J=np.eye(10);J[:4,:4]=A
        actual=pullback_hessian(H,A)
        expected=np.einsum("ki,...kl,lj->...ij",J,H,J,optimize=True)
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        H[...,4,5]=np.nan
        H[...,5,4]=np.nan
        out=pullback_hessian(H,A)
        self.assertTrue(np.isfinite(out[...,:4,:4]).all())
        self.assertTrue(np.isnan(out[...,4,5]).all())

    def test_oracle_pullback_matches_finite_diff_in_zx(self):
        A=np.array([[.08,.02,0.,0.],[0.,.11,0.,0.],
                    [0.,0.,.07,.01],[0.,.01,0.,.12]])
        b=np.array([.7,2.0,.6,2.2])
        Z=np.array([[.2,-.4,.1,.3]])
        X=np.array([[30.,20.,.2,80.,15.,.8],
                    [50.,40.,.7,20.,25.,.35]])
        T=Z@A.T+b
        h_eta,h_p=oracle_hessians(T,X)
        analytic=pullback_hessian(h_eta,A)
        def F(inputs):
            z=inputs[:4]
            x=inputs[4:]
            theta=z[None,:]@A.T+b
            eta,_=oracle_outputs(theta,x[None,:])
            return float(eta[0,0])
        location=np.r_[Z[0],X[0]]
        h=1e-4
        for i,j in ((0,0),(0,2),(0,4),(2,6),(4,4),(6,9)):
            ei=np.zeros(10);ei[i]=h
            ej=np.zeros(10);ej[j]=h
            numeric=(F(location+ei+ej)-F(location+ei-ej)
                    -F(location-ei+ej)+F(location-ei-ej))/(4*h*h)
            self.assertAlmostEqual(numeric,analytic[0,0,i,j],delta=2e-4,
                                  msg=f"Z-X second derivative {i},{j}")


class TestSixYAndHessianEstimator(unittest.TestCase):
    def test_all_six_responses_formed_correctly(self):
        eta=np.linspace(-2,2,15).reshape(3,5)
        p=1./(1+np.exp(-eta))
        en=.8*eta
        pn=1./(1+np.exp(-en))
        ys=make_six_Y(eta,p,en,pn,seed=99)
        self.assertEqual(len(ys),6)
        self.assertEqual(set(ys),{
            "oracle_choice","oracle_logit","oracle_probability",
            "nn_choice","nn_logit","nn_probability"})
        np.testing.assert_array_equal(ys["nn_choice"],(pn>=.5).astype(float))
        np.testing.assert_array_equal(ys["oracle_logit"],eta)
        np.testing.assert_array_equal(ys["oracle_probability"],p)
        self.assertTrue(np.isin(ys["oracle_choice"],[0,1]).all())
        # Shared uniforms make paired Bernoulli draws equal if their p agree.
        ys_both=make_six_Y(eta,p,eta,p,seed=99,
                          oracle_choice_mode="sampled",nn_choice_mode="sampled")
        np.testing.assert_array_equal(ys_both["oracle_choice"],ys_both["nn_choice"])

    def test_decoder_only_generates_values(self):
        torch.manual_seed(3)
        rng=np.random.default_rng(3)
        cfg=DiscreteNPConfig(x_dim=6,num_choices=2,latent_dim=4)
        model=DiscreteNeuralProcess(cfg).eval()
        ck={"model_config":cfg.to_dict(),"model_state_dict":model.state_dict()}
        Z=rng.normal(size=(4,4))
        X=np.column_stack([rng.uniform(5,100,7),
                           rng.uniform(5,100,7),
                           rng.uniform(.1,.9,7),
                           rng.uniform(5,100,7),
                           rng.uniform(5,100,7),
                           rng.uniform(.1,.9,7)])
        eta,p,error=decoder_y(ck,Z,X,batch_size=2,device="cpu")
        self.assertEqual(eta.shape,(4,7))
        self.assertEqual(p.shape,(4,7))
        self.assertLess(error,1e-6)
        np.testing.assert_allclose(1/(1+np.exp(-eta)),p,atol=1e-6)

    def test_matrix_estimator_six_channels_common_ZX(self):
        rng=np.random.default_rng(17)
        n,m=80,40
        Z=rng.normal(size=(n,4))
        X=np.column_stack([rng.uniform(.1,1,m) for _ in range(6)])
        base=Z[:,0,None]*X[None,:,2]+Z[:,2,None]**2*.2*X[None,:,5]
        p=1/(1+np.exp(-base))
        ys=make_six_Y(base,p,.8*base,1/(1+np.exp(-.8*base)),seed=17)
        estimator=TensorPolynomial(Z,X,degree_t=2,degree_x=2,
                                   ridge_t=.01,ridge_x=.01,seed=17)
        for k,y in ys.items():
            coef=estimator.fit(y)
            metrics=estimator.holdout_metrics(y,coef)
            self.assertTrue(np.isfinite(metrics["rmse"]),k)
            estimate=estimator.expected_hessian(coef,Z,np.full(n,1/n))
            self.assertEqual(estimate["conditional_mean"].shape,(10,10))
            np.testing.assert_allclose(estimate["conditional_mean"],
                                       estimate["conditional_mean"].T,atol=1e-12)

    def test_transformed_oracle_tt_mean_finite_despite_zero_amount(self):
        Z=np.array([[.0,.0,.0,.0],[.2,-.1,.1,-.2]])
        A=np.diag([.05,.2,.05,.3])
        b=np.array([.7,2.,.65,2.])
        T=Z@A.T+b
        X=np.array([[0.,30.,.2,50.,0.,.7],[30.,0.,.8,0.,20.,.4]])
        oracle=analytic_zx_moments(T,X,A,batch_size=1)
        self.assertEqual(oracle["logit"]["conditional_mean"].shape,(10,10))
        self.assertTrue(np.isfinite(oracle["logit"]["strict_mean"][:4,:4]).all())
        self.assertTrue(np.isnan(oracle["logit"]["strict_mean"][4,4]))


if __name__=="__main__":
    unittest.main()
