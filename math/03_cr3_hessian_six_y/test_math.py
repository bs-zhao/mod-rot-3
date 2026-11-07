"""Unit tests; no trained CR3 checkpoint or S5 file required.

python -m unittest discover -s math/03_cr3_hessian_six_y -p "test_*.py" -v
"""
from __future__ import annotations
import unittest

import numpy as np
import torch

from run import (fit_affine_map, regression_diagnostics,
                 decoder_responses, pair_metrics)
from data_only import TensorPolynomial
from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig, DiscreteNeuralProcess


class AffineMapTests(unittest.TestCase):
    def test_invertible_affine_ols_recovers_known_A_and_b(self):
        rng=np.random.default_rng(4)
        Z=rng.normal(size=(150,4))
        A=np.array([[1.3,.2,-.1,0.],
                    [-.2,1.7,.1,.15],
                    [.1,.2,1.1,-.1],
                    [0.,-.1,.05,1.4]])
        b=np.array([.4,-1.5,2.2,.1])
        T=Z@A.T+b
        fit=fit_affine_map(Z,T,np.arange(110))
        np.testing.assert_allclose(fit["A"],A,atol=2e-12)
        np.testing.assert_allclose(fit["b"],b,atol=2e-12)
        np.testing.assert_allclose(fit["mapped_Z"],Z,atol=2e-12)
        self.assertTrue(np.isfinite(fit["condition_number"]))
        splits={"train":np.arange(100),"validation":np.arange(100,125),
                "test":np.arange(125,150)}
        d=regression_diagnostics(fit,Z,T,splits)
        np.testing.assert_allclose(d["groups"]["test"]["R2_per_theta"],1.,atol=1e-12)

    def test_ols_map_uses_only_train_subjects(self):
        rng=np.random.default_rng(51)
        Z=rng.normal(size=(150,4))
        A=np.diag([1.,2.,3.,4.])
        T=Z@A.T+np.array([1.,2.,3.,4.])
        train=np.arange(90)
        T[90:]+=rng.normal(size=(60,4))*12
        fit=fit_affine_map(Z,T,train)
        np.testing.assert_allclose(fit["A"],A,atol=1e-12)
        np.testing.assert_allclose(fit["b"],[1.,2.,3.,4.],atol=1e-12)
        self.assertGreater(np.linalg.norm(fit["mapped_Z"][90:]-Z[90:]),1.)

    def test_singular_affine_map_is_rejected(self):
        rng=np.random.default_rng(11)
        Z=rng.normal(size=(50,4))
        T=Z@np.eye(4)
        T[:,3]=2*T[:,2]
        with self.assertRaisesRegex(RuntimeError,"singular|unstable"):
            fit_affine_map(Z,T,np.arange(40))

    def test_zero_intercept_option(self):
        rng=np.random.default_rng(8)
        Z=rng.normal(size=(120,4))
        A=np.diag([1.1,.3,5.,.8])
        T=Z@A.T
        fit=fit_affine_map(Z,T,np.arange(90),include_intercept=False)
        np.testing.assert_allclose(fit["mapped_Z"],Z,atol=1e-12)
        np.testing.assert_array_equal(fit["b"],np.zeros(4))


class DecoderYTests(unittest.TestCase):
    def test_decoder_outputs_only_numerical_Y_and_matches_softmax(self):
        torch.manual_seed(7)
        rng=np.random.default_rng(7)
        config=DiscreteNPConfig(x_dim=6,latent_dim=4,num_choices=2)
        model=DiscreteNeuralProcess(config).eval()
        ck=dict(model_config=config.to_dict(),
                model_state_dict=model.state_dict())
        X=np.column_stack((
            rng.uniform(0,120,12),rng.uniform(0,100,12),
            rng.uniform(.1,.9,12),rng.uniform(0,120,12),
            rng.uniform(0,100,12),rng.uniform(.1,.9,12)))
        Z=rng.normal(size=(5,4))
        actual,e=decoder_responses(ck,X,Z,batch_size=2,device="cpu")
        self.assertEqual(set(actual),{"nn_choiceprediction","nn_logit","nn_probability"})
        for x in actual.values():self.assertEqual(x.shape,(5,12))
        self.assertLess(e,1e-6)
        np.testing.assert_array_equal(actual["nn_choiceprediction"],
                                      (actual["nn_probability"]>=.5).astype(float))
        with torch.inference_mode():
            zz=torch.tensor(Z,dtype=torch.float32)
            xx=torch.tensor(X,dtype=torch.float32)[None].expand(5,-1,-1)
            pair=model.decode(xx,zz)
            expected=(pair[...,1]-pair[...,0]).numpy()
        np.testing.assert_allclose(actual["nn_logit"],expected,atol=1e-6)

    def test_six_different_Y_fit_same_T_and_X_design(self):
        rng=np.random.default_rng(13)
        n,m=90,30
        T=np.column_stack((
            rng.uniform(.15,.95,n),rng.uniform(.5,4.7,n),
            rng.uniform(.1,.92,n),rng.uniform(.6,3.6,n)))
        X=np.column_stack((
            rng.uniform(1,130,m),rng.uniform(1,100,m),
            rng.uniform(.1,.9,m),rng.uniform(1,130,m),
            rng.uniform(1,100,m),rng.uniform(.1,.9,m)))
        base=(T[:,0,None]*X[None,:,2]
              -T[:,1,None]*X[None,:,1]/100.
              +T[:,2,None]**2 * X[None,:,5])
        ys=dict(
            original_choice=(rng.random((n,m))<1/(1+np.exp(-base))).astype(float),
            original_logit=base,
            original_probability=1/(1+np.exp(-base)),
            nn_choiceprediction=(base+.1>0).astype(float),
            nn_logit=.7*base+.05,
            nn_probability=1/(1+np.exp(-(.7*base+.05))),
        )
        model=TensorPolynomial(T,X,degree_t=2,degree_x=2,
                               ridge_t=.01,ridge_x=.01,seed=51)
        self.assertEqual(len(ys),6)
        for y in ys.values():
            coef=model.fit(y)
            self.assertEqual(coef.shape,(15,28))
            diagnostics=model.holdout_metrics(y,coef)
            self.assertTrue(np.isfinite(diagnostics["rmse"]))
            result=model.expected_hessian(coef,T,np.ones(len(T))/len(T))
            self.assertEqual(result["conditional_mean"].shape,(10,10))
            np.testing.assert_allclose(
                result["conditional_mean"],
                result["conditional_mean"].T,atol=1e-12)

    def test_predictions_metrics(self):
        a=np.array([[1.,2.],[3.,4.]])
        report=pair_metrics(a,2*a+1)
        self.assertAlmostEqual(report["pearson_r"],1.)
        self.assertAlmostEqual(report["bias"],3.5)


if __name__=="__main__":
    unittest.main()
