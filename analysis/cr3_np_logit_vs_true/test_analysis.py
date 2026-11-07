"""Tests for CR3 NP decoder logit/probability analysis.

No saved 5,000-subject dataset or checkpoint required.
Run:
 python -m unittest discover -s analysis/cr3_np_logit_vs_true -p "test_*.py" -v
"""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from metrics import axis_stats, describe, summarize
from run import read_data, infer_all_and_decode, generator_logits
from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig, DiscreteNeuralProcess


class MetricsTests(unittest.TestCase):
    def test_per_subject_and_per_trial_pearson(self):
        true = np.arange(50, dtype=float).reshape(10,5) * .12
        true = true + np.arange(10)[:,None]**2 * .004
        pred = 2 * true + 1
        whole = summarize(true,pred)
        self.assertAlmostEqual(whole["pooled"]["pearson_r"],1.,places=12)
        self.assertAlmostEqual(whole["pooled"]["slope_pred_on_true"],2.,places=12)
        self.assertAlmostEqual(whole["within_trial_centered"]["pearson_r"],1.,places=12)
        self.assertAlmostEqual(whole["within_subject_centered"]["pearson_r"],1.,places=12)
        self.assertEqual(len(axis_stats(true,pred,axis=1)["rmse"]),10)
        self.assertEqual(len(axis_stats(true,pred,axis=0)["rmse"]),5)

    def test_shifted_class_logits_have_identical_binary_log_odds(self):
        raw = torch.tensor([[[11.,12.],[0.,2.],[-10.,1.]]])
        shifted = raw + torch.tensor([[[50.],[-9.],[1000.]]])
        delta = raw[...,1]-raw[...,0]
        delta2 = shifted[...,1]-shifted[...,0]
        self.assertTrue(torch.allclose(delta,delta2,atol=2e-5))
        self.assertTrue(torch.allclose(
            torch.softmax(raw,dim=-1)[...,1],torch.sigmoid(delta)))
        self.assertTrue(torch.allclose(
            torch.softmax(shifted,dim=-1)[...,1],torch.sigmoid(delta2),atol=2e-5))
        self.assertFalse(torch.allclose(raw[...,1],delta))


class GeneratorTests(unittest.TestCase):
    def test_true_logits_reproduce_s1_saved_probability_at_sure_outcomes(self):
        with tempfile.TemporaryDirectory() as td:
            filepath = Path(td)/"dataset.npz"
            rng = np.random.default_rng(42)
            n, m = 4, 500
            T = np.column_stack((rng.uniform(.4,.95,n),rng.uniform(.5,3,n),
                                  rng.uniform(.2,.9,n),rng.uniform(.8,3.2,n)))
            X = np.column_stack((
                rng.choice([0., 20., 100.], m),
                rng.choice([0., 30., 90.], m),
                rng.choice([0., .15, .5, .8, 1.], m),
                rng.choice([0., 40., 120.], m),
                rng.choice([0., 10., 80.], m),
                rng.choice([0., .2, .5, .9, 1.], m),
            ))
            true_eta=generator_logits(X[None,:,:],T[:,None,:])
            P=expit(true_eta)
            Y=(rng.random(P.shape)<P).astype(np.int64)
            np.savez_compressed(
                filepath,parameters=T,trial_counts=np.full(n,m),
                subject_index=np.repeat(np.arange(n),m),
                trial_index=np.tile(np.arange(m),n),
                x=np.broadcast_to(X,(n,m,6)).reshape(n*m,6).copy(),
                y=Y.reshape(n*m,1),predictive_true=P.reshape(n*m,1))
            t,x,y,p,eta,error=read_data(filepath)
            np.testing.assert_array_equal(t,T)
            np.testing.assert_array_equal(x,X)
            np.testing.assert_array_equal(y,Y)
            np.testing.assert_allclose(p,P)
            np.testing.assert_allclose(eta,true_eta)
            self.assertEqual(error,0.)


class NPInferenceTests(unittest.TestCase):
    def test_full_500_context_matches_direct_encoder_and_decoder(self):
        torch.manual_seed(7)
        rng=np.random.default_rng(7)
        cfg=DiscreteNPConfig(x_dim=6,num_choices=2,latent_dim=4)
        model=DiscreteNeuralProcess(cfg).eval()
        X=np.column_stack((
            rng.uniform(10,150,500),rng.uniform(10,80,500),
            rng.uniform(.1,.9,500),rng.uniform(10,150,500),
            rng.uniform(10,80,500),rng.uniform(.1,.9,500)))
        choice=rng.integers(0,2,(2,500),dtype=np.int64)
        ck={"preprocessing":{
            "encoder_mean":torch.zeros(7),
            "encoder_std":torch.ones(7)}}
        z,lv,logits,p,err=infer_all_and_decode(
            model,ck,X,choice,torch.device("cpu"),batch_size=2)
        with torch.no_grad():
            tx=torch.tensor(X,dtype=torch.float32).unsqueeze(0).expand(2,-1,-1)
            ty=torch.tensor(choice,dtype=torch.float32).unsqueeze(-1)
            mask=torch.ones((2,500),dtype=torch.bool)
            mu,logvar=model.encode_context(tx,ty,mask)
            raw=model.decode(tx,mu,mask)
            odds=raw[...,1]-raw[...,0]
        np.testing.assert_allclose(z,mu.numpy(),atol=1e-6)
        np.testing.assert_allclose(lv,logvar.numpy(),atol=1e-6)
        np.testing.assert_allclose(logits,odds.numpy(),atol=1e-6)
        np.testing.assert_allclose(p,expit(logits),atol=1e-6)
        self.assertLess(err,1e-6)


if __name__ == "__main__":
    unittest.main()
