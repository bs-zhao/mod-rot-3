"""Gap2+Gap3 numerical Hessian estimation and end-to-end mock-checkpoint tests.

Synthetic integration uses FAKE NP predictions injected as Y, testing the
shared-W plumbing without falsely claiming a trained checkpoint was tested.
"""
from __future__ import annotations
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.special import expit

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_gaps
import numeric
from oracle import DIMENSIONS, values
from run import rotation,rotate_h
from metrics import compute_metrics
from selected_metrics import compute_selected


class NumericalHessians(unittest.TestCase):
    def test_selected_metrics_identical_to_full_catalog(self):
        rng=np.random.default_rng(2026)
        d,p,n=4,6,25
        z=rng.normal(size=(n,d,d));z=(z+z.transpose(0,2,1))/2
        x=rng.normal(size=(n,d,p))
        full=compute_metrics(z,x)
        wanted=[name for name in full if
                (name.startswith(("zz_","zx_")) and
                 ("third" not in name and "fourth" not in name))]
        fast=compute_selected(z,x,wanted)
        unsupported=set(wanted)-set(fast)
        self.assertFalse(unsupported,f"Unsupported numeric metric keys: {sorted(unsupported)[:20]}")
        for key in wanted:
            self.assertAlmostEqual(full[key],fast[key],delta=1e-8*max(1,abs(full[key])),
                                   msg=key)

    def test_sieve_recovers_known_quadratic_hessians(self):
        rng=np.random.default_rng(9)
        n=600; p=3
        w=rng.uniform(-1,1,(n,p))
        # f = 2*w0**2 + 3*w0*w2 - .5*w1**2; H00=4,H02=3,H11=-1.
        y=2*w[:,0]**2+3*w[:,0]*w[:,2]-.5*w[:,1]**2
        # Duplicate 4 channels to exercise paired formula/NN outputs.
        outputs=np.stack([y,2*y,y+.1,2*y+.2],axis=1)
        def part(a,b):
            return dict(w=w[a:b],t=np.zeros((1,1)),x=np.zeros((1,1,2)))
        parts=dict(train=part(0,400),validation=part(400,500),test=part(500,600))
        Y=dict(train=outputs[:400],validation=outputs[400:500],test=outputs[500:])
        fit=numeric.sieve_fit(parts,Y,degree=2,ridge=1e-9)
        np.testing.assert_allclose(fit["zz"][:,0,0,0],4,rtol=1e-5,atol=1e-5)
        np.testing.assert_allclose(fit["zx"][:,0,0,1],3,rtol=1e-5,atol=1e-5)
        self.assertLess(max(fit["validation_rmse"]),1e-4)

    def test_rbf_outputs_symmetric_hessian_and_four_sources(self):
        rng=np.random.default_rng(17)
        w=rng.normal(size=(95,4))
        y=np.column_stack([w[:,0]**2+w[:,1]*w[:,2],
                          expit(w[:,0]),w[:,0]+w[:,1]**2,expit(w[:,1])])
        parts={k:dict(w=w[a:b],t=np.zeros((4,2)),x=np.zeros((4,2,2)))
               for k,a,b in (("train",0,60),("validation",60,75),("test",75,95))}
        Y=dict(train=y[:60],validation=y[60:75],test=y[75:])
        fit=numeric.rbf_fit(parts,Y,landmarks=14,bandwidth=1.8,ridge=1e-4,seed=6)
        self.assertEqual(fit["zz"].shape,(20,4,2,2))
        self.assertEqual(fit["zx"].shape,(20,4,2,2))
        self.assertTrue(np.isfinite(fit["zz"]).all())
        np.testing.assert_allclose(fit["zz"],fit["zz"].transpose(0,1,3,2))

    def test_local_hessian_rotation_is_not_rms_matrix_rotation(self):
        h=np.zeros((3,2,2))
        h[:,0,0]=[1,-1,2];h[:,1,1]=[-1,1,-2]
        x=np.ones((3,2,1))
        R=rotation(2,0,1,45)
        a,b=rotate_h(h,x,R)
        self.assertEqual(a.shape,(3,2,2))
        # Preserve Frobenius but generally not per-element RMS.
        np.testing.assert_allclose(np.sum(a*a),np.sum(h*h))
        self.assertGreater(abs(compute_metrics(h,x)["zz_rms_offdiag_l1_loss"]
                               -compute_metrics(a,b)["zz_rms_offdiag_l1_loss"]),.1)


class InMemoryFourColumnGapTest(unittest.TestCase):
    def test_full_gap2_gap3_with_fake_frozen_decoder_output(self):
        """No real NP used. Full pipeline test of folder layout and comparisons."""
        rng=np.random.default_rng(20261109)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            model="C1"; d,p=DIMENSIONS[model]; ns=24; nt=12
            theta=rng.normal(0,.5,(ns,d))
            z=theta + rng.normal(0,.01,(ns,d))
            x=rng.uniform(-1,1,(ns*nt,p))
            datadir=root/"outputs"/"choice"/model
            src=datadir/"s1_generate"/"s1_generate__dataset.npz"
            src.parent.mkdir(parents=True)
            np.savez_compressed(src,parameters=theta,
                trial_counts=np.full(ns,nt),subject_index=np.repeat(np.arange(ns),nt),
                trial_index=np.tile(np.arange(nt),ns),x=x,
                y=np.zeros(ns*nt,int),predictive_true=np.zeros((ns*nt,1)))
            ck=datadir/"s3_train"/"beta0p0005_lambda0_ctx0p25to0p8_L4"/"repeat01"/"s3_train__checkpoint.pt"
            lat=datadir/"s4_evaluate"/"beta0p0005_lambda0_ctx0p25to0p8_L4"/"repeat01"/"s4_evaluate__latent.npz"
            ck.parent.mkdir(parents=True);ck.write_bytes(b"FAKE NN CHECKPOINT FOR TEST ONLY")
            lat.parent.mkdir(parents=True)
            ids=dict(train_ids=np.arange(0,14),validation_ids=np.arange(14,19),test_ids=np.arange(19,24))
            np.savez_compressed(lat,z=z,theta_true=theta,**ids,
                dataset_sha256=np.array(numeric.digest(src)),checkpoint_sha256=np.array(numeric.digest(ck)),
                context_mode=np.array("all_observed_trials"))
            gap1=root/"gap1"
            for name in ("C1",):
                folder=gap1/name;folder.mkdir(parents=True)
                rows=[dict(model="C1",channel=ch,metric="zx_rms_total_l1_loss",
                           success_fraction=1,flat_planes=0)
                      for ch in ("logit","probability")]
                run_gaps.write_csv(folder/"ranking_by_model_channel.csv",rows)
            run_gaps.write_csv(gap1/"cross_model_ranking.csv",
                [dict(metric="zx_rms_total_l1_loss",full_coverage=1,
                      min_success_fraction=.5,mean_success_fraction=.5,multiway_better_than_true=0)])
            # Fake decoder generated Y has a deterministic DIFFERENT response
            # and a genuine z*x interaction; tests data-only estimator route.
            def fake_numeric_channels(t,x,z,model,checkpoint,device="cpu",batch_size=32):
                self.assertTrue(Path(checkpoint).is_file())
                ns,nt,p=x.shape
                eta,prob=values(np.repeat(t,nt,axis=0),x.reshape(-1,p),model)
                eta=eta.reshape(ns,nt);prob=prob.reshape(ns,nt)
                eta_nn=eta + .3*z[:,0,None]*x[:,:,0]
                p_nn=expit(eta_nn)
                return np.stack((eta,prob,eta_nn,p_nn),axis=-1)
            output=root/"output"
            with (mock.patch.object(run_gaps,"ROOT",root),
                  mock.patch.object(run_gaps,"numeric_channels",fake_numeric_channels)):
                run_gaps.main(["--models","C1","--gap1-dir",str(gap1),
                    "--output",str(output),"--train-subjects","9",
                    "--validation-subjects","4","--test-subjects","4",
                    "--trials-per-subject","5","--methods","sieve","rbf",
                    "--degree-grid","2","--ridge-grid","0.00001",
                    "--bandwidth-grid","1.8","--rbf-ridge-grid","0.0001",
                    "--landmarks","10","--angles","0,90",
                    "--planes","12","--plots","0","--top-per-model","1",
                    "--top-global","1"])
            files=list(output.glob("C1/*/threeway_landscape_comparisons.csv"))
            self.assertEqual(len(files),1)
            path=files[0].parent
            with (path/"threeway_curves.csv").open(newline="",encoding="utf-8") as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),2*2*2*3)
            self.assertEqual({r["source"] for r in rows},
                             {"analytic_T_hat","gap2_formula_Y","gap3_nn_Y"})
            self.assertEqual({r["method"] for r in rows},{"sieve","rbf"})
            self.assertTrue((output/"cross_model_threeway_ranking.csv").is_file())
            self.assertTrue((path/"sieve__test_hessians.npz").is_file())
            meta=json.loads((path/"metadata.json").read_text())
            self.assertTrue(meta["no_neural_autograd"])
            self.assertGreater(meta["Y_test_agreement"]["formula_vs_nn_logit_rmse"],0)


if __name__=="__main__":
    unittest.main()
