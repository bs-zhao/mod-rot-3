"""Synthetic proof checks for Stage-3 polar decomposition.

Numerical tests do NOT infer any real CR3 experimental effects.
"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import evaluate, MODELS
from polar_audit import analyze, polar_matrices, bootstrap_methods

class PolarProofTests(unittest.TestCase):
    def test_identity_exact_for_four_model_dimensions(self):
        rng=np.random.default_rng(2310)
        for model,d in MODELS.items():
            with self.subTest(model=model):
                n=1200
                z=rng.normal(size=(n,d))
                mixing=rng.normal(size=(d,d))
                t=z@mixing + rng.normal(0,.2,size=(n,d))
                tr=np.arange(800)
                # Mimic Stage-1 TRAIN-only standardization.
                z=(z-z[tr].mean(0))/z[tr].std(0)
                t=(t-t[tr].mean(0))/t[tr].std(0)
                cov=z[tr].T@z[tr]/len(tr)
                eig,v=np.linalg.eigh(cov)
                whitening=(v/np.sqrt(eig))@v.T
                pol=polar_matrices(z,t,tr,whitening)
                self.assertAlmostEqual(pol['train_mse_gap'],pol['exact_gap'],places=10)
                self.assertLess(np.linalg.norm(pol['Q'].T@pol['Q']-np.eye(d)),1e-8)
                self.assertLess(np.linalg.norm(pol['Q']@pol['H']-pol['C']),1e-8)
                self.assertGreaterEqual(pol['exact_gap'],-1e-13)

    def test_saved_oracle_and_paired_bootstrap(self):
        rng=np.random.default_rng(2311)
        d=4;n=450
        z=rng.normal(size=(n,d))
        Q,_=np.linalg.qr(rng.normal(size=(d,d)))
        t=z@(Q*np.array([.5,1.2,1.9,.6])[None,:])+rng.normal(0,.1,size=(n,d))
        splits=dict(train=np.arange(300),
                    validation=np.arange(300,360),
                    test=np.arange(360,n))
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'stage1'
            stage3=Path(tmp)/'stage3'
            evaluate('CR3',z,t,splits,source,source='synthetic',plots=False)
            results=analyze('CR3',source,stage3,bootstrap_reps=30,seed=67,plots=False)
            self.assertTrue(results['checks']['ols_reconstruction'])
            self.assertTrue(results['checks']['msegap_exact'])
            self.assertEqual(len(results['test_methods']),3)
            self.assertTrue((stage3/'bootstrap_metric_deltas_vs_ols.csv').exists())
            self.assertTrue((stage3/'H_symmetric_scale_mixing.csv').exists())
            self.assertLess(results['geometry']['H_offdiag_norm_ratio'],.2)
            b=results['geometry']
            self.assertAlmostEqual(b['train_mse_gap_observed'],
                                   b['train_mse_gap_theory'],places=10)
            # Bootstrap evaluates frozen TEST predictions; no change to the fit.
            data=np.load(source/'mappings_and_U.npz')
            zstd=data['Z_std'];tstd=data['T_std'];testids=data['test_ids']
            preds={'direct_ols_oracle':data['ols_affine__U_std']}
            rows,detail=bootstrap_methods(preds,tstd,testids,reps=20,seed=7)
            self.assertTrue(all(abs(r['diff_mean'])<1e-12 for r in rows))
            self.assertTrue(all(abs(r['delta_test_r2'])<1e-12 for r in detail))
            data.close()

if __name__=='__main__':
    unittest.main()
