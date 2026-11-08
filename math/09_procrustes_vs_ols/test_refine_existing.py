"""Synthetic tests for second-stage optimal orthogonal+scale refinement.

Tests preserve the original OLS Oracle and verify test labels do not enter
the fitted optimizer. Tests are NOT measurements of empirical C-model performance.
"""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import evaluate, procrustes, prepare
from refine_existing import fit_joint, run_analysis, profile


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(153)

    def test_profiled_objective(self):
        d,n=4,200
        z=self.rng.normal(size=(n,d))
        t=self.rng.normal(size=(n,d))
        Q,_=procrustes(z,t)
        G=z.T@z/n;C=z.T@t/n;y2=float(np.sum(t*t)/n)
        loss,scales=profile(Q,G,C,y2)
        self.assertAlmostEqual(loss,float(np.mean((z@(Q*scales[None,:])-t)**2)),places=10)

    def test_joint_can_recover_orthogonal_plus_scale(self):
        rng=self.rng;d,n=4,500
        z=rng.normal(size=(n,d))
        Q_true,_=np.linalg.qr(rng.normal(size=(d,d)))
        t=z@(Q_true*np.array([.3,1.3,2.1,-.8])[None,:])
        Q0,_=procrustes(z,t)
        B_ols=np.linalg.lstsq(z,t,rcond=None)[0]
        solution=fit_joint(z,t,Q0,B_ols,restarts=1,maxiter=250)
        self.assertLess(solution['train_loss'],1e-8)
        np.testing.assert_allclose(solution['Q'].T@solution['Q'],np.eye(d),atol=1e-8)
        np.testing.assert_allclose(z@solution['B'],t,atol=1e-3)

    def test_immutable_ols_and_test_no_leakage(self):
        rng=self.rng;d=4;n=320
        z=rng.normal(size=(n,d))
        b=np.eye(d)*1.4 + rng.normal(0,.12,size=(d,d))
        t=z@b + rng.normal(0,.08,size=(n,d))
        ids=dict(train=np.arange(210),validation=np.arange(210,260),
                 test=np.arange(260,n))
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            source=p/'stage1'
            evaluate('CR3',z,t,ids,source,source='synthetic',plots=False)
            with np.load(source/'mappings_and_U.npz',allow_pickle=False) as f:
                original=f['ols_affine__B'].copy()
            s1=run_analysis('CR3',source,p/'refine1',plots=False,restarts=0,maxiter=60)
            expected=s1['models']['direct_ols_oracle']['B']
            np.testing.assert_array_equal(expected,original)
            sol=s1['joint_fit']
            self.assertGreaterEqual(s1['models']['procrustes_od_scaled']
                                     ['train_metrics']['mean_r2']-(-1),0)
            # Optimization uses TRAIN only; changing TEST T cannot affect Q.
            t2=t.copy()
            t2[ids['test']]=rng.normal(0,7,size=(len(ids['test']),d))
            source2=p/'stage1_second'
            evaluate('CR3',z,t2,ids,source2,source='synthetic',plots=False)
            s2=run_analysis('CR3',source2,p/'refine2',plots=False,restarts=0,maxiter=60)
            np.testing.assert_allclose(s1['joint_fit']['Q'],s2['joint_fit']['Q'],atol=1e-8)
            np.testing.assert_allclose(s1['models']['direct_ols_oracle']['B'],
                                       s2['models']['direct_ols_oracle']['B'],atol=1e-10)
            self.assertNotAlmostEqual(s1['models']['direct_ols_oracle']['test_metrics']['mean_r2'],
                                      s2['models']['direct_ols_oracle']['test_metrics']['mean_r2'],places=3)


if __name__=='__main__':
    unittest.main()
