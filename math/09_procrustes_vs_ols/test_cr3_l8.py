"""Synthetic checks for CR3 historical L8 -> T4 Oracle reproducibility.

Only synthetic data; these tests never assert empirical CR3 model performance.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
# recheck_cr3_l8 imports modrot3, so make the repository's np package importable.
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'np'/'src'))
from recheck_cr3_l8 import run_analysis, rectangular_procrustes


class HistoricalL8OracleTests(unittest.TestCase):
    def test_rectangular_orthogonality_and_leakage(self):
        rng=np.random.default_rng(231)
        n=520
        z=rng.normal(size=(n,8))*np.arange(1,9)[None,:]*.25 + np.arange(8)[None,:]*.7
        b=rng.normal(size=(8,4))
        t=z@b + np.array([1.4,2.2,.1,-.7])
        ids=dict(train=np.arange(320),validation=np.arange(320,400),test=np.arange(400,n))
        q=rectangular_procrustes(rng.normal(size=(320,8)),rng.normal(size=(320,4)))
        np.testing.assert_allclose(q.T@q,np.eye(4),atol=1e-10)
        with tempfile.TemporaryDirectory() as td:
            old=run_analysis(z,t,ids,Path(td)/'baseline',plots=False)
            self.assertEqual(len(old['methods']),6)
            r2=old['methods']['direct_ols_train']['test_metrics']['mean_r2']
            self.assertGreater(r2,.999999)
            self.assertGreater(old['methods']['s7_exact_ridge']['test_metrics']['mean_r2'],.999999)
            self.assertEqual(old['input_shapes']['z'],[n,8])
            self.assertEqual(old['input_shapes']['t'],[n,4])
            # Map back from standardized regression to original T units.
            A=np.asarray(old['matrices']['A_raw_Z_to_T'])
            intercept=np.asarray(old['matrices']['b_raw_Z_to_T'])
            np.testing.assert_allclose(z@A.T+intercept,t,atol=1e-9)
            np.testing.assert_allclose(A,b.T,atol=1e-9)

            modified=t.copy()
            modified[ids['test']]=rng.normal(size=(len(ids['test']),4))
            new=run_analysis(z,modified,ids,Path(td)/'perturbed',plots=False)
            np.testing.assert_allclose(new['matrices']['B_direct_ols_train'],
                                       old['matrices']['B_direct_ols_train'],atol=1e-12)
            self.assertLess(new['methods']['direct_ols_train']['test_metrics']['mean_r2'],.5)


if __name__=='__main__':
    unittest.main()
