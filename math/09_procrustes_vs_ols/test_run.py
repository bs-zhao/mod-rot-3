"""Synthetic unit tests requiring only NumPy and the Python standard library."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import prepare, procrustes, fit_all, evaluate, diagnostics, MODELS


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(41)

    def test_procrustes_is_orthogonal_and_proper(self):
        X=self.rng.normal(size=(600,4))
        reflection=np.diag([1.,1.,1.,-1.])
        Y=X@reflection
        Q,_=procrustes(X,Y)
        self.assertLess(np.linalg.norm(Q.T@Q-np.eye(4)),1e-10)
        self.assertLess(np.linalg.norm(X@Q-Y),1e-9)
        self.assertLess(np.linalg.det(Q),0)
        Qso,_=procrustes(X,Y,proper=True)
        self.assertAlmostEqual(np.linalg.det(Qso),1.,places=10)
        self.assertGreater(np.linalg.norm(X@Qso-Y),0.01)

    def test_general_affine_shear_needs_ols(self):
        n=1400;d=4
        Z=self.rng.normal(size=(n,d)) + np.array([4,-1,2,.7])
        A=np.array([[2,0.2,0,0],[.1,.8,.3,0],[0,-.3,1.8,.1],[.5,0,.4,.6]])
        T=Z@A.T+np.array([8,-3,1,2])
        train=np.arange(1000)
        Zs,Ts,normal=prepare(Z,T,train)
        fitted=fit_all(Zs,Ts,train,normal['whitening'])
        self.assertLess(np.linalg.norm(Zs@fitted['ols_affine']['B']-Ts),1e-9)
        self.assertGreater(np.mean((Zs@fitted['procrustes_od']['B']-Ts)**2),.005)
        self.assertTrue(np.isfinite(Zs@fitted['whiten_procrustes_od']['B']).all())

    def test_all_four_models_and_holdout_no_leakage(self):
        for model,d in MODELS.items():
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                n=370
                z=self.rng.normal(size=(n,d))
                A=np.eye(d)*1.5 + np.tril(np.ones((d,d)),-1)*.12
                T=z@A + np.arange(d)*1.8+.5
                split=dict(train=np.arange(250),validation=np.arange(250,290),test=np.arange(290,n))
                first=evaluate(model,z,T,split,tmp,source='synthetic',plots=False)
                self.assertEqual(len(first['methods']),6)
                self.assertAlmostEqual(first['test_primary']['ols_affine']['mean_r2'],1.,places=10)
                self.assertEqual(len(list(Path(tmp).glob('corr__*__test.csv'))),6)
                saved=np.load(Path(tmp)/'mappings_and_U.npz')
                B0=saved['ols_affine__B'].copy();saved.close()
                T2=T.copy();T2[split['test']]=self.rng.normal(size=(len(split['test']),d))
                second=evaluate(model,z,T2,split,tmp,source='synthetic',plots=False)
                saved=np.load(Path(tmp)/'mappings_and_U.npz')
                np.testing.assert_allclose(saved['ols_affine__B'],B0)
                self.assertNotAlmostEqual(first['test_primary']['ols_affine']['mean_r2'],
                                          second['test_primary']['ols_affine']['mean_r2'],places=3)
                saved.close()


if __name__=='__main__':unittest.main()
