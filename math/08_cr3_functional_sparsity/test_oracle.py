"""Self-contained tests for CR3 derivative oracle and functional structure."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from oracle import sample_prior, outputs, derivatives
from run import rotation, matrix_rms, finite_effects, soft_l0

class TestOracle(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(1001)
        self.t=sample_prior(10,rng)
        self.x=np.column_stack([rng.uniform(10,180,10),rng.uniform(10,180,10),
                                rng.uniform(.1,.9,10),rng.uniform(10,180,10),
                                rng.uniform(10,180,10),rng.uniform(.1,.9,10)])

    def test_analytic_vs_finite_differences(self):
        t,x=self.t,self.x
        zz,zx=derivatives(t,x)
        h=5e-5
        for c in (0,1):
            for i in range(4):
                for j in range(4):
                    ti=np.zeros(4);tj=np.zeros(4);ti[i]=h;tj[j]=h
                    fd=(outputs(t+ti+tj,x)[:,c]-outputs(t+ti-tj,x)[:,c]
                        -outputs(t-ti+tj,x)[:,c]+outputs(t-ti-tj,x)[:,c])/(4*h*h)
                    np.testing.assert_allclose(zz[:,c,i,j],fd,atol=2e-5,rtol=1e-3)
                for j in range(6):
                    ti=np.zeros(4);dx=np.zeros(6);ti[i]=h;dx[j]=h
                    fd=(outputs(t+ti,x+dx)[:,c]-outputs(t+ti,x-dx)[:,c]
                        -outputs(t-ti,x+dx)[:,c]+outputs(t-ti,x-dx)[:,c])/(4*h*h)
                    np.testing.assert_allclose(zx[:,c,i,j],fd,atol=2e-5,rtol=2e-3)

    def test_rotation_equivariance_and_support(self):
        zz,zx=derivatives(self.t,self.x)
        st=np.std(self.t,axis=0);sx=np.std(self.x,axis=0)
        a,b=matrix_rms(zz,zx,rotation(0,1,0),st,sx)
        c,d=matrix_rms(zz,zx,rotation(0,1,45),st,sx)
        np.testing.assert_allclose(np.linalg.norm(b,axis=(1,2)),np.linalg.norm(d,axis=(1,2)),rtol=1e-12)
        # lambda cannot change gain value on logit scale: structural zero.
        self.assertEqual(b[0,1,0],0.)
        self.assertEqual(b[0,1,3],0.)
        self.assertGreater(d[0,1,0],0.)
        self.assertGreater(d[0,1,3],0.)
        self.assertTrue(np.all(c>=0))

    def test_function_value_contrast_and_soft_support(self):
        st=np.std(self.t,axis=0);sx=np.std(self.x,axis=0)
        selfv,pair=finite_effects(self.t,self.x,rotation(0,1,0),st,sx,.025)
        self.assertEqual(selfv.shape,(2,4))
        self.assertEqual(pair.shape,(2,4,6))
        self.assertLess(pair[0,1,0],1e-10)
        self.assertGreater(pair[0,0,0],1e-9)
        a,h,_=soft_l0(np.array([[2.,2.],[.001,2.]]),.1)
        b,k,_=soft_l0(np.array([[2.,2.2],[.5,.5]]),.1)
        self.assertEqual(h,3)
        self.assertEqual(k,4)
        self.assertLess(a,b)

if __name__=='__main__': unittest.main()
