import itertools
import unittest
import numpy as np
from taylor_hessian import fit_hessian_taylor,taylor_features


class TaylorHessianTests(unittest.TestCase):
    def test_reconstructs_unique_cubic_monomial_using_only_hessian_samples(self):
        rng=np.random.default_rng(112);ns=30;nx=8;d=3;n=ns*nx
        z=np.repeat(rng.normal(size=(ns,d)),nx,axis=0);x=np.tile(rng.normal(size=(nx,d)),(ns,1))
        h=np.zeros((n,2*d,2*d))
        for i,j,k in itertools.permutations(range(d)):h[:,i,j]=z[:,k]
        for i in range(d):h[:,i,d+i]=h[:,d+i,i]=1.
        fields=dict(h=h,strict_z=z,strict_x=x,strict_subject_groups=np.repeat(np.arange(ns),nx))
        tensors=fit_hessian_taylor(fields,np.eye(d),d,2)
        for i,j,k in itertools.permutations(range(d)):np.testing.assert_allclose(tensors[2][:,i,j,k],1.,atol=1e-12)
        np.testing.assert_allclose(tensors[1],0,atol=1e-12)
        vector=taylor_features(tensors,np.eye(d),3)
        cubic=vector[-10:]
        self.assertEqual(np.count_nonzero(cubic>1e-10),1)
        self.assertAlmostEqual(cubic.max(),1.,places=12)


if __name__=='__main__':unittest.main()
