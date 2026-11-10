import unittest
import numpy as np
from strict_hessian import HessianLoss


class FunctionalRankTests(unittest.TestCase):
    def test_mixing_two_separable_hessian_effects_increases_soft_rank(self):
        rng=np.random.default_rng(817);ns=40;nx=12;z0=rng.normal(size=(ns,2));bank=np.linspace(-1,1,nx)
        z=np.repeat(z0,nx,axis=0);x=np.tile(bank,ns);h=np.zeros((len(x),3,3))
        # f=.5*z0^2*x + .25*z1^2*x^2 is a genuine scalar Hessian.
        h[:,0,0]=x;h[:,1,1]=.5*x*x
        h[:,0,2]=h[:,2,0]=z[:,0];h[:,1,2]=h[:,2,1]=z[:,1]*x
        fields=dict(h=h,strict_x=x[:,None],strict_z=z,strict_subject_groups=np.repeat(np.arange(ns),nx))
        loss=HessianLoss(fields,np.eye(2),2,'hs_b06_rank_degree1_ux_soft_t0.03')
        r=np.array([[1.,1.],[-1.,1.]])/np.sqrt(2)
        self.assertGreater(loss(r),loss(np.eye(2))+.5)


if __name__=='__main__':unittest.main()
