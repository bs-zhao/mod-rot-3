import unittest
import numpy as np
from strict_hessian import HessianLoss,CATALOG,penalty


class StrictHessianTests(unittest.TestCase):
    def test_tail_matches_the_history_examples(self):
        a=np.array([2.,2.,.01,.001]);b=np.array([1.,.5,.01,.01])
        for score in ['tail1','tail2sq','relative_tail2']:
            self.assertLess(penalty(a,score,k=2),penalty(b,score,k=2))

    def test_entire_catalogue_scale_sign_and_permutation_invariance(self):
        rng=np.random.default_rng(14);n=24;d=4;p=3
        h=rng.normal(size=(n,d+p,d+p));h=(h+h.transpose(0,2,1))/2
        fields=dict(h=h,strict_x=rng.normal(size=(n,p)),third=rng.normal(size=(n,d+p,d+p,d+p)),strict_subject_groups=np.zeros(n,dtype=int))
        q,_=np.linalg.qr(rng.normal(size=(d,d)));base=rng.normal(size=(d,d));cov=base@base.T+np.eye(d)
        changed=(np.array([.001,-20.,.3,-100.])[:,None]*q)[[3,1,0,2]]
        for name in CATALOG:
            loss=HessianLoss(fields,cov,d,name)
            self.assertAlmostEqual(loss(q),loss(changed),places=9,msg=name)

    def test_interaction_can_turn_into_self_curvature(self):
        d=2;n=20;h=np.zeros((n,3,3));h[:,0,1]=h[:,1,0]=1
        fields=dict(h=h,strict_x=np.linspace(-1,1,n)[:,None])
        r=np.array([[1.,1.],[-1.,1.]])/np.sqrt(2)
        off=HessianLoss(fields,np.eye(d),d,'hs_uu_off_rms_l1')
        own=HessianLoss(fields,np.eye(d),d,'hs_uu_self_rms_l1')
        self.assertLess(off(r),off(np.eye(d)))
        self.assertGreater(own(r),own(np.eye(d)))


if __name__=='__main__':unittest.main()
