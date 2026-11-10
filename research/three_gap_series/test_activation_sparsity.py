import unittest
import numpy as np
from strict_hessian import HessianLoss


class ActivationSparsityTests(unittest.TestCase):
    def test_rare_interaction_has_smaller_support_at_equal_rms(self):
        n=40;h=np.zeros((n,3,3));h[:,0,2]=h[:,2,0]=1.
        rare=np.zeros_like(h);rare[:4,0,2]=rare[:4,2,0]=np.sqrt(10)
        meta=dict(strict_x=np.linspace(-1,1,n)[:,None],strict_subject_groups=np.repeat([0,1],20))
        for kind in ['hoyer','kurtosis']:
            metric=f'hs_b10_activation_pooled_{kind}_t0.03'
            dense_loss=HessianLoss(dict(meta,h=h),np.eye(2),2,metric)(np.eye(2))
            rare_loss=HessianLoss(dict(meta,h=rare),np.eye(2),2,metric)(np.eye(2))
            self.assertLess(rare_loss,dense_loss/5)


if __name__=='__main__':unittest.main()
