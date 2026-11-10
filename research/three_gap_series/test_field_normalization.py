import unittest
import numpy as np
from strict_hessian import HessianLoss


class FieldNormalizationTests(unittest.TestCase):
    def test_point_normalization_measures_pattern_independent_of_local_amplitude(self):
        rng=np.random.default_rng(911);d=3;p=2;n=40
        h=rng.normal(size=(n,d+p,d+p));h=(h+h.transpose(0,2,1))/2
        fields=dict(h=h,strict_x=rng.normal(size=(n,p)),strict_subject_groups=np.repeat([0,1],20))
        b=rng.normal(size=(d,d));cov=b@b.T+np.eye(d);q,_=np.linalg.qr(rng.normal(size=(d,d)))
        metric='hs_b09_b02_unique_edges_soft_t0.02_point'
        original=HessianLoss(fields,cov,d,metric)
        changed=dict(fields,h=h*rng.uniform(.5,2,(n,1,1)))
        self.assertAlmostEqual(original(q),HessianLoss(changed,cov,d,metric)(q),places=11)


if __name__=='__main__':unittest.main()
