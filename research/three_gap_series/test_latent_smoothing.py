import unittest
import numpy as np
from latent_smoothing import integrate_latent_hessian


class LatentSmoothingTests(unittest.TestCase):
    def test_subject_constant_hessian_keeps_its_information_at_all_bandwidths(self):
        rng=np.random.default_rng(149);ns=8;nx=6;d=3;p=2
        block=rng.normal(size=(nx,d+p,d+p));block=(block+block.transpose(0,2,1))/2
        fields=dict(h=np.tile(block,(ns,1,1)),strict_subject_groups=np.repeat(np.arange(ns),nx),
            strict_z=np.repeat(rng.normal(size=(ns,d)),nx,axis=0))
        for bandwidth in [.25,.5,1.,2.,'mean']:
            smooth=integrate_latent_hessian(fields,np.eye(d),d,bandwidth)
            np.testing.assert_allclose(smooth['h'],fields['h'],atol=1e-12)


if __name__=='__main__':unittest.main()
