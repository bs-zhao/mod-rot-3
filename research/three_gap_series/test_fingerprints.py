import unittest
import numpy as np
import torch
from core import formula
from fingerprints import fit_power_fingerprint,loss_mirror,fit_proportional_fingerprint

class AlgebraicFingerprintTests(unittest.TestCase):
    def test_power_exponent_with_signed_nuisance_amplitudes(self):
        rng=np.random.default_rng(15);scales=np.array([.6,.8,1.,1.2]);a=np.array([.3,.55,.8,1.1])
        amplitude=rng.normal(size=(4,30))
        values=scales[None,:,None]**a[:,None,None]*amplitude[:,None,:]
        got,error,bound=fit_power_fingerprint(values,scales)
        np.testing.assert_allclose(got,a,atol=1e-6)
        self.assertLess(error.max(),1e-10);self.assertEqual(bound,0)

    def test_gain_loss_mirror_is_a_proportional_response(self):
        rng=np.random.default_rng(6);x=rng.uniform(5,150,(20,6));x[:,[1,4]]=0;x[:,[2,5]]=rng.uniform(.1,.9,(20,2))
        mirror=loss_mirror(x);parameters=np.array([[.8,2,.7,2.2],[.6,1.4,.9,1.8]])
        def evaluate(bank):return np.array([[float(formula('CR3',torch.tensor(t),torch.tensor(trial))) for trial in bank] for t in parameters])
        lam,error=fit_proportional_fingerprint(evaluate(x),evaluate(mirror))
        np.testing.assert_allclose(lam,parameters[:,1],atol=1e-12)
        self.assertLess(error.max(),1e-20)

if __name__=='__main__':unittest.main()
