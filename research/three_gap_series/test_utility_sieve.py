import unittest
import numpy as np
import torch
from utility_sieve import stimulus_basis_numpy,stimulus_basis_torch

class StructuredSieveTests(unittest.TestCase):
    def test_option_antisymmetry_and_numpy_torch_agreement(self):
        x=np.array([[100,0,.5,0,0,.5],[50,20,.1,100,0,.9],[0,100,0,100,0,1],[80,40,.3,150,20,.7]],float)
        expected=stimulus_basis_numpy(x)
        actual=torch.vmap(stimulus_basis_torch)(torch.tensor(x)).numpy()
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        swapped=np.concatenate((x[:,3:],x[:,:3]),axis=1)
        np.testing.assert_allclose(stimulus_basis_numpy(swapped),-expected,atol=1e-12)

if __name__=='__main__':unittest.main()
