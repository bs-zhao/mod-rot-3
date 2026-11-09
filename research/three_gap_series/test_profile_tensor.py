import unittest
import numpy as np
import torch
from profile_tensor import stimulus_library_numpy,stimulus_library_torch

class StimulusDictionaryTests(unittest.TestCase):
    def test_raw_and_comparison_dictionary_semantics(self):
        rng=np.random.default_rng(11)
        for model,width in [('C1',3),('C2',4),('C3',4)]:
            x=rng.uniform(-1,1,(10,width))
            expected=stimulus_library_numpy(model,x)
            got=torch.vmap(lambda x:stimulus_library_torch(model,x))(torch.tensor(x)).numpy()
            np.testing.assert_allclose(got,expected,atol=1e-14)
            if model!='C1':
                swapped=stimulus_library_numpy(model,x[:,[1,0,3,2]])
                np.testing.assert_allclose(swapped[:,1:],-expected[:,1:],atol=1e-14)
                np.testing.assert_equal(swapped[:,0],expected[:,0])

if __name__=='__main__':unittest.main()
