import unittest
import numpy as np
import torch
from core import formula
from inverse_control import design_tensors,eta_from_parameters

class KnownModelControlTests(unittest.TestCase):
    def test_vectorized_formula_with_zero_amounts_and_probability_endpoints(self):
        x=np.array([[100,0,.5,0,0,.5],[50,20,.1,100,0,.9],[0,100,0,100,0,1]],float)
        t=torch.tensor([[.7,1.8,.6,2.1],[.9,2.5,.8,1.5]],dtype=torch.float64)
        got=eta_from_parameters(t,design_tensors(x,'cpu',torch.float64))
        expected=torch.stack([torch.stack([formula('CR3',row,torch.tensor(trial)) for trial in x]) for row in t])
        torch.testing.assert_close(got,expected,atol=1e-12,rtol=1e-12)

if __name__=='__main__':unittest.main()
