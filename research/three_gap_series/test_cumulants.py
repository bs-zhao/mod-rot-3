import itertools
import unittest
import numpy as np
from scipy.optimize import linear_sum_assignment
from cumulants import three_skew_directions

class CumulantControlTests(unittest.TestCase):
    def test_three_skewed_factors_with_independent_symmetric_noise(self):
        mixing=np.array([[1,.2,-.1,.3],[.1,1,.3,-.2],[.2,-.1,1,.1],[.3,.2,.1,1.]])
        points=[]
        for first in itertools.product([-.5,2.],repeat=3):
            weight=4**sum(v==-.5 for v in first)
            for symmetric in [-1.,1.]:
                points.extend([[*first,symmetric]]*weight)
        z=np.asarray(points)@mixing.T
        noise=np.array([.1,-.2,.15,.05])
        z=np.concatenate((z+noise,z-noise))
        directions,info=three_skew_directions(z,starts=12,seed=1)
        target=mixing[:,:3]/np.linalg.norm(mixing[:,:3],axis=0)
        match=abs(directions.T@target);i,j=linear_sum_assignment(-match)
        self.assertGreater(float(match[i,j].min()),.9999)
        self.assertLess(info['loss'],.001)

if __name__=='__main__':unittest.main()
