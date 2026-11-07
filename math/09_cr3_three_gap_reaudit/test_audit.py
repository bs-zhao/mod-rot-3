"""Regression tests: 07's relaxed tolerance is NOT exact endpoint recovery."""
from __future__ import annotations
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import PLANES,near,audit


class TestThreeGapAudit(unittest.TestCase):
    def test_near_is_not_exact(self):
        a=[0,5,10,15,80,90]
        np.testing.assert_array_equal(near(a,10),[1,1,1,0,1,1])
        np.testing.assert_array_equal(near(a,0),[1,0,0,0,0,1])

    def test_sequential_gates(self):
        rows=[]
        for target in ('logit','probability'):
            for plane in PLANES:
                rows.append(dict(target=target,plane=plane,metric='toy_loss',
                    analytic_min_angle=0.,formula_min_angle=0.,nn_min_angle=40.,
                    analytic_relative_variation=.5,formula_relative_variation=.5,
                    nn_relative_variation=.5,
                    analytic_formula_curve_pearson=.99,
                    analytic_nn_curve_pearson=.1))
        d,rank=audit(pd.DataFrame(rows))
        self.assertEqual(len(d),12)
        self.assertTrue((rank.gap1_near==6).all())
        self.assertTrue((rank.gap2_near==6).all())
        self.assertTrue((rank.gap3_near==0).all())

    def test_flat_or_low_correlations_not_pass(self):
        rows=[]
        for plane in PLANES:
            rows.append(dict(target='logit',plane=plane,metric='bad_loss',
                analytic_min_angle=0.,formula_min_angle=0.,nn_min_angle=0.,
                analytic_relative_variation=.5,formula_relative_variation=.001,
                nn_relative_variation=.5,
                analytic_formula_curve_pearson=.2,
                analytic_nn_curve_pearson=.99))
        _,rank=audit(pd.DataFrame(rows))
        self.assertEqual(int(rank.gap1_near.iloc[0]),6)
        self.assertEqual(int(rank.gap2_near.iloc[0]),0)
        self.assertEqual(int(rank.gap3_near.iloc[0]),0)


if __name__=='__main__':
    unittest.main()
