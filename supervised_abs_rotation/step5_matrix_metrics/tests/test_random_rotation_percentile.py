import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from random_rotation_percentile import summarize
from tree_rotation_stability import random_rotation


def test_zero_favorable_draws_have_exact_simultaneous_binomial_bound():
    result=summarize(np.arange(64.)+2,1.)
    assert result['random_at_least_as_good_as_reference']==0
    expected=1-(.05/4)**(1/64)
    np.testing.assert_allclose(result['random_favorable_tail_upper_percent']/100,expected,atol=1e-15)
    assert result['within_15_percent_random_tail_bound']


def test_ties_do_not_make_a_flat_indicator_look_like_a_good_rank():
    result=summarize(np.ones(64),1.)
    assert result['random_numerical_ties']==64
    assert result['random_at_least_as_good_as_reference']==64
    assert result['random_favorable_tail_upper_percent']==100
    assert not result['within_15_percent_random_tail_bound']


def test_nonzero_count_bound_inverts_the_binomial_tail_and_draws_are_proper_rotations():
    result=summarize(np.r_[np.zeros(3),np.full(61,2.)],1.)
    upper=result['random_favorable_tail_upper_percent']/100
    np.testing.assert_allclose(binom.cdf(3,64,upper),.05/4,atol=1e-14)
    rng=np.random.default_rng(541)
    for d in (4,6):
        q=random_rotation(d,rng)
        np.testing.assert_allclose(q.T@q,np.eye(d),atol=1e-14)
        np.testing.assert_allclose(np.linalg.det(q),1,atol=1e-14)
