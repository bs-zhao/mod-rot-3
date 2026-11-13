import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from predictive_variance_effects import allocate


def test_variance_game_does_not_underweight_a_pure_pair_relative_to_a_main_effect():
    # Independent centered pair variance 2, independent main variance 1.
    # Oracle explained variance, not a function-family assumption in the scorer.
    raw=np.array([(2*bool(mask&1 and mask&2)+bool(mask&4))/3 for mask in range(8)])
    effects,game,audit=allocate(raw,3)
    np.testing.assert_allclose(effects,[1/3,1/3,1/3,0],atol=1e-15)
    np.testing.assert_array_equal(game,raw)
    assert audit['maximum_monotone_envelope_increase']==0


def test_estimator_nonmonotonicity_and_unexplained_variance_are_retained_in_audits():
    raw=np.array([0,.5,-.2,.4])
    effects,game,audit=allocate(raw,2)
    np.testing.assert_allclose(game,[0,.5,0,.5])
    np.testing.assert_allclose(effects,[.5,0,.5])
    assert audit['maximum_monotone_envelope_increase']>.09
    assert audit['fraction_negative_raw_skill']==.25


def test_multioutput_allocation_commutes_with_permuting_inputs():
    rng=np.random.default_rng(932);raw=rng.uniform(-.1,.9,(16,3,2));raw[0]=0
    perm=[2,0,3,1]
    old_masks=[sum(((mask>>i)&1)<<perm[i] for i in range(4)) for mask in range(16)]
    expected,_,_=allocate(raw,4);actual,_,_=allocate(raw[old_masks],4)
    np.testing.assert_allclose(actual,np.concatenate((expected[perm],expected[-1:])),atol=1e-15)
    np.testing.assert_allclose(actual.sum(0),1,atol=1e-15)
