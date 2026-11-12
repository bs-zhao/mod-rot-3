import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mobius_function_profiles as mobius
import shapley_function_profiles as shapley


def test_full_conditional_dividends_reconstruct_and_reallocate_to_exact_shapley():
    rng=np.random.default_rng(897)
    for d in (4,6):
        values=rng.normal(size=(2**d,13,7));values[0]=0.
        parts=mobius.dividends(values)
        np.testing.assert_allclose(parts.sum(0),values[-1],atol=1e-13)
        expected=shapley.shares_from_coalitions(dict(enumerate(values)),d)
        np.testing.assert_allclose(mobius.allocate_shapley(parts),expected,atol=1e-13)
        np.testing.assert_allclose(mobius.dividends(torch.tensor(values)).numpy(),parts,atol=1e-13)


def test_sparse_pair_interaction_is_retained_as_its_own_hyperedge():
    rng=np.random.default_rng(898);parts=np.zeros((16,20,3));parts[1]=rng.normal(size=(20,3))
    parts[6]=rng.normal(size=(20,3))
    values=np.zeros_like(parts)
    for mask in range(16):
        values[mask]=sum(parts[sub] for sub in range(16) if sub&mask==sub)
    np.testing.assert_allclose(mobius.dividends(values),parts,atol=1e-14)
