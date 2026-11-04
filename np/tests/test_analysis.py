"""Baseline PCA/Ridge recovery test, isolated from rotations."""
import numpy as np
from modrot3.analysis import ridge_probe, corr

def test_ridge_recovery_heldout():
    rng = np.random.default_rng(10)
    z = rng.normal(size=(150,4))
    theta = z @ rng.normal(size=(4,4))
    r = ridge_probe(z, theta, np.arange(100), np.arange(100,125), np.arange(125,150))
    assert min(r["test_r2"]) > .9999
    assert corr(theta,z).shape == (4,4)
