"""Scientific invariants for the empirical projection experiment."""
import sys
from pathlib import Path

import numpy as np
import pytest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import additive_profiles as ap


@pytest.mark.parametrize("family", ["poly3", "poly4", "rbf3", "rbf4"])
def test_signed_permutation_has_same_projection_scores(family):
    rng = np.random.default_rng(19)
    u = rng.normal(size=(600, 3))
    target = np.column_stack([np.sin(u[:, 0]) + u[:, 1]*u[:, 2], u[:, 2]**2])
    split = (np.arange(360), np.arange(360, 480), np.arange(480, 600))
    a, _ = ap.projection_scores(u, target, split, family)
    b, _ = ap.projection_scores(u[:, [2, 0, 1]] * [-1, 1, -1], target, split, family)
    for key in a:
        assert b[key] == pytest.approx(a[key], abs=2e-7, rel=2e-7)


def test_additive_truth_requires_interactions_after_rotation():
    rng = np.random.default_rng(20)
    u = rng.normal(size=(800, 3))
    y = np.column_stack([u[:, 0]**3, u[:, 1]**2])
    split = (np.arange(480), np.arange(480, 640), np.arange(640, 800))
    angle = .5
    q = np.eye(3); q[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    base, _ = ap.projection_scores(u, y, split, "poly3")
    rot, _ = ap.projection_scores(u @ q, y, split, "poly3")
    assert base["additive_test_nmse"] < 1e-9
    assert rot["additive_test_nmse"] > .05
    assert base["pairwise_test_nmse"] < 1e-8
    assert rot["pairwise_test_nmse"] < 1e-8


def test_dependent_input_projection_does_not_invent_pair_effect():
    rng = np.random.default_rng(21)
    u = rng.normal(size=(800, 3)); u[:, 1] += .8*u[:, 0]
    y = (u[:, 0]**2 + .3*u[:, 1]**3)[:, None]
    split = (np.arange(480), np.arange(480, 640), np.arange(640, 800))
    scores, _ = ap.projection_scores(u, y, split, "poly3")
    assert scores["pair_projection_total_energy"] < 1e-12


def test_response_profiles_use_all_rows_and_hold_out_real_trials(tmp_path):
    rng = np.random.default_rng(22)
    n, trials, d = 30, 100, 3
    u = rng.normal(size=(n, d))
    sid = np.repeat(np.arange(n), trials)
    x = rng.normal(size=(n*trials, 2))
    y = (u[sid, 0]*x[:, 0] + u[sid, 1]*x[:, 1]**2)[:, None]
    np.save(tmp_path / "trial_index.npy", np.tile(np.arange(trials), n))
    data = SimpleNamespace(w=np.column_stack([u[sid], x]), y=y,
        d=d, n=n, m=n*trials, channels=("logit_nn",), folder=tmp_path,
        offsets=np.arange(n+1)*trials)
    profiles, audit = ap.x_profiles(data, centers=10)
    assert profiles.shape[0:2] == (n, 1)
    assert audit["all_subjects"] == n
    assert audit["all_rows"] == n*trials
    # Nonzero ridge leaves a small bias even for exactly represented data.
    assert audit["validation_nmse"][0] < 1e-6
    assert audit["sobol_independence_assumed"] is False
