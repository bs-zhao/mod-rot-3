"""CR4-specific regression tests; all legacy model inputs stay unchanged."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import expit
from modrot3.cli import parse_args
from modrot3.analysis import param_names
from modrot3 import pipeline
from modlearn_rot.generators.dataset import GeneratedDataset
from modlearn_rot.generators.registry import get_model

REF = Path(__file__).resolve().parents[1] / "reference" / "cr4"
sys.path.insert(0, str(REF))
from s0_s1_cr4 import (sample_bank, sample_theta, theta_as_cr3, logits,
                      verify_bank, PARAMETER_NAMES)
from s2_oracle_cr4 import gradient, fit_subject, BOUNDS3
from s0_cr2_design_recovery import logits as old_cr2_logits


def test_random_bank_is_reproducible_interior_and_has_no_special_strata():
    x = sample_bank(seed=20261023)
    assert x.shape == (500,6)
    assert verify_bank(x)
    assert np.array_equal(x,sample_bank(seed=20261023))
    assert np.all(x[:, [0,1,3,4]] > 0)
    assert np.all((x[:, [2,5]] > 0) & (x[:, [2,5]] < 1))
    assert not np.any(x[:, 2] == x[:, 5])
    assert not np.any((x[:, 2] == .5) & (x[:, 5] == .5))
    assert not np.any(x[:, [0,1,3,4]] == 0)


def test_cr4_formula_matches_original_cr3_at_alpha_one():
    x = sample_bank(31, n=20)
    theta = sample_theta(6, np.random.default_rng(33))
    assert theta.shape == (6,3)
    assert np.array_equal(theta_as_cr3(theta)[:, 0], np.ones(6))
    predicted = logits(x[None,:,:],theta[:,None,:])
    original = old_cr2_logits(x[None,:,:],theta_as_cr3(theta)[:,None,:])
    np.testing.assert_allclose(predicted, original, atol=1e-12, rtol=0)


def test_oracle_gradient_matches_finite_difference():
    x = sample_bank(53,n=100)
    theta = np.array([2.1, .74, 2.2])
    analytic = gradient(x, theta)
    numeric = np.column_stack([
        (logits(x,theta+np.eye(3)[j]*1e-6)
         -logits(x,theta-np.eye(3)[j]*1e-6))/(2e-6)
        for j in range(3)
    ])
    np.testing.assert_allclose(analytic,numeric,rtol=1e-5,atol=1e-8)


def test_bounded_oracle_mle_is_finite_without_special_trials():
    x = sample_bank(17)
    theta = np.array([2.0,.7,2.2])
    p = expit(logits(x, theta))
    y = np.random.default_rng(99).binomial(1,p)
    estimated = fit_subject(x,y,2,np.random.default_rng(50))
    assert estimated.shape == (3,)
    assert np.isfinite(estimated).all()
    assert np.all(estimated >= BOUNDS3[:,0] - 1e-8)
    assert np.all(estimated <= BOUNDS3[:,1] + 1e-8)


def test_cr4_training_adapter_standardizes_decoder_x_only_for_cr4(tmp_path,monkeypatch):
    bank = sample_bank(24)
    path = tmp_path / "CR4_3P_random500.npz"
    np.savez_compressed(path,X=bank)
    monkeypatch.setattr(pipeline,"_cr4_bank",lambda:path)
    mu,sd=pipeline.cr4_input_scaler()
    n=3
    trial=np.broadcast_to(bank,(n,500,6)).reshape(-1,6).copy()
    data=GeneratedDataset(parameters=np.ones((n,3)),
          trial_counts=np.full(n,500),
          subject_index=np.repeat(np.arange(n),500),
          trial_index=np.tile(np.arange(500),n),
          x=trial,
          y=np.zeros((n*500,1),dtype=int),
          predictive_true=np.full((n*500,1),.5))
    cooked=pipeline._discrete_data(SimpleNamespace(model="CR4"),data)
    actual=cooked.x.numpy()
    np.testing.assert_allclose(actual[0],(bank-mu)/sd,atol=2e-7)
    np.testing.assert_allclose(actual.mean(axis=(0,1)),np.zeros(6),atol=1e-6)
    np.testing.assert_allclose(actual.std(axis=(0,1)),np.ones(6),atol=1e-6)
    np.testing.assert_array_equal(data.x,trial)
    assert cooked.choice.shape==(n,500)


def test_cr4_registered_without_changing_legacy_config():
    args=parse_args("test",["--model","CR4","--latent-dim","3"])
    assert args.model=="CR4" and args.latent_dim==3
    assert args.data_seed==20261028
    assert tuple(param_names("CR4"))==PARAMETER_NAMES
    assert pipeline.out_dir("CR4").name=="CR4"
    for name in ("C1","C2","C3"):
        assert get_model(name).model_id==name
    assert parse_args("test",["--model","CR3"]).model=="CR3"
