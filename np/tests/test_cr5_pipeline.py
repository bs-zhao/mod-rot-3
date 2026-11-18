"""CR5 controlled-experiment regression tests (no real 5000-subject data required)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import expit
from modrot3.cli import parse_args
from modrot3.analysis import param_names
from modrot3 import pipeline
from modlearn_rot.generators.dataset import (GeneratedDataset,load_dataset)
from modlearn_rot.generators.registry import get_model

REF=Path(__file__).resolve().parents[1]/"reference"/"cr5"
sys.path.insert(0,str(REF))
import s0_s1_cr5 as cr5
from s2_oracle_cr5 import evaluate,fisher_diagnostics
from s0_s1_cr4 import sample_theta,theta_as_cr3,logits as cr4_logits
from s0_cr2_design_recovery import logits as cr3_logits


def fixture_bank():
    # Deliberately four disjoint 60+60+40+340 classes in CR3's native order,
    # with NO FOSD; each option has a higher payoff in a different state.
    X=[];code=[]
    groups=[
        (0,60,lambda i:[120+i*.01,0,.25,80+i*.01,0,.70]),
        (1,60,lambda i:[120+i*.01,100+i*.01,.50,80+i*.01,60+i*.01,.50]),
        (2,40,lambda i:[0,100+i*.01,.90,0,80+i*.01,.40]),
        (3,340,lambda i:[120+i*.01,100+i*.01,.35,80+i*.01,60+i*.01,.65]),
    ]
    for label,n,fn in groups:
        X.extend([fn(i) for i in range(n)])
        code.extend([label]*n)
    return np.asarray(X,dtype=np.float64),np.asarray(code,dtype=np.int64)


def create_temporary_cr3_and_cr5(tmp_path,monkeypatch):
    x,codes=fixture_bank()
    source=tmp_path/"CR3"/"s0_design"/"CR3_4P_fixed500.npz"
    source.parent.mkdir(parents=True)
    np.savez_compressed(source,X=x,type_codes=codes,type_names=np.asarray(list(cr5.COUNTS)),
                        parameter_names=np.asarray(["alpha","lambda_loss","gamma","tau"]))
    target=tmp_path/"CR5"/"s0_design"/"CR5_3P_cr3bank500.npz"
    manifest=target.parent/"CR5_3P_design.json"
    ds=tmp_path/"CR5"/"s1_generate"/"s1_generate__dataset.npz"
    monkeypatch.setattr(cr5,"CR3_BANK",source)
    monkeypatch.setattr(cr5,"BANK",target)
    monkeypatch.setattr(cr5,"DESIGN",manifest)
    monkeypatch.setattr(cr5,"DATASET",ds)
    return x,codes,source,target,ds


def test_cr5_formula_is_cr4_formula_and_cr3_alpha_one_even_at_zero():
    x,_=fixture_bank()
    theta=sample_theta(4,np.random.default_rng(31))
    actual=cr4_logits(x[None],theta[:,None])
    target=cr3_logits(x[None],theta_as_cr3(theta)[:,None])
    np.testing.assert_allclose(actual,target,rtol=0,atol=1e-12)
    assert np.isfinite(actual).all()


def test_cr5_design_is_exact_cr3_copy_and_dataset_has_three_params(tmp_path,monkeypatch):
    x,codes,cr3_bank,bank,dataset=create_temporary_cr3_and_cr5(tmp_path,monkeypatch)
    cr5.design(SimpleNamespace())
    original,labels,names,meta=cr5.validate_immutable_bank()
    assert np.array_equal(original,x)
    assert np.array_equal(labels,codes)
    assert meta["cr3_source_sha256"]==cr5.hash_file(cr3_bank)
    cr5.dataset(SimpleNamespace(subjects=4,data_seed=20261028))
    ds=load_dataset(dataset)
    assert ds.parameters.shape==(4,3)
    assert ds.x.shape==(2000,6)
    assert np.array_equal(ds.x.reshape(4,500,6),
                          np.broadcast_to(x,(4,500,6)))
    assert np.allclose(ds.predictive_true.reshape(4,500),
                       expit(cr4_logits(x[None],ds.parameters[:,None])))
    assert np.array_equal(ds.trial_counts,np.full(4,500))
    assert list(names)==list(cr5.COUNTS)
    manifest=json.loads((dataset.parent/"s1_generate__manifest.json").read_text())
    assert manifest["original_X_and_trial_order_equals_CR3"]


def test_cr5_bank_changes_trigger_hard_failure(tmp_path,monkeypatch):
    x,codes,source,bank,ds=create_temporary_cr3_and_cr5(tmp_path,monkeypatch)
    cr5.design(SimpleNamespace())
    with np.load(source,allow_pickle=False) as saved:
        copy={k:saved[k].copy() for k in saved.files}
    copy["X"][0,0] += 1
    np.savez_compressed(source,**copy)
    import pytest
    with pytest.raises(RuntimeError,match="(changed|deviates)"):
        cr5.validate_immutable_bank()


def test_cr5_adapter_standardizes_full_cr3_bank_without_changing_s1(tmp_path,monkeypatch):
    x,code,_,bank,_=create_temporary_cr3_and_cr5(tmp_path,monkeypatch)
    cr5.design(SimpleNamespace())
    monkeypatch.setattr(pipeline,"_cr5_bank",lambda:bank)
    n=3
    raw=np.broadcast_to(x,(n,500,6)).reshape(-1,6).copy()
    ds=GeneratedDataset(parameters=np.ones((n,3)),trial_counts=np.full(n,500),
        subject_index=np.repeat(np.arange(n),500),
        trial_index=np.tile(np.arange(500),n),
        x=raw,y=np.zeros((n*500,1),dtype=np.int64),
        predictive_true=np.full((n*500,1),.5))
    mu,sd=pipeline.cr5_input_scaler()
    cooked=pipeline._discrete_data(SimpleNamespace(model="CR5"),ds)
    np.testing.assert_allclose(cooked.x.numpy()[0],(x-mu)/sd,atol=2e-7)
    np.testing.assert_allclose(cooked.x.numpy().mean(axis=(0,1)),0,atol=1e-6)
    # Float32 accumulation over a repeated fixed bank incurs ~1e-5 rounding.
    np.testing.assert_allclose(cooked.x.numpy().std(axis=(0,1)),1,atol=2e-5)
    assert np.array_equal(ds.x,raw)
    assert np.any(ds.x[:,1]==0)  # genuine loss=0 trials have been retained


def test_cr5_oracle_fisher_and_parameter_recovery_are_same_cr4_estimator():
    x,_=fixture_bank()
    theta=sample_theta(8,np.random.default_rng(44))
    fisher=fisher_diagnostics(x,theta)
    assert fisher["full_rank_fraction"]>0
    result,truth,mle,y=evaluate(x,3,2,44)
    assert result["parameters"]==["lambda_loss","gamma","tau"]
    assert truth.shape==mle.shape==(3,3)
    assert np.isfinite(mle).all()
    assert y.shape==(3,500)


def test_cr5_registered_all_s1_s7_while_legacy_models_unchanged():
    a=parse_args("test",["--model","CR5","--latent-dim","3"])
    assert a.model=="CR5" and a.data_seed==20261028
    assert tuple(param_names("CR5"))==("lambda_loss","gamma","tau")
    assert pipeline.out_dir("CR5").name=="CR5"
    for name in ("CR3","CR4"):
        assert parse_args("test",["--model",name]).model==name
    for name in ("C1","C2","C3"):
        assert get_model(name).model_id==name



def test_cr5_oracle_handles_sure_gain_and_loss_endpoints():
    """CR5 copies CR3, whose 500-item bank can include p=0/1 exactly."""
    from s2_oracle_cr4 import gradient
    x, _ = fixture_bank()
    x = x.copy()
    x[0,2] = 0.
    x[1,2] = 1.
    x[2,5] = 0.
    x[3,5] = 1.
    ref = sample_theta(12,np.random.default_rng(82))
    with np.errstate(all="raise"):
        grads = gradient(x, ref[0])
        info = fisher_diagnostics(x, ref)
    assert grads.shape == (500,3)
    assert np.isfinite(grads).all()
    assert np.isfinite(info["min_eigenvalue_median"])
    assert info["full_rank_fraction"] > 0
