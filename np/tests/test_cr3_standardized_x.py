"""CR3 normalized-X checkpoint branch regression tests.

The S1 CR3 design and raw-X legacy checkpoints remain immutable. Only the
new default CR3 S3-S7 branch uses the exact CR4/5 bank-zscore policy.
"""
from types import SimpleNamespace
import numpy as np

from modrot3 import pipeline
from modrot3.cli import parse_args
from modlearn_rot.generators.dataset import GeneratedDataset


def _fixture_bank():
    rng=np.random.default_rng(123)
    x=np.empty((500,6),dtype=np.float64)
    x[:,[0,1,3,4]]=rng.uniform(0,200,(500,4))
    x[:,[2,5]]=rng.uniform(0,1,(500,2))
    x[0,1]=0; x[1,4]=0; x[2,2]=0; x[3,5]=1
    return x


def test_cr3_default_condition_is_xstd_and_legacy_remains_distinct():
    args=parse_args("test",["--model","CR3","--beta","0.0005",
                             "--latent-dim","4","--repeat","1"])
    assert pipeline.condition(args)=="beta0p0005_lambda0_ctx0p25to0p8_L4_xstd"
    p=pipeline.train_path(args)
    assert "_xstd" in str(p)
    assert "_xstd" in str(pipeline.latent_path(args))
    assert "s3_train" in str(p)
    assert pipeline.dataset_path("CR3")==pipeline.out_dir("CR3")/"s1_generate"/"s1_generate__dataset.npz"
    for model in ("CR4","CR5"):
        a=parse_args("test",["--model",model,"--beta","0.0005","--latent-dim","3"])
        assert pipeline.condition(a)=="beta0p0005_lambda0_ctx0p25to0p8_L3"
    assert not hasattr(args,"x_standardize")


def test_cr3_adapter_uses_same_raw_bank_zscore_as_cr4_cr5(tmp_path,monkeypatch):
    bank=_fixture_bank()
    path=tmp_path/"CR3_4P_fixed500.npz"
    np.savez_compressed(path,X=bank)
    monkeypatch.setattr(pipeline,"_cr3_bank",lambda:path)
    mu,sd=pipeline.cr3_input_scaler()
    n=5
    raw=np.broadcast_to(bank,(n,500,6)).reshape(-1,6).copy()
    theta=np.ones((n,4))
    data=GeneratedDataset(parameters=theta,
            trial_counts=np.full(n,500),
            subject_index=np.repeat(np.arange(n),500),
            trial_index=np.tile(np.arange(500),n),
            x=raw,y=np.zeros((n*500,1),dtype=np.int64),
            predictive_true=np.full((n*500,1),0.5))
    adapted=pipeline._discrete_data(SimpleNamespace(model="CR3"),data)
    assert adapted.x.shape==(n,500,6)
    np.testing.assert_allclose(adapted.x.numpy()[0],(bank-mu)/sd,
                               rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(adapted.x.numpy().mean(axis=1),np.zeros((n,6)),
                               atol=2e-6)
    assert np.array_equal(data.x,raw)
    assert np.any(data.x[:,1]==0) and np.any(data.x[:,[2,5]]==0)
    assert np.all(np.isfinite(adapted.x.numpy()))


def test_cr3_bank_transform_and_cr5_bank_transform_match_for_identical_x(tmp_path,monkeypatch):
    bank=_fixture_bank()
    a=tmp_path/"CR3.npz";b=tmp_path/"CR5.npz"
    np.savez_compressed(a,X=bank)
    np.savez_compressed(b,X=bank)
    monkeypatch.setattr(pipeline,"_cr3_bank",lambda:a)
    monkeypatch.setattr(pipeline,"_cr5_bank",lambda:b)
    a_mu,a_sd=pipeline.cr3_input_scaler()
    b_mu,b_sd=pipeline.cr5_input_scaler()
    assert np.array_equal(a_mu,b_mu) and np.array_equal(a_sd,b_sd)
