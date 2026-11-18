"""Regression guard: all rotation suites use EVERY subject, not NP splits.

This tests real loader behavior with a fake 500-trial-bank S1/S3/S5
source, where TRAIN-only and ALL-subject PCA centering are visibly different.
"""
from types import SimpleNamespace
from pathlib import Path
import numpy as np

from rotations.hessian_sparsity import data as hd
from rotations.cr4_functional_axes import data as fd


def _source(tmp_path):
    base=tmp_path/"outputs"/"choice"/"CR4"
    condition="beta0p0005_lambda0_ctx0p25to0p8_L3"
    ck=base/"s3_train"/condition/"repeat01"/"s3_train__checkpoint.pt"
    latent=base/"s4_evaluate"/condition/"repeat01"/"s4_evaluate__latent.npz"
    dataset=base/"s1_generate"/"s1_generate__dataset.npz"
    bank=base/"s0_design"/"CR4_3P_random500.npz"
    for path in (ck,latent,dataset,bank):
        path.parent.mkdir(parents=True,exist_ok=True)
    ck.write_bytes(b"known checkpoint")
    rng=np.random.default_rng(71)
    x=rng.normal(size=(500,6))
    np.savez_compressed(bank,X=x)
    subjects=40
    z=rng.normal(size=(subjects,3))
    z[20:]+=np.array([4.,-2.,3.])
    raw_x=np.broadcast_to(x,(subjects,500,6)).reshape(-1,6)
    np.savez_compressed(dataset,
        trial_counts=np.full(subjects,500),
        subject_index=np.repeat(np.arange(subjects),500),
        x=raw_x,parameters=rng.normal(size=(subjects,3)))
    tr=np.arange(20);va=np.arange(20,30);te=np.arange(30,40)
    np.savez_compressed(latent,z=z,
        dataset_sha256=hd.sha256(dataset),
        checkpoint_sha256=hd.sha256(ck),
        context_mode="all_observed_trials",
        train_ids=tr,validation_ids=va,test_ids=te)
    return condition,z,tr


def test_both_loaders_have_all_40_subjects_in_rotation_and_full_cohort_pca(
        tmp_path,monkeypatch):
    condition,z,tr=_source(tmp_path)
    monkeypatch.setattr(hd,"ROOT",tmp_path)
    monkeypatch.setattr(fd,"ROOT",tmp_path)
    h=hd.load_source("CR4",condition,1,3,dict(
        seed=12,fit_x=25,valid_x=20,whiten_floor=1e-5,
        grouping="pooled"))
    f=fd.load_source(SimpleNamespace(
        model="CR4",condition=condition,repeat=1,rotation_dim=3),
        dict(seed=12,fit_x=25,valid_x=20,whiten_floor=1e-5))
    for source in (h,f):
        np.testing.assert_array_equal(source["all_subject_ids"],np.arange(40))
        np.testing.assert_array_equal(source["fit_subject_ids"],np.arange(40))
        np.testing.assert_array_equal(source["valid_subject_ids"],np.arange(40))
        np.testing.assert_allclose(source["pca_mean"],z.mean(0),atol=1e-12)
        assert not np.allclose(source["pca_mean"],z[tr].mean(0),atol=.05)
        assert not set(source["fit_x_ids"]) & set(source["valid_x_ids"])
        assert source["rotation_subject_scope"]=="all_subjects_no_test"
        np.testing.assert_allclose(source["q"].mean(0),np.zeros(3),atol=1e-12)
