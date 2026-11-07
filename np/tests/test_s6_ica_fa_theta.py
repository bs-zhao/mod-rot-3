"""TRAIN-only PCA/ICA/4-factor FA vs TEST theta audits in S6."""
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import runpy

import numpy as np
import pytest


def _s6():
    return runpy.run_path(str(Path(__file__).resolve().parents[1]/"s6_pca.py"))


def test_unsupervised_ica_and_four_factor_analysis_have_correct_shapes():
    ns=_s6()
    rng=np.random.default_rng(2061)
    z=rng.normal(size=(100,4))
    z[:,1]=.4*z[:,0]+z[:,1]
    out=ns["fit_unsupervised_ica_fa"](
        z,np.arange(65),factor_count=4,seed=31,ica_max_iter=2000)
    assert out["ic_scores"].shape==(100,4)
    assert out["factors"].shape==(100,4)
    assert out["ica_components"].shape==(4,4)
    assert out["fa_components"].shape==(4,4)
    assert np.all(np.isfinite(out["ic_scores"]))
    assert np.all(np.isfinite(out["factors"]))
    assert np.allclose(out["train_mean"],z[:65].mean(0))
    assert np.allclose(out["train_std"],z[:65].std(0))
    with pytest.raises(ValueError,match="positive"):
        ns["fit_unsupervised_ica_fa"](z,np.arange(65),factor_count=0)


def test_association_reports_signed_test_only_and_all_scatter_panels(tmp_path):
    ns=_s6()
    rng=np.random.default_rng(31)
    n=70
    latent=rng.normal(size=(n,4))
    theta=rng.normal(size=(n,4))
    theta[:,0]=latent[:,1]*2
    theta[:,1]=-latent[:,2]
    test=np.arange(50,70)
    names=["alpha","lambda_loss","gamma","tau"]
    summary=ns["save_theta_component_diagnostics"](
        tmp_path,"pc",latent,theta,test,names,["PC1","PC2","PC3","PC4"])
    r=np.asarray(summary["test_pearson_r"])
    assert r.shape==(4,4)
    assert np.isclose(r[0,1],1.0,atol=1e-12)
    assert np.isclose(r[1,2],-1.0,atol=1e-12)
    assert summary["test_subjects"]==20
    for filename in ("s12__pc_theta_correlations.csv",
                     "s12__pc_theta_pairwise.csv",
                     "s12__pc_theta_heatmap.png",
                     "s12__pc_theta_scatter.png"):
        f=tmp_path/filename
        assert f.is_file() and f.stat().st_size>0
    with (tmp_path/"s12__pc_theta_correlations.csv").open(newline="") as file:
        rows=list(csv.DictReader(file))
    assert list(rows[0])==["parameter","PC1","PC2","PC3","PC4"]
    assert float(rows[1]["PC3"]) < -0.999


def test_full_s6_L4_writes_pc_ic_fa_outputs_no_retraining(tmp_path):
    ns=_s6()
    rng=np.random.default_rng(811)
    n=110
    tr=np.arange(70)
    va=np.arange(70,90)
    te=np.arange(90,110)
    z=rng.normal(size=(n,4))
    theta=np.stack([z[:,0],z[:,1],z[:,2],z[:,3]],axis=1)
    src=tmp_path/"frozen_s5.npz"
    src.write_bytes(b"fake")
    f=ns["pca_report"]
    state=f.__globals__
    state["load_frozen"]=lambda a:(z,theta,dict(train=tr,validation=va,test=te),src)
    state["sha256"]=lambda p:"frozen_hash"
    state["stage_folder"]=lambda a,stage:tmp_path/stage
    state["condition"]=lambda a:"synthetic_L4"
    args=SimpleNamespace(
        model="CR3",beta=.00005,latent_dim=4,
        pc_var_threshold=.05,z_var_threshold=.01,
        fa_factors=4,ica_seed=31,ica_max_iter=2000,
    )
    f(args)
    folder=tmp_path/"s12_z_diagnostics"
    for key in ("pc","ic","fa"):
        for ext in ("correlations.csv","pairwise.csv",
                    "heatmap.png","scatter.png"):
            fn=folder/f"s12__{key}_theta_{ext}"
            assert fn.is_file() and fn.stat().st_size>0,fn
        with (folder/f"s12__{key}_theta_correlations.csv").open(newline="") as fp:
            rows=list(csv.DictReader(fp))
        assert len(rows)==4
        assert len(rows[0])==5
        assert rows[0]["parameter"]=="alpha"
    for name in ("s12__ica.npz","s12__factor_analysis.npz"):
        assert (folder/name).is_file()
    with np.load(folder/"s12__ica.npz") as saved:
        assert saved["transformed_z"].shape==(n,4)
        assert str(saved["source_latent_sha256"])=="frozen_hash"
        assert np.all(np.array_equal(saved[k+"_ids"],v)
                      for k,v in (("train",tr),("validation",va),("test",te)))
    with np.load(folder/"s12__factor_analysis.npz") as saved:
        assert saved["transformed_z"].shape==(n,4)
        assert saved["components"].shape==(4,4)
    summary=json.loads((folder/"s12__component_theta_summary.json").read_text())
    assert summary["factor_analysis"]["n_factors"]==4
    assert summary["ica"]["n_components"]==4
    assert len(summary["pca_theta"]["test_pearson_r"])==4
    old=json.loads((folder/"s12__pca_summary.json").read_text())
    assert "theta_vs_pc_test_corr" in old
    assert "component_theta_analysis" in old
    # S7 consumes the ORIGINAL unchanged PCA schema.
    with np.load(folder/"s12__pca.npz") as saved:
        assert saved["transformed_z"].shape==(n,4)
        assert "explained_variance_ratio" in saved


def test_factor_count_clips_to_latent_dim():
    ns=_s6()
    rng=np.random.default_rng(12)
    z=rng.normal(size=(90,3))
    out=ns["fit_unsupervised_ica_fa"](
        z,np.arange(60),factor_count=4,seed=7,ica_max_iter=2000)
    assert out["factors"].shape==(90,3)
    assert out["ic_scores"].shape==(90,3)
