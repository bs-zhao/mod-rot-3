"""S4 original-Z posterior distributions and exact per-axis diagonal KL tests.

All tests synthesize S5 arrays and do not retrain or re-extract an NP.
"""
from pathlib import Path
import json
import runpy

import numpy as np
import pytest


def _namespace():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1]/"s4_evaluate.py")
    )


def _data(n=32, dim=8):
    rng=np.random.default_rng(20261106)
    means=rng.normal(size=(n,dim))*.12
    means[:,0]+=2.0
    logvar=np.full((n,dim),-1.0)
    logvar[:,0]=0.
    splits=dict(train=np.arange(0,20),
                validation=np.arange(20,26),
                test=np.arange(26,n))
    return means,logvar,splits


def test_exact_kl_and_population_moments():
    stats=_namespace()["_posterior_z_statistics"]
    means,logvar,splits=_data()
    rows,summary=stats(means,logvar,splits)
    assert len(rows)==8
    assert summary["latent_dim"]==8
    assert summary["split_sizes"]=={"train":20,"validation":6,"test":6}
    assert np.isclose(rows[0]["mean"],means[:,0].mean())
    assert np.isclose(rows[0]["variance"],means[:,0].var(ddof=0))
    assert np.isclose(rows[0]["posterior_variance_mean"],1.0)
    expected=.5*np.mean(means[:,0]**2)
    assert np.isclose(rows[0]["mean_kl_nats"],expected)
    assert np.isclose(rows[1]["posterior_variance_mean"],np.exp(-1))
    assert np.isclose(rows[1]["mean_kl_nats"],
                      .5*np.mean(means[:,1]**2+np.exp(-1)))
    assert np.isclose(sum(x["mean_kl_nats"] for x in rows),
                      summary["mean_total_kl_nats"])
    assert 0 <= summary["train_mean_pca_participation_ratio"] <= 8
    assert len(summary["train_mean_pca_eigenvalues"])==8
    correlation=summary["posterior_mean_correlations"]
    assert set(correlation)=={"all","train","validation","test"}
    matrix=np.asarray(correlation["all"])
    assert matrix.shape==(8,8)
    assert np.allclose(matrix,matrix.T)
    assert np.allclose(np.diag(matrix),1)
    assert np.allclose(matrix,np.corrcoef(means,rowvar=False))
    assert np.isclose(sum(summary["train_mean_pca_explained_variance_ratio"]),1)


def test_zero_mean_unit_var_posterior_has_zero_kl():
    stats=_namespace()["_posterior_z_statistics"]
    means=np.zeros((10,4),float)
    logvar=np.zeros_like(means)
    splits=dict(train=np.arange(6),validation=np.arange(6,8),
                test=np.arange(8,10))
    rows,summary=stats(means,logvar,splits)
    assert all(row["mean_kl_nats"]==0 for row in rows)
    assert all(row["variance"]==0 for row in rows)
    assert all(row["posterior_variance_mean"]==1 for row in rows)
    assert summary["train_mean_pca_participation_ratio"]==0
    assert not np.any(summary["posterior_mean_correlations"]["all"])


def test_z_histogram_artifacts_and_checkpoint_validation(tmp_path):
    ns=_namespace()
    save=ns["_save_z_diagnostics"]
    means,logvar,splits=_data()
    folder=tmp_path/"s3_evaluate"
    folder.mkdir()
    latent=tmp_path/"s4_evaluate__latent.npz"
    np.savez_compressed(
        latent,z=means,z_logvar=logvar,context_mode=np.array("all_observed_trials"),
        checkpoint_sha256=np.array("ck-ok"),dataset_sha256=np.array("data-ok"),
        train_ids=splits["train"],validation_ids=splits["validation"],
        test_ids=splits["test"])
    save.__globals__["latent_path"]=lambda args:latent
    save.__globals__["sha256"]=lambda path:"latent-sha"
    args=type("Args",(),dict(model="CR3",beta=0.0005,latent_dim=8))()
    state=dict(model_config=dict(latent_dim=8),subject_splits=splits)
    diagnostics=save(args,folder,"ck-ok","data-ok",state)
    assert diagnostics["status"]=="complete"
    for file in ("s3_evaluate__z_dimension_histograms.png",
                 "s3_evaluate__z_variance_and_kl.png",
                 "s3_evaluate__z_pca_spectrum.png",
                 "s3_evaluate__z_dimension_stats.csv",
                 "s3_evaluate__z_diagnostics.json",
                 "s3_evaluate__z_correlation_heatmap.png",
                 "s3_evaluate__z_correlations_all.csv",
                 "s3_evaluate__z_correlations_train.csv",
                 "s3_evaluate__z_correlations_validation.csv",
                 "s3_evaluate__z_correlations_test.csv"):
        assert (folder/file).is_file(),file
        assert (folder/file).stat().st_size>0
    report=json.loads((folder/"s3_evaluate__z_diagnostics.json").read_text())
    assert len(report["per_dimension"])==8
    assert np.asarray(report["posterior_mean_correlations"]["all"]).shape==(8,8)
    import csv
    with (folder/"s3_evaluate__z_correlations_all.csv").open(newline="") as fp:
        matrix=list(csv.reader(fp))
    assert matrix[0]==["dimension",*["z"+str(j+1) for j in range(8)]]
    assert len(matrix)==9 and len(matrix[1])==9
    assert np.isclose(float(matrix[1][1]),1)
    with pytest.raises(RuntimeError, match="different S3 checkpoint"):
        save(args,folder,"different","data-ok",state)
    with pytest.raises(RuntimeError, match="different S1 dataset"):
        save(args,folder,"ck-ok","different",state)


def test_missing_s5_is_optional_and_invalid_inputs_rejected(tmp_path,capsys):
    ns=_namespace()
    save=ns["_save_z_diagnostics"]
    save.__globals__["latent_path"]=lambda args:tmp_path/"missing.npz"
    args=type("Args",(),dict(model="CR3",beta=0.0005,latent_dim=8))()
    got=save(args,tmp_path,"ck","data",{})
    assert got["status"]=="skipped_missing_s5"
    assert "Run np/s5_extract_z.py" in capsys.readouterr().out
    stats=ns["_posterior_z_statistics"]
    means,lv,ids=_data()
    with pytest.raises(ValueError,match="Non-finite"):
        invalid=means.copy();invalid[0,0]=np.nan
        stats(invalid,lv,ids)
    with pytest.raises(ValueError,match="not disjoint"):
        invalid_ids=dict(ids);invalid_ids["validation"]=ids["train"][:6]
        stats(means,lv,invalid_ids)
