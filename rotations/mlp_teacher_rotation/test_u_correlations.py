"""Direct U-axis and theta-U pairwise heatmap/scatter tests."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from .u_correlations import (
    u_pearson_correlation,save_u_correlation_audit,
    save_u_pairwise_scatter,save_theta_u_pairwise_scatter,
)


def _fixture():
    rng=np.random.default_rng(20261106)
    n,k=48,4
    U=rng.normal(size=(n,k))
    U[:,2]=0.8*U[:,0]+0.5*U[:,2]
    split=(np.arange(0,28),np.arange(28,38),np.arange(38,48))
    theta=rng.normal(size=(len(split[2]),4))
    theta[:,0]+=U[split[2],0]
    return U,theta,split


def test_exact_pearson_all_axes_and_constant_axis_policy():
    U,_,_= _fixture()
    r=u_pearson_correlation(U)
    assert r.shape==(4,4)
    assert np.allclose(r,r.T,atol=1e-12)
    assert np.allclose(np.diag(r),1)
    assert np.allclose(r,np.corrcoef(U,rowvar=False),atol=1e-12)
    assert r[0,2]>0.5
    U[:,3]=2.
    r=u_pearson_correlation(U)
    assert np.allclose(r[3],0)
    assert np.allclose(r[:,3],0)
    assert np.all(np.isfinite(r))
    with pytest.raises(ValueError):
        u_pearson_correlation(np.array([1,2,3]))


def test_split_heatmaps_csvs_and_both_scatter_atlases(tmp_path):
    U,theta,(tr,va,te)=_fixture()
    info=save_u_correlation_audit(tmp_path,U,tr,va,te)
    assert set(info["splits"])=={"train","validation","test","all"}
    for name in ("train","validation","test","all"):
        fn=tmp_path/f"rotation__u_correlations_{name}.csv"
        assert fn.is_file()
        with fn.open(newline="",encoding="utf-8") as stream:
            rows=list(csv.reader(stream))
        assert rows[0]==["axis","u1","u2","u3","u4"]
        assert [row[0] for row in rows[1:]]==["u1","u2","u3","u4"]
        observed=np.array([[float(v) for v in row[1:]] for row in rows[1:]])
        sel={"train":tr,"validation":va,"test":te,"all":np.arange(len(U))}[name]
        assert np.allclose(observed,np.corrcoef(U[sel],rowvar=False))
    for name in ("rotation__test_u_corr.png","rotation__u_corr_by_split.png",
                 "rotation__test_u_pairwise_scatter.png",
                 "rotation__u_correlations.json"):
        path=tmp_path/name
        assert path.is_file() and path.stat().st_size>0
    c=np.corrcoef(np.concatenate([theta,U[te]],axis=1),rowvar=False)[:4,4:]
    save_theta_u_pairwise_scatter(
        tmp_path,theta,U[te],["a","b","c","d"],corr_matrix=c,
        matching={0:0,1:1,2:2,3:3})
    assert (tmp_path/"rotation__test_theta_u_pairwise_scatter.png").is_file()


def test_posthoc_existing_run_reuses_saved_u_without_refit(tmp_path):
    from .plot_existing import add_plots_to_existing_run
    U,theta,(tr,va,te)=_fixture()
    root=tmp_path/"run_saved"
    root.mkdir()
    (root/"rotation__fit.json").write_text(
        json.dumps(dict(package="mlp_teacher_rotation",model="CR3")))
    (root/"rotation__test_evaluation.json").write_text(
        json.dumps(dict(parameter_names=["alpha","lambda_loss","gamma","tau"])))
    np.savez_compressed(root/"rotation__test_truth.npz",
                        theta_test=theta,test_ids=te)
    for path in (root,root/"candidate__jointdiag13"):
        path.mkdir(exist_ok=True)
        np.savez_compressed(path/"rotation__U.npz",U=U,
                            train_ids=tr,validation_ids=va,test_ids=te)
    dirs=add_plots_to_existing_run(root)
    assert len(dirs)==2
    for path in dirs:
        assert (path/"rotation__test_u_pairwise_scatter.png").is_file()
        assert (path/"rotation__test_theta_u_pairwise_scatter.png").is_file()
        assert (path/"rotation__test_u_corr.png").is_file()
        assert (path/"rotation__u_correlations_test.csv").is_file()


def test_posthoc_no_test_truth_still_outputs_u_u(tmp_path):
    from .plot_existing import add_plots_to_existing_run
    U,_,(tr,va,te)=_fixture()
    root=tmp_path/"run_saved"
    root.mkdir()
    (root/"rotation__fit.json").write_text(
        json.dumps(dict(package="mlp_teacher_rotation",model="CR3")))
    np.savez_compressed(root/"rotation__U.npz",U=U,
                        train_ids=tr,validation_ids=va,test_ids=te)
    assert add_plots_to_existing_run(root)==[root]
    assert (root/"rotation__test_u_pairwise_scatter.png").exists()
    assert not (root/"rotation__test_theta_u_pairwise_scatter.png").exists()
