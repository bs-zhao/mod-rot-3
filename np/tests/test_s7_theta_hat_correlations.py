"""S7: all-Z, TRAIN variance-filtered raw-Z and ratio-filtered PC theta-hat audits."""
import csv
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from modrot3.analysis import corr, ridge_probe
from modrot3.cli import parse_args


def load_stage():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] /
                              "s7_regression.py"))


def test_s7_z_variance_selection_is_strict_and_train_only():
    ns=load_stage()
    hook=ns["configure_s7_parser"]
    args=parse_args("S7 test",[
        "--model","CR3","--z-var-threshold","0.05",
        "--pca-threshold","0.10"],configure_parser=hook)
    assert args.z_var_threshold==.05
    assert args.pca_threshold==.10
    # TEST variation cannot make a TRAIN-constant feature be selected.
    z=np.array([[1,0,0],[2,0,0],[3,0,0],[4,100,0],[5,200,5.]],float)
    selected,variances=ns["select_z_above_threshold"](z,[0,1,2],.05)
    assert selected.tolist()==[0]
    assert variances[1]==0.0
    assert variances[2]==0.0
    with pytest.raises(ValueError,match="z-var-threshold"):
        ns["select_z_above_threshold"](z,[0,1,2],-1)
    with pytest.raises(ValueError,match="No original Z dimension"):
        ns["select_z_above_threshold"](z,[0,1,2],1e6)


def test_true_theta_by_theta_hat_matrix_is_not_a_single_diagonal_score(tmp_path):
    ns=load_stage()
    rng=np.random.default_rng(20261108)
    n=35
    theta=rng.normal(size=(n,4))
    prediction=theta@np.array([
        [1,.4,0,0],
        [0,1,.15,0],
        [0,0,1,-.3],
        [0,0,0,1],
    ])
    fits={key:dict(theta_linear_pred=prediction.copy(),
                   test_r2=[.1,.2,.3,.4])
          for key in ("full_z","selected_z","selected_pc")}
    names=["alpha","lambda_loss","gamma","tau"]
    audit=ns["theta_hat_correlation_audit"](
        tmp_path,theta,names,fits,
        test_ids=np.arange(100,100+n),
        z_variance_threshold=.05,pca_threshold=.10,
        selected_z_indices=np.array([0,1]),
        selected_pc_indices=np.array([0,1,2]))
    expected=corr(theta,prediction)
    got=np.array(audit["methods"]["full_z"][
        "test_pearson_r_true_vs_predicted"])
    assert np.allclose(got,expected)
    assert abs(got[0,1])>.1
    assert len(audit["methods"])==3
    for kind in ("full_z","selected_z","selected_pc"):
        with (tmp_path/f"s12__theta_hat_corr_{kind}.csv").open(
            newline="",encoding="utf-8") as f:
            rows=list(csv.reader(f))
        assert rows[0]==["true_theta / predicted_theta",
                          *["pred_"+x for x in names]]
        assert [r[0] for r in rows[1:]]==names
        csv_values=np.array([[float(v) for v in row[1:]]
                             for row in rows[1:]])
        assert np.allclose(csv_values,expected)
        assert (tmp_path/f"s12__theta_hat_corr_{kind}.png").is_file()
    assert (tmp_path/"s12__theta_hat_corr_3way.png").is_file()
    with np.load(tmp_path/"s12__theta_hat_test_predictions.npz") as f:
        assert np.array_equal(f["test_ids"],np.arange(100,100+n))
        assert np.allclose(f["theta_hat_selected_z"],prediction)
        assert f["selected_z_indices_1based"].tolist()==[1,2]
        assert f["selected_pc_indices_1based"].tolist()==[1,2,3]


def test_full_s7_three_independent_regressions_and_existing_outputs(tmp_path):
    ns=load_stage()
    f=ns["regression_report"]
    rng=np.random.default_rng(1091)
    n,d=115,4
    z=rng.normal(size=(n,d))*np.array([2,.8,.07,.009])
    theta=np.column_stack([
        z[:,0]+rng.normal(size=n)*.02,
        z[:,1]+rng.normal(size=n)*.02,
        z[:,0]-.5*z[:,1]+rng.normal(size=n)*.02,
        z[:,1]-.4*z[:,0]+rng.normal(size=n)*.02,
    ])
    splits=dict(train=np.arange(75),
                validation=np.arange(75,95),
                test=np.arange(95,115))
    source=tmp_path/"synthetic_s5.npz"
    source.write_bytes(b"dummy")
    folder=tmp_path/"s12_z_diagnostics"
    folder.mkdir()
    np.savez_compressed(folder/"s12__pca.npz",
        source_latent_sha256=np.array("synthetic_sha"),
        transformed_z=z,components=np.eye(d),
        explained_variance_ratio=np.array([.6,.3,.08,.02]),
        explained_variance=np.array([6,3,.8,.2]),
        train_mean=np.zeros(d),
        train_ids=splits["train"],
        validation_ids=splits["validation"],
        test_ids=splits["test"])
    st=f.__globals__
    st["load_frozen"]=lambda a:(z,theta,splits,source)
    st["sha256"]=lambda p:"synthetic_sha"
    st["stage_folder"]=lambda a,stage:tmp_path/stage
    st["train_path"]=lambda a:tmp_path/"checkpoint.pt"
    st["condition"]=lambda a:"mock"
    args=SimpleNamespace(model="CR3",latent_dim=d,pca_threshold=.10,
                         z_var_threshold=.05)
    f(args)
    assert (folder/"s12__theta_hat_ridge_3way.csv").is_file()
    for name in ("full_z","selected_z","selected_pc"):
        assert (folder/f"s12__theta_hat_corr_{name}.csv").is_file()
        assert (folder/f"s12__theta_hat_corr_{name}.png").is_file()
    with (folder/"s12__theta_hat_ridge_3way.csv").open(
        newline="",encoding="utf-8") as fp:
        rows=list(csv.DictReader(fp))
    assert len(rows)==12
    assert [r["representation"] for r in rows[::4]]==[
        "full_z","selected_z","selected_pc"]
    assert [int(rows[i]["n_features"]) for i in (0,4,8)]==[4,2,2]
    audit=json.loads((folder/"s12__theta_hat_corr_3way.json").read_text())
    assert audit["selected_original_z_indices_1based"]==[1,2]
    assert audit["selected_pc_indices_1based"]==[1,2]
    with np.load(folder/"s12__theta_hat_test_predictions.npz") as data:
        assert data["theta_true"].shape==(20,4)
        manually=ridge_probe(z[:,[0,1]],theta,**{
            "train":splits["train"],"validation":splits["validation"],
            "test":splits["test"]})
        assert np.allclose(data["theta_hat_selected_z"],
                           manually["theta_linear_pred"])
        for name,arr in [
            ("full_z",data["theta_hat_full_z"]),
            ("selected_z",data["theta_hat_selected_z"]),
            ("selected_pc",data["theta_hat_selected_pc"]),
        ]:
            reference=corr(theta[splits["test"]],arr)
            actual=audit["methods"][name]["test_pearson_r_true_vs_predicted"]
            assert np.allclose(reference,actual)
    # Older S7 products remain unchanged in path and schema.
    assert (folder/"s12__pca_threshold_ridge.csv").is_file()
    assert (folder/"s12__full_z_ridge.csv").is_file()
    assert (folder/"s12__ridge_input_comparison.csv").is_file()
    assert (folder/"s12__pca_and_regression.png").is_file()
