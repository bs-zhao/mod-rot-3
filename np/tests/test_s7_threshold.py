"""S7 extra individual-PC variance threshold analysis; no NP training needed."""
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

from modrot3.cli import parse_args


def _s7():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "s7_regression.py"))


def test_strict_individual_variance_threshold_and_errors():
    select = _s7()["select_pcs_above_threshold"]
    ratio = np.array([.55, .25, .10, .08, .02])
    assert select(ratio).tolist() == [0, 1]
    assert select(ratio, .099).tolist() == [0, 1, 2]
    assert select(ratio, 0).tolist() == list(range(5))
    with pytest.raises(ValueError, match="No PCA component"):
        select(ratio, .6)
    with pytest.raises(ValueError, match="pca-threshold"):
        select(ratio, 1)
    with pytest.raises(ValueError, match="pca-threshold"):
        select(ratio, -1)
    with pytest.raises(ValueError, match="pca-threshold"):
        select(ratio, float("nan"))


def test_s7_only_cli_threshold_defaults_and_override():
    hook = _s7()["configure_s7_parser"]
    a = parse_args("S7 test", ["--model", "CR3", "--beta", "0",
                                "--latent-dim", "8"], configure_parser=hook)
    assert a.pca_threshold == .10
    b = parse_args("S7 test", ["--pca-threshold", "0.22"],
                   configure_parser=hook)
    assert b.pca_threshold == .22
    c = parse_args("S7 test", ["--pca-variance-threshold", "0.3"],
                   configure_parser=hook)
    assert c.pca_threshold == .3
    # Other stages' CLI is unchanged: no S7-only option.
    with pytest.raises(SystemExit):
        parse_args("S6", ["--pca-threshold", ".3"])


def test_s7_additional_artifacts_without_replacing_original(tmp_path):
    namespace = _s7()
    f = namespace["regression_report"]
    rng = np.random.default_rng(20261106)
    n, d = 80, 4
    z = rng.normal(size=(n, d))
    truth = z @ rng.normal(size=(d, d)) + rng.normal(size=(n, d))*.02
    splits = {
        "train": np.arange(50),
        "validation": np.arange(50, 65),
        "test": np.arange(65, 80),
    }
    source = tmp_path / "synthetic_s5.npz"
    source.write_bytes(b"dummy Z")
    folder = tmp_path / "s12_z_diagnostics"
    folder.mkdir(parents=True)
    np.savez_compressed(
        folder/"s12__pca.npz",
        source_latent_sha256=np.array("synthetic_sha"),
        transformed_z=z,
        train_ids=splits["train"],
        validation_ids=splits["validation"],
        test_ids=splits["test"],
        explained_variance_ratio=np.array([.65, .20, .10, .05]),
        components=np.eye(d), train_mean=np.zeros(d),
        explained_variance=np.array([6.5, 2., 1., .5]),
    )
    state = f.__globals__
    state["load_frozen"] = lambda _: (z, truth, splits, source)
    state["sha256"] = lambda _: "synthetic_sha"
    state["train_path"] = lambda _: tmp_path/"checkpoint.pt"
    state["condition"] = lambda _: "synthetic"
    state["stage_folder"] = lambda _, stage: tmp_path/stage
    args=SimpleNamespace(model="CR3",beta=0.,latent_dim=d,
                         pca_threshold=.10)
    f(args)

    import json
    report=json.loads((folder/"s12__pca_threshold_ridge.json").read_text())
    assert report["threshold"] == .10
    assert report["retained_pc_count"] == 2
    assert report["retained_pc_indices_1based"] == [1, 2]
    assert np.isclose(report["retained_variance_fraction"], .85)
    assert len(report["test_r2"]) == d
    for name in ("s12__full_z_ridge.csv","s12__pca_ridge_curve.csv",
                 "s12__pca_and_regression.png","s12__ridge_scatter.png",
                 "s12__diagnostics.json","s12__pca_threshold_ridge.csv",
                 "s12__pca_threshold_predictions.npz",
                 "s12__pca_threshold_scatter.png",
                 "s12__ridge_input_comparison.csv",
                 "s12__ridge_input_deltas.csv",
                 "s12__ridge_input_comparison_scatter.png",
                 "s12__ridge_input_comparison_r2.png"):
        assert (folder/name).is_file(), name
    with np.load(folder/"s12__pca_threshold_predictions.npz") as fz:
        assert fz["theta_threshold_pca_pred"].shape == (15,d)
        assert fz["retained_pc_indices_1based"].tolist()==[1,2]
    old=json.loads((folder/"s12__diagnostics.json").read_text())
    assert old["full_z_ridge"]["test_r2"]
    assert old["pca_variance_threshold_ridge"]["retained_pc_count"]==2
    s4=json.loads((tmp_path/"s4_evaluate"/"s4_evaluate__summary.json").read_text())
    assert "results" in s4 and "pca_variance_threshold_ridge" in s4


def test_s7_prints_and_compares_all_z_and_thresholded_pcs(tmp_path, capsys):
    import csv
    from types import SimpleNamespace
    ns=_s7()
    f=ns["regression_report"]
    rng=np.random.default_rng(99)
    n,d=90,4
    z=rng.normal(size=(n,d))
    truth=z@rng.normal(size=(d,d)) + rng.normal(size=(n,d))*.03
    splits=dict(train=np.arange(60),validation=np.arange(60,75),
                test=np.arange(75,90))
    folder=tmp_path/"s12_z_diagnostics"
    folder.mkdir()
    source=tmp_path/"s5.npz"
    source.write_bytes(b"synthetic")
    np.savez_compressed(folder/"s12__pca.npz",
        source_latent_sha256=np.array("same"),transformed_z=z,
        train_ids=splits["train"],validation_ids=splits["validation"],
        test_ids=splits["test"],
        explained_variance_ratio=np.array([.65,.20,.10,.05]),
        components=np.eye(d),train_mean=np.zeros(d),
        explained_variance=np.array([6.5,2.,1.,.5]))
    st=f.__globals__
    st["load_frozen"]=lambda _: (z,truth,splits,source)
    st["sha256"]=lambda _: "same"
    st["train_path"]=lambda _: tmp_path/"ck.pt"
    st["condition"]=lambda _: "synthetic"
    st["stage_folder"]=lambda _,stage:tmp_path/stage
    f(SimpleNamespace(model="CR3",beta=0.,latent_dim=d,pca_threshold=.10))
    printed=capsys.readouterr().out
    assert "FULL Z (all raw Z)" in printed
    assert "SELECTED PCs (thresholded PCA)" in printed
    assert "Full-Z predictors: Z1-Z4 (4 jointly)" in printed
    assert "Retained PCA components: 2 / 4 (PC1, PC2)" in printed
    assert printed.count("alpha: TEST R²=")==2
    assert printed.count("lambda_loss: TEST R²=")==2
    assert "SELECTED PCs minus FULL Z" in printed
    with (folder/"s12__ridge_input_comparison.csv").open(newline="") as fp:
        rows=list(csv.DictReader(fp))
    assert len(rows)==8
    assert {r["representation"] for r in rows}=={"full_z","threshold_pca"}
    assert all(int(r["feature_count"])==4 for r in rows[:4])
    assert all(int(r["feature_count"])==2 for r in rows[4:])
    assert all(r["features"]=="PC1,PC2" for r in rows[4:])
    with (folder/"s12__ridge_input_deltas.csv").open(newline="") as fp:
        deltas=list(csv.DictReader(fp))
    assert len(deltas)==4
    assert np.isclose(
        float(deltas[0]["delta_test_r2_pca_minus_full_z"]),
        float(rows[4]["test_r2"])-float(rows[0]["test_r2"]))
