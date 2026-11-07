"""S6 PC explained-variance and raw-Z absolute-variance threshold diagnostics."""
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import runpy

import numpy as np
import pytest
from modrot3.cli import parse_args


def _ns():
    return runpy.run_path(str(Path(__file__).resolve().parents[1]/"s6_pca.py"))


def test_s6_only_threshold_cli_and_strict_cutoffs():
    ns=_ns()
    hook=ns["configure_s6_parser"]
    args=parse_args("S6",["--pc-var-threshold",".05",
                            "--z-var-threshold",".2"],configure_parser=hook)
    assert (args.pc_var_threshold,args.z_var_threshold)==(.05,.2)
    assert ns["select_above_threshold"](
        np.array([.6,.2,.05,.04]),.05,"test").tolist()==[0,1]
    assert ns["select_above_threshold"](
        np.array([.6,.2,.05,.04]),.6,"test").size==0
    with pytest.raises(ValueError,match="threshold"):
        ns["select_above_threshold"](np.array([.2]),float("nan"),"test")
    with pytest.raises(ValueError,match="threshold"):
        ns["select_above_threshold"](np.array([.2]),-.1,"test")
    with pytest.raises(SystemExit):
        parse_args("S7",["--pc-var-threshold",".05"])


def test_pearson_matrix_is_signed_and_handles_constant():
    ns=_ns()
    rng=np.random.default_rng(20261106)
    x=rng.normal(size=(70,4))
    x[:,3]=2.
    corr=ns["z_axis_pearson"](x)
    assert corr.shape==(4,4)
    assert np.allclose(corr[:3,:3],np.corrcoef(x[:,:3],rowvar=False))
    assert np.allclose(corr[3],0.)
    assert np.allclose(corr[:,3],0.)
    assert np.all(np.isfinite(corr))


def test_s6_exports_both_selections_and_preserves_original_pca(tmp_path):
    ns=_ns()
    f=ns["pca_report"]
    rng=np.random.default_rng(11)
    n=90
    tr=np.arange(55)
    va=np.arange(55,70)
    te=np.arange(70,90)
    z=rng.normal(size=(n,4))*np.array([2.,.9,.24,.04])
    theta=rng.normal(size=(n,4))
    source=tmp_path/"source.npz"
    source.write_bytes(b"frozen")
    glob=f.__globals__
    glob["load_frozen"]=lambda a:(z,theta,{"train":tr,"validation":va,"test":te},source)
    glob["sha256"]=lambda p:"synthetic_sha"
    glob["train_path"]=lambda a:tmp_path/"checkpoint"
    glob["condition"]=lambda a:"synthetic_beta"
    glob["stage_folder"]=lambda a,stage:tmp_path/stage
    args=SimpleNamespace(model="CR3",beta=.0005,latent_dim=4,
                         pc_var_threshold=.05,z_var_threshold=.2)
    f(args)
    folder=tmp_path/"s12_z_diagnostics"
    data=json.loads((folder/"s12__selected_dimensions.json").read_text())
    assert data["pc"]["selected_count"]==2
    assert data["pc"]["selected_names"]==["PC1","PC2"]
    assert data["z"]["selected_names"]==["z1","z2"]
    for stem in ("s12__selected_pc","s12__selected_z"):
        for suffix in ("_correlations.csv",
                       "_correlations_train.csv",
                       "_correlations_validation.csv",
                       "_correlations_test.csv",
                       "_correlation_heatmap.png",
                       "_pairwise_scatter.png"):
            path=folder/(stem+suffix)
            assert path.is_file() and path.stat().st_size>0,path
    with (folder/"s12__selected_z_correlations.csv").open(newline="") as fs:
        rows=list(csv.reader(fs))
    assert rows[0]==["dimension","z1","z2"]
    assert np.isclose(float(rows[1][2]),np.corrcoef(z[:,0],z[:,1])[0,1])
    with (folder/"s12__selected_pc_correlations_train.csv").open(newline="") as fs:
        pc_rows=list(csv.reader(fs))
    assert abs(float(pc_rows[1][2]))<1e-10
    with np.load(folder/"s12__pca.npz") as saved:
        assert saved["transformed_z"].shape==(n,4)
        assert str(saved["source_latent_sha256"])=="synthetic_sha"
    original=json.loads((folder/"s12__pca_summary.json").read_text())
    assert "theta_vs_pc_test_corr" in original
    assert original["selected_pc_analysis"]["selected_count"]==2
    assert original["selected_z_analysis"]["selected_count"]==2


def test_zero_retained_axis_does_not_break_pca_outputs(tmp_path):
    ns=_ns()
    out=ns["selected_pairwise_diagnostics"](
        tmp_path,np.array([[1.,2.],[3.,4.],[5.,6.]]),
        np.array([],dtype=int),"z",
        np.array([0,1]),np.array([],int),np.array([2]))
    assert out["status"]=="none_above_threshold"
    assert out["selected_count"]==0
