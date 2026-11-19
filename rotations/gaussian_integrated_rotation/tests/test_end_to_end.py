"""Tiny end-to-end path with synthetic Z, artificial q/X points and a toy decoder.

No real checkpoint/GPU needed. Tests stage I/O, ALL-subject Gaussian,
separate FIT/VALID quadrature, blind SO(3), and post-freeze Oracle.
"""
import json
from pathlib import Path
import numpy as np
import torch

from rotations.gaussian_integrated_rotation import run as gifrun
from rotations.gaussian_integrated_rotation import data as gifdata


class Toy(torch.nn.Module):
    def decode(self,x,z):
        # Deliberately nonlinear q+X function; has nonzero UU Hessian.
        logit=(z[:,0,None]**2*x[...,0] +
               z[:,1,None]*z[:,2,None]*x[...,1] +
               z[:,2,None]**2*x[...,2])
        return torch.stack((torch.zeros_like(logit),logit),dim=-1)


def _fake_np(tmp_path):
    base=tmp_path/"outputs"/"choice"/"CR4"
    condition="beta0p0005_lambda0_ctx0p25to0p8_L3"
    ck=base/"s3_train"/condition/"repeat01"/"s3_train__checkpoint.pt"
    latent=base/"s4_evaluate"/condition/"repeat01"/"s4_evaluate__latent.npz"
    dataset=base/"s1_generate"/"s1_generate__dataset.npz"
    bank=base/"s0_design"/"CR4_3P_random500.npz"
    for p in (ck,latent,dataset,bank):
        p.parent.mkdir(parents=True,exist_ok=True)
    ck.write_bytes(b"fake checkpoint -- patched restore")
    rng=np.random.default_rng(230)
    z=rng.normal(size=(35,3))
    z[20:]+=np.array([2.,-1.,1.])
    bank_x=np.empty((500,6))
    bank_x[:,[0,1,3,4]]=rng.uniform(5,200,size=(500,4))
    bank_x[:,[2,5]]=rng.uniform(.05,.95,size=(500,2))
    np.savez_compressed(bank,X=bank_x)
    theta=z@np.array([[1.,.2,.1],[-.2,1.,0],[0.,-.15,1.]]).T
    np.savez_compressed(dataset,parameters=theta)
    np.savez_compressed(latent,
        z=z,context_mode="all_observed_trials",
        checkpoint_sha256=gifdata.sha256(ck),
        dataset_sha256=gifdata.sha256(dataset),
        train_ids=np.arange(20),
        validation_ids=np.arange(20,27),
        test_ids=np.arange(27,35))
    return condition,base,z


def test_complete_gaussian_run_and_oracle_after_freeze(tmp_path,monkeypatch):
    condition,base,z=_fake_np(tmp_path)
    monkeypatch.setattr(gifdata,"ROOT",tmp_path)
    def mocked_restore(src,device):
        # Test that the actual integration queries this decoder at
        # artificial Gaussian nodes. It must never receive subject rows.
        assert len(src["q"])==len(z)
        return Toy(),torch.device("cpu"),np.zeros(6),np.ones(6)
    monkeypatch.setattr(gifrun,"restore",mocked_restore)
    argv=["--model","CR4","--tag","tiny","--no-evaluate",
          "--set","z_fit_points=8","--set","z_validation_points=8",
          "--set","x_bank_limit=8","--set","z_batch=2",
          "--set","x_batch=4","--set","restarts=2","--set","maxiter=30",
          "--set","oracle_restarts=2","--set","oracle_maxiter=40"]
    out=gifrun.main(argv)
    assert (out/"fit_complete.json").is_file()
    assert (out/"rotation.npz").is_file()
    assert (out/"moments_fit.npz").is_file()
    assert (out/"moments_validation.npz").is_file()
    assert not (out/"oracle_comparison.json").exists()
    meta=json.loads((out/"protocol.json").read_text())
    assert meta["all_subject_count"]==35
    assert meta["subjects_used_only_for"]=="Gaussian mu and covariance; post-freeze Oracle"
    assert meta["gaussian"]["fit"]["points"]==8
    assert meta["X"]["fit"]["count"]==8
    with np.load(out/"rotation.npz") as a:
        assert a["R"].shape==(3,3)
        assert len(a["U_all_subjects"])==35
    # Re-entry does NOT redo derivatives/optimization: only post-fit oracle.
    again=gifrun.main([x for x in argv if x!="--no-evaluate"])
    assert again==out
    assert (out/"oracle_comparison.json").exists()
    report=json.loads((out/"oracle_comparison.json").read_text())
    assert report["oracle_used_to_select_blind_R"] is False
    assert report["participant_scope"]=="ALL-subject in-sample oracle, no TEST"
    assert set(report["comparison"])=={
        "identity_PCA","optimized_R_star","oracle_procrustes","oracle_correlation"}
