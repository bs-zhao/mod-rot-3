"""ALL-subject supervised Oracle comparison is separate from blind R* fitting."""
from __future__ import annotations

import json
import numpy as np
import pytest

from rotations.hessian_sparsity import oracle_compare as oc
from rotations.hessian_sparsity.loss import summarize_fields,HessianLoss


def orthogonal(d,seed):
    rng=np.random.default_rng(seed)
    q,_=np.linalg.qr(rng.normal(size=(d,d)))
    if np.linalg.det(q)<0:q[0]*=-1
    return q


@pytest.mark.parametrize("dimension",[3,4])
def test_oracle_uses_all_subjects_and_recovers_high_alignment(dimension):
    rng=np.random.default_rng(114+dimension)
    q=rng.normal(size=(300,dimension))
    true_R=orthogonal(dimension,121+dimension)
    theta=q@true_R.T
    Rpro,meta=oc.procrustes_oracle(q,theta)
    # z-scoring the target theta can slightly change its optimal SO(d)
    # Procrustes mapping when latent dimensions have unequal variation.
    assert float(np.mean(np.diag(oc._corr(theta,q@Rpro.T))**2))>.99
    assert oc._is_so(Rpro)
    Rcorr,report=oc.correlation_oracle(
        q,theta,Rpro,restarts=3,maxiter=80,seed=41)
    assert oc._is_so(Rcorr)
    assert report["used_hessian_losses_for_oracle"] is False
    assert report["no_test_split"] is True
    np.testing.assert_allclose(np.abs(np.diag(oc._corr(theta,q@Rcorr.T))),
                               np.ones(dimension),atol=1e-6)


def _fake_run(tmp_path,monkeypatch,d=3):
    monkeypatch.setattr(oc,"ROOT",tmp_path)
    model="CR4"
    root=tmp_path/"outputs"/"choice"/model
    cond="beta0p0005_lambda0_ctx0p25to0p8_L3"
    parent=root/"rotations"/"hessian_sparsity"/cond/"repeat01"
    out=parent/"main_joint_pooled_fakefingerprint"
    out.mkdir(parents=True)
    s3=root/"s3_train"/cond/"repeat01"/"s3_train__checkpoint.pt"
    s5=root/"s4_evaluate"/cond/"repeat01"/"s4_evaluate__latent.npz"
    dataset=root/"s1_generate"/"s1_generate__dataset.npz"
    bank=root/"s0_design"/"CR4_3P_random500.npz"
    for f in (s3,s5,bank):
        f.parent.mkdir(parents=True,exist_ok=True)
        f.write_bytes(b"immutable source for oracle comparison test")
    rng=np.random.default_rng(23)
    n=90
    q=rng.normal(size=(n,d))
    truth=orthogonal(d,17)
    theta=q@truth.T+.03*rng.normal(size=(n,d))
    dataset.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dataset,parameters=theta)
    all_ids=np.arange(n)
    fx=np.arange(10);vx=np.arange(10,20)
    Rstar=np.eye(d)
    np.savez_compressed(out/"rotation.npz",R=Rstar,q=q,
                        all_subject_ids=all_ids,fit_x_ids=fx,
                        valid_x_ids=vx)
    flds=[]
    for off in (0,10):
        H=rng.normal(size=(12,10,d,d))
        H=(H+H.swapaxes(-1,-2))/2
        J=rng.normal(size=(12,10,d,6))
        flds.append(dict(UU=H,UX=J,group_codes=np.zeros(10,dtype=int)))
    np.savez_compressed(out/"fit_fields.npz",
                        H_UU=flds[0]["UU"],H_UX=flds[0]["UX"],
                        condition_codes=flds[0]["group_codes"])
    np.savez_compressed(out/"validation_fields.npz",
                        H_UU=flds[1]["UU"],H_UX=flds[1]["UX"],
                        condition_codes=flds[1]["group_codes"])
    fit_loss=HessianLoss(summarize_fields(flds[0]),mode="joint")
    val_loss=HessianLoss(summarize_fields(flds[1]),mode="joint")
    hashes={name:oc.sha256(file) for name,file in (
        ("checkpoint",s3),("latent",s5),("dataset",dataset),("bank",bank))}
    protocol=dict(model=model,condition=cond,repeat=1,
                  hashes=hashes,
                  config=dict(mode="joint",grouping="pooled",
                              uu_weight=1.,ux_weight=1.,epsilon=1e-8),
                  rotation_subject_scope="all_subjects_no_test",
                  all_subject_count=n,
                  fit_x_ids=fx.tolist(),valid_x_ids=vx.tolist())
    (out/"protocol.json").write_text(json.dumps(protocol))
    (out/"fit_complete.json").write_text(json.dumps(
        dict(rotation_frozen=True,mode="joint",grouping="pooled",
             oracle_theta_used_for_optimization=False)))
    (out/"loss_diagnostics.json").write_text(json.dumps(
        dict(fit_loss=fit_loss.scalar(Rstar),
             validation_loss=val_loss.scalar(Rstar))))
    return out,truth


def test_postfit_oracle_uses_existing_fields_and_writes_independent_report(tmp_path,monkeypatch):
    out,truth=_fake_run(tmp_path,monkeypatch)
    report=oc.compare_existing(out,oracle_restarts=4,oracle_maxiter=90)
    assert (out/"oracle_comparison.json").is_file()
    assert (out/"oracle_rotations.npz").is_file()
    assert (out/"oracle_loss_comparison.csv").is_file()
    assert (out/"oracle_loss_comparison.png").is_file()
    assert (out/"oracle_axis_overlap.png").is_file()
    assert report["original_R_star_remains_unchanged"]
    assert report["theta_used_in_blind_optimization"] is False
    results=report["comparison"]
    assert set(results)=={"identity_PCA","optimized_R_star",
                          "oracle_R_procrustes","oracle_R_correlation"}
    assert results["oracle_R_correlation"]["ALL"]["mean_matched_all_r2"]>.995
    assert report["oracle_minus_optimized"]["FIT"]==pytest.approx(
        results["oracle_R_correlation"]["FIT"]["total"]-
        results["optimized_R_star"]["FIT"]["total"])
    with np.load(out/"rotation.npz") as saved:
        np.testing.assert_array_equal(saved["R"],np.eye(3))
        np.testing.assert_array_equal(saved["all_subject_ids"],np.arange(90))
    before=(out/"oracle_comparison.json").read_bytes()
    oc.compare_existing(out)
    assert (out/"oracle_comparison.json").read_bytes()==before


def test_postfit_comparison_refuses_inconsistent_hessian_loss(tmp_path,monkeypatch):
    out,_=_fake_run(tmp_path,monkeypatch)
    p=out/"loss_diagnostics.json"
    bad=json.loads(p.read_text())
    bad["fit_loss"]["total"]+=.01
    p.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError,match="no longer reproduces"):
        oc.compare_existing(out,oracle_restarts=1,oracle_maxiter=20)


def test_no_oracle_if_frozen_marker_does_not_prove_theta_blind(tmp_path,monkeypatch):
    out,_=_fake_run(tmp_path,monkeypatch)
    (out/"fit_complete.json").write_text('{"rotation_frozen":true}')
    with pytest.raises(RuntimeError,match="theta-blind"):
        oc.compare_existing(out,oracle_restarts=1,oracle_maxiter=20)


def test_old_subset_protocol_is_rejected_for_true_all_subject_comparison(tmp_path,monkeypatch):
    out,_=_fake_run(tmp_path,monkeypatch)
    p=out/"protocol.json"
    meta=json.loads(p.read_text())
    del meta["rotation_subject_scope"]
    p.write_text(json.dumps(meta))
    with pytest.raises(RuntimeError,match="ALL-subject rotation"):
        oc.compare_existing(out,oracle_restarts=1,oracle_maxiter=20)
