from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
import torch

from experiments.cr3_condition_identifiability.design import (
    Design, NAMES, condition_mask, derivative_masks, physical_finite_probes,
    load_design, EXPECTED
)
from experiments.cr3_condition_identifiability.derivatives import (
    compute_fields, true_generator_selfcheck
)
from experiments.cr3_condition_identifiability.objectives import (
    moments,score_all,plane_rotation,alignment,matrix_from_params,
    candidate_rotations,scan
)


def make_synthetic_design():
    X=np.array([
        [50,0,.3,30,0,.7],       # gain-only
        [50,10,.5,30,20,.5],    # gamma-neutral
        [0,50,.3,0,30,.7],      # loss-only
        [30,10,.3,50,20,.7],    # general
    ],dtype=np.float64)
    codes=np.array([0,1,2,3])
    theta=np.array([[.6,1.7,.65,2.4],[.8,2.1,.75,1.9]],dtype=np.float64)
    return Design(x=X,codes=codes,theta=theta,
        mu=np.array([.7,1.9,.70,2.2]),
        sd=np.array([.15,.35,.2,.55]),x_sd=np.array([42,37,.26,41,35,.24]),
        subject_ids=np.arange(2),trial_ids=np.arange(4),
        bank_hash="mock",dataset_hash="mock")


def test_condition_labels_and_boundary_masks():
    d=make_synthetic_design()
    groups=condition_mask(d.x,d.codes)
    assert all(int(groups[n].sum())==1 for n in NAMES)
    ambient,tangent=derivative_masks(d.x,d.codes)
    assert not ambient[0,1] and not ambient[0,4]
    assert not ambient[2,0] and not ambient[2,3]
    assert ambient[1,2] and ambient[1,5]
    assert not tangent[1,2] and not tangent[1,5]
    assert tangent[3].all()


def test_physical_boundary_one_sided_is_finite_contrast_not_classic_derivative():
    d=make_synthetic_design()
    xp,xm,width,flag,ambient,tangent,onesided=physical_finite_probes(d.x,d.codes)
    assert width[0,1]==5.0 and onesided[0,1]
    assert xp[0,1,1]==5.0 and xm[0,1,1]==0.0
    assert flag[0,1]==1
    assert flag[1,2]==2
    assert width[1,2]==.10
    assert flag[3,0]==0
    assert (xp>=0).all() and (xm>=0).all()
    assert (xp[:,:,2]<=1).all() and (xm[:,:,5]<=1).all()


@pytest.fixture(scope="module")
def fields_and_moments():
    d=make_synthetic_design()
    f=compute_fields(d,channel="logit",device="cpu",batch=4)
    return d,f,moments(d,f)


def test_analytic_gradients_true_structural_zero_and_ux_masks(fields_and_moments):
    d,f,stats=fields_and_moments
    assert np.allclose(f["G"][:,0,1],0,atol=1e-12)
    assert np.allclose(f["G"][:,1,2],0,atol=1e-12)
    assert np.isfinite(f["finite_UX"]).all()
    assert not f["ambient_valid"][0,1]
    assert f["finite_kind"][0,1]==1
    assert np.linalg.norm(f["finite_UX"][:,0,:,1])>0
    assert stats["C_support"]["ambient"][0,1]==0
    assert stats["C_support"]["finite_boundary"][0,1]==1
    assert np.linalg.norm(stats["F"][0][:,1])<1e-10
    assert np.linalg.norm(stats["F"][1][:,2])<1e-10
    assert stats["counts"].tolist()==[1,1,1,1]


def test_prob_chain_and_fisher_uses_logit_score(fields_and_moments):
    d,logit,sl=fields_and_moments
    prob=compute_fields(d,channel="prob",device="cpu",batch=4)
    sp=moments(d,prob)
    p=logit["p"]
    assert np.allclose(prob["G"],p[...,None]*(1-p[...,None])*logit["G"],rtol=2e-10)
    assert np.allclose(sp["F"],sl["F"],atol=1e-10,rtol=1e-10)
    assert not np.allclose(sp["G_cov"],sl["G_cov"])


def test_rotation_moment_contractions_and_true_axis_diagnostics(fields_and_moments):
    _,_,stats=fields_and_moments
    I=np.eye(4)
    P=plane_rotation(0,1,90)
    a=score_all(stats,I)
    perm=score_all(stats,P)
    for key in a:
        assert np.isclose(a[key],perm[key],atol=1e-10,rtol=1e-10,equal_nan=True),key
    middle=score_all(stats,plane_rotation(0,1,45))
    assert middle["A_two_condition_null"]>a["A_two_condition_null"]
    assert middle["A_two_condition_null"]>=0
    assert np.allclose(matrix_from_params(np.zeros(6)),I)
    assert alignment(P)["axis_max_angle_deg"]<1e-6


def test_scan_does_not_select_identity_or_oracle_centered_planes(fields_and_moments):
    _,_,stats=fields_and_moments
    rows,best=scan(stats,mixtures=("empirical",),random_count=4,
                   seed=19,optimize=False)
    assert any(r["kind"]=="truth_reference" for r in rows)
    assert any(r["kind"]=="plane" for r in rows)
    assert all(item["best"]["kind"]=="haar" for item in best.values())
    assert all(np.isfinite(item["best"]["axis_mean_abs_cos"]) for item in best.values())
    assert any(k.startswith("empirical/A_") for k in best)


def test_real_reference_formula_independent_selfcheck(fields_and_moments):
    d,_,_=fields_and_moments
    assert true_generator_selfcheck(d)<2e-10


def test_load_original_bank_on_synthetic_files(tmp_path):
    d=make_synthetic_design()
    rows=[]
    codes=[]
    for idx,count in enumerate([60,60,40,340]):
        rows.extend([d.x[idx].copy() for _ in range(count)])
        codes.extend([idx]*count)
    bank=tmp_path/"bank.npz"
    dataset=tmp_path/"dataset.npz"
    np.savez_compressed(bank,X=np.array(rows),type_codes=np.array(codes),
                        type_names=np.array(NAMES))
    theta=np.vstack([d.theta]*10)
    np.savez_compressed(dataset,parameters=theta)
    got=load_design(subjects=4,stimuli=16,seed=44,bank=bank,dataset=dataset)
    assert len(got.x)==16 and len(got.theta)==4
    assert np.bincount(got.codes,minlength=4).min()>=1
    assert np.allclose(got.mu,theta.mean(axis=0))


def test_run_mock_precheck_and_full_smoke(tmp_path,monkeypatch):
    from experiments.cr3_condition_identifiability import run
    d=make_synthetic_design()
    rows=[]
    codes=[]
    for idx,count in enumerate([60,60,40,340]):
        rows.extend([d.x[idx].copy() for _ in range(count)])
        codes.extend([idx]*count)
    bank=tmp_path/"bank.npz"
    dataset=tmp_path/"dataset.npz"
    np.savez_compressed(bank,X=np.array(rows),type_codes=np.array(codes),
                        type_names=np.array(NAMES))
    np.savez_compressed(dataset,parameters=np.vstack([d.theta]*4))
    monkeypatch.setattr(run,"BASE",tmp_path/"out")
    shared=["--tag","mini","--bank",str(bank),"--dataset",str(dataset),
            "--subjects","2","--stimuli","8","--channels","logit","--batch","8",
            "--random-rotations","2","--no-optimize"]
    run.main(["--stage","precheck",*shared])
    assert (tmp_path/"out/mini/precheck.json").exists()
    run.main(["--stage","all",*shared])
    result=tmp_path/"out/mini"
    assert (result/"logit/fields.npz").exists()
    assert (result/"logit/moments.npz").exists()
    assert (result/"logit/rotations_all_scores.csv").exists()
    assert (result/"logit/best_rotations.json").exists()
    assert (result/"REPORT.md").exists()
    run.main(["--stage","search",*shared])
    assert (result/"REPORT.md").exists()
