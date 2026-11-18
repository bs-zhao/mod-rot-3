"""Pure analytic group-sparsity invariants and lightweight integration smoke."""
from __future__ import annotations

import json
import numpy as np
import pytest

from experiments.cr3_condition_identifiability.objectives import (
    plane_rotation, C_KINDS
)
from experiments.cr3_condition_identifiability.dimension_sparsity import (
    prepare, evaluate, dimension_loss, mixing_weights, _normalized_axis_energy,
)
from experiments.cr3_condition_identifiability.run import save_moments
from experiments.cr3_condition_identifiability.run_group_sparsity import main


def stats_four_groups():
    # Four synthetic conditions each depend on ONE different U coordinate.
    G = np.stack([np.diag(np.eye(4)[j]) for j in range(4)])
    H4 = np.zeros((4,4,4,4,4),dtype=float)
    cov={k:np.zeros((4,6,4,4)) for k in C_KINDS}
    support={k:np.zeros((4,6)) for k in C_KINDS}
    support["tangent"][:,0]=1.
    cov["tangent"][:,0]=2*G
    return dict(F=G.copy(),G_cov=G,H4=H4,
                C_cov=cov,C_support=support,counts=np.array([25]*4),
                channel="logit",x_scale="standardized")


def test_stratification_recovers_axis_when_pooling_blinds_it():
    M=prepare(stats_four_groups())
    identity=evaluate(M,np.eye(4),recipe="U",metric="lp",aggregation="stratified")
    pooled=evaluate(M,np.eye(4),recipe="U",metric="lp",aggregation="pooled")
    R=plane_rotation(0,1,45)
    moved=evaluate(M,R,recipe="U",metric="lp",aggregation="stratified")
    moved_pooled=evaluate(M,R,recipe="U",metric="lp",aggregation="pooled")
    assert np.isclose(identity["score"],1)
    assert np.isclose(moved["score"],.5*np.sqrt(2)+.5)
    assert np.isclose(pooled["score"],2)
    assert np.isclose(moved_pooled["score"],2)
    assert moved["score"]>identity["score"]


def test_general_rule_does_not_predeclare_active_coordinate_or_dimension():
    q=np.array([.7,.3,0,0])
    a=dimension_loss(q,"auto_topk",complexity_penalty=.3)
    assert a["selected_k"]==2
    assert np.isclose(a["loss"],.15)
    P=plane_rotation(0,2,90)
    M=prepare(stats_four_groups())
    for recipe in ("U","U_UU","U_UU_UX"):
        for metric in ("lp","auto_topk"):
            for agg in ("stratified","pooled"):
                x=evaluate(M,np.eye(4),recipe=recipe,metric=metric,aggregation=agg)
                y=evaluate(M,P,recipe=recipe,metric=metric,aggregation=agg)
                assert np.isclose(x["score"],y["score"],atol=1e-11), (recipe,metric,agg)


def test_exact_hessian_row_energy_second_moment_contraction():
    rng=np.random.default_rng(81)
    G=np.stack([np.eye(4)]*4)
    Hs=rng.normal(size=(16,4,4))
    Hs=(Hs+Hs.transpose(0,2,1))/2
    H4=np.einsum("nab,ncd->abcd",Hs,Hs)/len(Hs)
    s=stats_four_groups()
    s["G_cov"]=G
    s["H4"][:]=H4[None]
    M=prepare(s)
    R=plane_rotation(0,3,37)
    actual=np.einsum("ia,nab,jb->nij",R,Hs,R)
    exact=np.mean(np.sum(actual**2,axis=-1),axis=0)
    computed=np.diag(R@M["components"]["UU"][0]@R.T)
    assert np.allclose(computed,exact,atol=1e-12,rtol=1e-12)


def test_ux_support_and_tangent_do_not_fabricate_x_boundary_derivative():
    raw=stats_four_groups()
    raw["C_support"]["tangent"][:,0]=0.
    # Even if the stale conditional-covariance slot is nonzero, a support
    # fraction of zero must contribute precisely nothing.
    M=prepare(raw)
    assert np.allclose(M["components"]["UX"],0.)
    x=evaluate(M,np.eye(4),recipe="U_UU_UX")
    assert all(v["active_components"]==["U"] for v in x["by_condition"])
    assert np.isfinite(x["score"])


def test_identical_universal_rule_for_every_condition_and_mixture():
    s=stats_four_groups()
    s["counts"]=np.array([60,60,40,340])
    M=prepare(s)
    for balance in (0, .5, 1):
        w=mixing_weights(M["counts"],balance)
        assert np.isclose(w.sum(),1)
        x=evaluate(M,np.eye(4),recipe="U",metric="lp",balance=balance)
        assert np.isclose(x["score"],1)
        assert len({v["loss"] for v in x["by_condition"]})==1
    assert np.allclose(mixing_weights(M["counts"],0),
                       np.array([.12,.12,.08,.68]))
    assert np.allclose(mixing_weights(M["counts"],1),np.ones(4)/4)
    with pytest.raises(ValueError):
        mixing_weights(M["counts"],1.5)


def test_moments_reusable_from_npz_and_cli_output_is_isolated(tmp_path):
    source=tmp_path/"source"
    (source/"logit").mkdir(parents=True)
    save_moments(source/"logit/moments.npz",stats_four_groups())
    output=tmp_path/"new_group_only"
    args=[
        "--source-dir",str(source),"--output-dir",str(output),
        "--tag","test","--channels","logit",
        "--random-rotations","3","--no-optimize","--no-plots",
        "--balances","0","1",
    ]
    main(args)
    assert (output/"logit/rotation_scores.csv").exists()
    assert (output/"logit/best_rotations.json").exists()
    assert (output/"logit/moment_summaries.json").exists()
    assert (output/"REPORT.md").exists()
    report=json.loads((output/"logit/best_rotations.json").read_text())
    assert "U__lp__balance0__stratified" in report
    assert "U__lp__balance0__pooled" in report
    assert report["U__lp__balance0__stratified"]["identity_score"]==pytest.approx(1)
    assert report["U__lp__balance1__pooled"]["identity_score"]==pytest.approx(2)
    # Existing output tagged protocol is protected.
    with pytest.raises(RuntimeError,match="different config"):
        main(args + ["--power",".4"])
