"""CR3 4P rotations and X-standardization compatibility with CR4/5."""
from __future__ import annotations

from types import SimpleNamespace
import json
import numpy as np
import pytest

from rotations.cr4_functional_axes.objective_cr3_4d import (
    fit_four_axes,sphere_starts,quartic,curvature_tensor)
from rotations.cr4_functional_axes.evaluation import parameter_diagnostics
from rotations.cr4_functional_axes.run import read_config


def _synthetic(seed,subjects=32,trials=40,Q=None):
    """Structured 4D derivatives with a unique gain axis and curvatures."""
    rng=np.random.default_rng(seed)
    eta=rng.normal(size=(subjects,trials))
    grad=rng.normal(size=(subjects,trials,4))
    grad[...,3]=eta
    # diagonal q-curvature energies separated by large margins
    H=np.zeros((subjects,trials,4,4),dtype=float)
    H[...,0,0]=.02*rng.normal(size=(subjects,trials))
    H[...,1,1]=.35*rng.normal(size=(subjects,trials))
    H[...,2,2]=2.1*rng.normal(size=(subjects,trials))
    H[...,3,3]=.15*rng.normal(size=(subjects,trials))
    if Q is not None:
        grad=np.einsum("ia,sta->sti",Q,grad)
        H=np.einsum("ia,stab,jb->stij",Q,H,Q)
    return dict(eta=eta,G=grad,H=H)


def _cfg():
    return dict(gain_ridge=1e-4,gain_fit_max_ratio=.2,
        gain_valid_max_ratio=.3,gain_min_gap=.06,
        gain_min_energy_share=.02,min_curvature_gap=.001,
        curvature_grid=91,sphere_points=200)


def test_cr3_so4_uses_a_real_sphere_search_and_returns_so4():
    rng=np.random.default_rng(6)
    Q,_=np.linalg.qr(rng.normal(size=(4,4)))
    if np.linalg.det(Q)<0:Q[:,0]*=-1
    fit=_synthetic(1,Q=Q)
    val=_synthetic(2,subjects=25,trials=29,Q=Q)
    R,metrics,landscape=fit_four_axes(fit,val,_cfg())
    assert R.shape==(4,4)
    np.testing.assert_allclose(R@R.T,np.eye(4),rtol=0,atol=1e-8)
    assert np.linalg.det(R)>0
    assert abs(R[3]@Q[:,3])>.995
    assert metrics["method"].startswith("CR3_SO4")
    assert metrics["first_curvature_sphere"]["sphere_points"]==200
    assert len(landscape["angle_deg"])==91
    assert np.isfinite(metrics["fit_gain_ratio"])
    assert metrics["fit_gain_ratio"]<.05
    assert len(metrics["fit_curvature_by_axis"])==4


def test_fibonacci_sphere_and_curvature_positive():
    starts=sphere_starts(101)
    assert starts.shape==(101,3)
    np.testing.assert_allclose(np.linalg.norm(starts,axis=1),1.,atol=1e-12)
    H=np.zeros((3,8,4,4))
    H[...,0,0]=2.
    T=curvature_tensor(H,np.eye(4)[:,:3])
    assert quartic(T,[1,0,0])>quartic(T,[0,1,0])
    with pytest.raises(ValueError,match="sphere_points"):
        sphere_starts(4)


def _eval_setup(tmp_path,dim):
    rng=np.random.default_rng(9)
    n=70
    theta=rng.normal(size=(n,4))
    U=theta[:,:dim].copy()
    source=dict(dataset=tmp_path/"dataset.npz",model="CR3",
                tr=np.arange(40),te=np.arange(40,n),q=U)
    np.savez_compressed(source["dataset"],parameters=theta)
    (tmp_path/"fit_complete.json").write_text('{"R_frozen":true}')
    np.savez_compressed(tmp_path/"rotation.npz",R=np.eye(dim))
    return source,U


def test_cr3_evaluation_four_axes_and_three_axis_partial(tmp_path):
    for dim in (3,4):
        folder=tmp_path/f"k{dim}"
        folder.mkdir()
        src,U=_eval_setup(folder,dim)
        scores=parameter_diagnostics(src,U,np.eye(dim),folder)
        result=scores["candidate"]
        assert result["total_parameters"]==4
        assert result["matched_parameters"]==dim
        assert len(result["signed_all_corr"])==4
        assert len(result["signed_all_corr"][0])==dim
        assert len(result["unmatched_parameters"])==4-dim
        assert result["full_parameter_axis_recovery_possible"]==(dim==4)
        assert abs(result["mean_all_r2"]-1)<1e-10
        assert (folder/"candidate_all_correlations.png").is_file()


def test_source_transform_must_not_accept_legacy_raw_cr3(monkeypatch):
    """A missing standardization cannot silently be replaced by raw X."""
    from rotations.cr4_functional_axes import teacher
    from unittest.mock import patch
    fake=dict(model="CR3",checkpoint="unused",root="unused",
              z=np.zeros((5,4)))
    with patch.object(teacher,"load_checkpoint",return_value={
        "extra_state":{},"model_config":{}}):
        with pytest.raises(RuntimeError,match="decoder-X standardization"):
            teacher.restore(fake,device="cpu")
