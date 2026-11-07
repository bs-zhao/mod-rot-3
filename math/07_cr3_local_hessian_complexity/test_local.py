"""Tests for experiment 07's local derivative and sparsity objectives.

All tests are small and CPU-only. The end-to-end test creates a complete
step04-style fixture, fits the 6 original numerical Y and checks new outputs.
"""
from __future__ import annotations
import json
import tempfile
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from run import main
from local_metrics import complexity_metrics
from controlled_sweeps import sweep_grids,sweep_metrics
from analytic_reference import rotate_fields,reference_fields
from local_polynomial import polynomial_field_shake
from comparisons import compare_three_arrays
from pairwise_sweeps import pairwise_additivity
from fast_tensor import RotationTensor


def random_local_fields(N=5,M=7,seed=8):
    rng=np.random.default_rng(seed)
    fields={
      "value":rng.normal(size=(N,M)),
      "zz":rng.normal(size=(N,M,4,4)),
      "zx":rng.normal(size=(N,M,4,6)),
      "zzz":rng.normal(size=(N,M,4,4,4)),
      "zzx":rng.normal(size=(N,M,4,4,6)),
      "zxx":rng.normal(size=(N,M,4,6,6)),
      "zzzz":rng.normal(size=(N,M,4,4,4,4)),
      "zzxx":rng.normal(size=(N,M,4,4,6)),
      "zxxx":rng.normal(size=(N,M,4,6)),
    }
    return fields


def test_local_metrics_all_blocks_and_finite():
    data=random_local_fields()
    result=complexity_metrics(data,np.ones(4),np.ones(6),include_fourth=True)
    assert len(result)>200
    assert all(np.isfinite(float(v)) for v in result.values())
    expected=(
        "zz_self_z_third_rms_loss",
        "zz_other_z_third_rms_loss",
        "zz_diagonal_X_modulation_third_rms_loss",
        "zx_own_X_modulation_third_rms_loss",
        "zz_self_z_fourth_rms_loss",
        "zz_mean_flat_softl0_t0p005_loss",
        "zx_rms_zx_row_mean_tail2sq_k2_loss",
        "zx_local_row_mean_capped_l2_t0p005_loss",
    )
    for key in expected:
        assert key in result,key


def test_own_z_curvature_distinguishes_quadratic_from_cubic_with_x_effects():
    rng=np.random.default_rng(17)
    U=rng.normal(size=(80,4))
    anchors=np.array([3,10,15,24])
    grids,position=sweep_grids(U,anchors,levels=9)
    M=6
    factors=np.linspace(.3,1.8,M)
    results=[]
    for power in (2,3):
        fields=np.zeros((4,len(anchors),9,M))
        curved=np.zeros_like(fields)
        for i in range(4):
            z=grids[i,:,:,i]
            fields[i]=z[:,:,None]**power*factors
            curved[i]=(power*(power-1)*z[:,:,None]**(power-2))*factors
        ans=sweep_metrics(dict(value=fields,curvature=curved,position=position))
        results.append(ans)
    quad,cubic=results
    assert quad["sweep_mean_self_curvature_variance_loss"]<1e-16
    assert cubic["sweep_mean_self_curvature_variance_loss"]>.1
    assert quad["sweep_mean_quadratic_approx_relative_mse_loss"]<1e-13
    assert cubic["sweep_mean_quadratic_approx_relative_mse_loss"]>1e-7
    assert cubic["sweep_mean_cubic_approx_relative_mse_loss"]<1e-12


def test_rotate_full_tensors_preserves_shapes_and_orthogonal_energy():
    rng=np.random.default_rng(1)
    N,M=3,4
    data={k:np.asarray(v)[None,...] for k,v in random_local_fields(N,M).items()}
    angle=np.pi/6
    R=np.eye(4)
    R[:2,:2]=[[np.cos(angle),-np.sin(angle)],
              [np.sin(angle),np.cos(angle)]]
    rot=rotate_fields(data,R)
    for key in data:
        assert rot[key].shape==data[key].shape,(key,rot[key].shape)
        # Orthogonally transforming every parameter index preserves full
        # tensor Frobenius norm for each source and each fixed input.
        assert_allclose(np.linalg.norm(rot[key]),
                        np.linalg.norm(data[key]),rtol=1e-12,atol=1e-10)
    identity=rotate_fields(data,np.eye(4))
    for key in data:assert_allclose(identity[key],data[key],atol=1e-11)


def test_local_polynomial_exact_z_cubic_and_z2_x_modulation():
    rng=np.random.default_rng(2026)
    Z=rng.uniform(-1,1,size=(90,4))
    X=rng.uniform(.1,.9,size=(40,6))
    core=(Z[:,0,None]**3*X[None,:,2]+
          .4*Z[:,1,None]**2*X[None,:,5])
    Y=np.stack([(.6+c*.2)*core for c in range(6)],axis=-1)
    fit=RotationTensor(X,Y,np.arange(75),
                       degree_u=3,degree_x=1,
                       ridge_u=1e-10,ridge_x=1e-10,
                       heldout_ids=np.arange(75,90),validation_limit=6)
    polynomial,fields,_,_,_=polynomial_field_shake(
        fit,np.eye(4),Z,np.zeros(4),np.arange(75,82),X[:8],fourth=True)
    vals=fields["zzz"][0,...,0,0,0]
    true=6*.6*X[:8,2][None,:]
    assert_allclose(vals,true,atol=.03,rtol=.02)
    # d³f/dz1² dx5 = 2*.4*.6, independent of z1 and X.
    assert_allclose(fields["zzx"][0,...,1,1,5],.48,atol=.03,rtol=.03)



def test_pairwise_ZZ_additivity_gap_detects_cross_product():
    rng=np.random.default_rng(32)
    U=rng.uniform(-1,1,size=(50,4))
    X=rng.uniform(.1,.9,size=(7,6))
    class FakePoly:
        def __init__(self,interaction):
            self.X=X
            self.interaction=interaction
        def evaluate(self,Unew):
            v=Unew[:,0]*Unew[:,1] if self.interaction else (
                Unew[:,0]**2+Unew[:,1]**3)
            return np.broadcast_to(v[None,:,None],(6,len(Unew),len(X))).copy()
    class FakeOracle:
        def __init__(self,interaction):self.interaction=interaction
        def oracle_outputs(self,T,X):
            v=T[:,0]*T[:,1] if self.interaction else (
                T[:,0]**2+T[:,1]**3)
            values=np.broadcast_to(v[:,None],(len(T),len(X))).copy()
            return values,values
    separate=pairwise_additivity(FakePoly(False),U,[2,8],np.zeros(4),
                                 np.eye(4),FakeOracle(False),levels=5)
    interacting=pairwise_additivity(FakePoly(True),U,[2,8],np.zeros(4),
                                    np.eye(4),FakeOracle(True),levels=5)
    key="zz_pair_01_additivity_gap_loss"
    assert separate[("analytic","logit")][key]<1e-20
    assert interacting[("analytic","logit")][key]>.05
    assert interacting[("formula_y","logit")][key]>.05
    assert interacting[("nn_y","logit")][key]>.05


def test_three_way_gap_separates_estimator_vs_nn_fit():
    truth=np.arange(24,dtype=float).reshape(2,3,4)
    formula=truth+.5
    nn=formula-.25
    m=compare_three_arrays(truth,formula,nn,field="zz",target="probability")
    assert m["formula_fit_relative_error"]>m["nn_fit_minus_formula_fit_relative_error"]
    assert m["nn_fit_relative_error"]<m["formula_fit_relative_error"]
    assert not m["binary_choice_sampling_caveat"]


def test_analytic_CR3_local_fields_and_FD_stability():
    T=np.array([[.60,2.0,.65,2.2],
                [.78,2.4,.75,2.0]],dtype=float)
    X=np.array([[60.,25.,.3,40.,20.,.7],
                [35.,15.,.4,55.,26.,.6]],dtype=float)
    fields,cfg=reference_fields(T,X,fourth=True)
    assert fields["zz"].shape==(3,2,2,4,4)
    assert fields["zzzz"].shape==(3,2,2,4,4,4,4)
    assert np.isfinite(fields["zzzz"]).all()
    half,_=reference_fields(T,X,fourth=True,
                            tsteps=np.asarray(cfg["tstep"])/2,
                            xsteps=np.asarray(cfg["xstep"])/2)
    # Third derivatives should be stable under moderately finer FD.
    err=np.linalg.norm(fields["zzz"]-half["zzz"])
    norm=max(np.linalg.norm(half["zzz"]),1e-10)
    assert err/norm<.02


def test_step07_full_pipeline_writes_three_way_metric_outputs():
    from scipy.special import expit
    rng=np.random.default_rng(22)
    N,M=66,36
    Z=rng.normal(size=(N,4))
    A=np.array([[.09,.1,.02,.01],
                [.01,.12,.13,.02],
                [.01,.02,.08,.14],
                [.04,-.03,.05,.18]])
    b=np.array([.65,2.2,.7,2.1])
    T=Z@A.T+b
    X=np.column_stack((rng.uniform(5,100,size=M),
                       rng.uniform(5,75,size=M),
                       rng.uniform(.1,.9,size=M),
                       rng.uniform(5,100,size=M),
                       rng.uniform(5,75,size=M),
                       rng.uniform(.1,.9,size=M)))
    # Input Y = oracle logistic and hard/sampled choices; six channels.
    from analytic_reference import oracle_outputs
    eta,prob=oracle_outputs(T,X)
    Y=dict(oracle_choice=(rng.random((N,M))<prob).astype(float),
           oracle_logit=eta,oracle_probability=prob,
           nn_choice=(prob>.5).astype(float),
           nn_logit=eta*.92,nn_probability=expit(eta*.92))
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        source=root/"repeat01";source.mkdir()
        np.savez_compressed(source/"all_Y_and_mapping.npz",
            Z_all_trial=Z,T_true=T,X_fixed=X,A=A,b=b,
            T_hat_AZ_plus_b=T,**Y)
        eye=np.eye(10)
        np.savez_compressed(source/"all_matrices.npz",
            map_fit_ids=np.arange(N),
            sieve_train_ids=np.arange(48),
            sieve_test_ids=np.arange(48,N),
            analytic_logit__conditional_mean=eye,
            analytic_logit__fraction_defined=np.ones_like(eye),
            analytic_probability__conditional_mean=eye,
            analytic_probability__fraction_defined=np.ones_like(eye))
        (source/"summary.json").write_text('{"no_nn_hessian":true}')
        output=root/"results"
        result=main([
           "--source",str(source),"--output",str(output),
           "--planes","01","--angles","0,90",
           "--degree-u","2","--degree-x","2",
           "--eval-subjects","6","--eval-stimuli","6",
           "--anchors","2","--sweep-levels","6",
           "--ridge-u","1e-6","--ridge-x","1e-6",
           "--no-plots","--no-fd-convergence","--no-fourth",
           "--no-heldout-rmse"])
        assert result["total_local_derivative_methods"]==18
        assert result["metrics_count_per_method"]>200
        for name in ("all_local_metrics.csv","strict_screen_all.csv",
                     "three_way_local_derivative_errors.csv","summary.json"):
            assert (output/name).is_file(),name
        import csv
        with (output/"all_local_metrics.csv").open() as fp:
            rows=list(csv.DictReader(fp))
        assert len(rows)==18
        assert "sweep_mean_self_curvature_variance_normalized_loss" in rows[0]
