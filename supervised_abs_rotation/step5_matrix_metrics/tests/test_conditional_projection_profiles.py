import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import conditional_projection_profiles as projection


def test_innovation_matches_nested_least_squares_increment_and_context_orthogonality():
    rng=np.random.default_rng(909);context=rng.normal(size=(120,7));feature=rng.normal(size=(120,12))
    feature+=context@rng.normal(size=(7,12));target=rng.normal(size=(120,8))
    q,audit=projection.innovation_basis(torch.tensor(feature),torch.tensor(context))
    direct=np.column_stack((context,feature));full=direct@np.linalg.lstsq(direct,target,rcond=None)[0]
    lower=context@np.linalg.lstsq(context,target,rcond=None)[0]
    np.testing.assert_allclose(q.numpy()@(q.numpy().T@target),full-lower,atol=1e-13)
    assert audit['maximum_context_inner_product']<1e-12
    values,checks=projection.mode_energies(q,torch.tensor(target),[(0,torch.eye(8,dtype=torch.float64))])
    np.testing.assert_allclose(values[0][1].sum(),np.sum((full-lower)**2)/len(target),atol=1e-13)
    assert checks[0]['captured_energy_fraction']<=1.


def test_innovation_projection_is_unchanged_by_invertible_feature_coordinates_and_context_addition():
    rng=np.random.default_rng(910);context=rng.normal(size=(100,6));feature=rng.normal(size=(100,10))
    transform=np.linalg.qr(rng.normal(size=(10,10)))[0]@np.diag(np.linspace(.3,4.,10))
    changed=feature@transform+context@rng.normal(size=(6,10))
    first,_=projection.innovation_basis(torch.tensor(feature),torch.tensor(context))
    second,_=projection.innovation_basis(torch.tensor(changed),torch.tensor(context))
    np.testing.assert_allclose(first@first.T,second@second.T,atol=1e-13)


def test_numerical_context_residue_is_not_promoted_into_a_new_feature_space():
    rng=np.random.default_rng(911);context=rng.normal(size=(90,7));feature=context@rng.normal(size=(7,13))
    q,audit=projection.innovation_basis(torch.tensor(feature),torch.tensor(context))
    assert q.shape==(90,0) and audit['innovation_rank']==0


def test_PRESS_residuals_replay_explicit_leave_one_out_nested_fits():
    rng=np.random.default_rng(912);context=np.column_stack((np.ones(70),rng.normal(size=(70,3))))
    feature=rng.normal(size=(70,5));target=rng.normal(size=(70,4))
    q0=projection.basis(torch.tensor(context));q1,_=projection.innovation_basis(torch.tensor(feature),torch.tensor(context))
    lower,full,audit=projection.press_residuals(q0,q1,torch.tensor(target))
    direct=np.column_stack((context,feature));expected0=[];expected1=[]
    for i in range(len(target)):
        train=np.arange(len(target))!=i
        expected0.append(target[i]-context[i]@np.linalg.lstsq(context[train],target[train],rcond=None)[0])
        expected1.append(target[i]-direct[i]@np.linalg.lstsq(direct[train],target[train],rcond=None)[0])
    np.testing.assert_allclose(lower,expected0,atol=1e-13)
    np.testing.assert_allclose(full,expected1,atol=1e-13)
    values,checks=projection.press_mode_gains(q0,q1,torch.tensor(target),[(0,torch.eye(4,dtype=torch.float64))])
    check=checks['X_columns'][0]
    np.testing.assert_allclose(values[0][1].sum()-check['negative_gain_energy'],check['signed_risk_improvement'],atol=1e-13)


def test_weighted_ridge_PRESS_matches_explicit_penalized_refits():
    rng=np.random.default_rng(913);context=np.column_stack((np.ones(65),rng.normal(size=(65,3))))
    feature=rng.normal(size=(65,7));target=rng.normal(size=(65,4));ridge=.2
    design=np.column_stack((context,feature));penalty=np.diag([0.]*4+[ridge]*7)
    coefficient=np.linalg.solve(design.T@design+penalty,design.T@target)
    import conditional_model_quality as quality
    affine=context;residual=feature-affine@np.linalg.lstsq(affine,feature,rcond=None)[0]
    scale=np.sum(residual*residual)/feature.shape[1]
    train,PRESS,audit=quality.audit(torch.tensor(design),torch.tensor(coefficient),torch.tensor(target),4,ridge/scale)
    expected=[]
    for i in range(len(target)):
        keep=np.arange(len(target))!=i
        beta=np.linalg.solve(design[keep].T@design[keep]+penalty,design[keep].T@target[keep])
        expected.append(target[i]-design[i]@beta)
    np.testing.assert_allclose(PRESS,expected,atol=1e-13)
    np.testing.assert_allclose(train,target-design@coefficient,atol=1e-13)
    assert audit['relative_squared_prediction_replay_error']<1e-25
