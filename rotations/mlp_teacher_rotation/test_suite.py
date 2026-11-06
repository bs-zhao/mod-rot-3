"""Fast package-local tests: soft labels, tensor rotation, direct MLP gradients."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig, DiscreteNeuralProcess
from .surrogate import make_proxy
from .teacher import evaluate_soft_teacher, direct_fingerprints
from .optimization import MetricBank, ALL_OBJECTIVES, rotation_from_angles, optimize_one

HERE=Path(__file__).resolve().parent


def cfg(**updates):
    c=json.loads((HERE/"config.json").read_text())
    c.update(seed=11,anchor_subjects=16,direct_anchor_subjects=10,
             direct_anchor_x=8,teacher_batch_subjects=10,
             direct_batch_subjects=5,starts=1,maxiter=2,
             make_plots=False,full_mdl=False)
    c.update(updates)
    return c


def soft_data():
    rng=np.random.default_rng(12)
    n,t,k=72,15,3
    q=rng.normal(size=(n,k))
    x=rng.normal(size=(n,t,3)).astype("float32")
    logit=.5*q[:,0,None]*x[:,:,0]+.3*q[:,1,None]**2*x[:,:,1]
    y=1/(1+np.exp(-logit))
    mask=np.ones_like(y,dtype=bool)
    return q,x,y,mask


def test_soft_teacher_profile_is_valid_and_all_objectives_finite():
    q,x,y,mask=soft_data()
    proxy=make_proxy(q,x,y,mask,cfg())
    rng=np.random.default_rng(23)
    gp=rng.normal(size=(10,3,8))
    gl=gp*1.7
    bank=MetricBank(proxy,{"probability":gp,"logit":gl},cfg())
    assert len(ALL_OBJECTIVES)==19
    R=rotation_from_angles(np.array([.1,.2,-.3]),3)
    assert np.allclose(R@R.T,np.eye(3),atol=1e-9)
    for name in ALL_OBJECTIVES:
        assert np.isfinite(bank.score(R,name)),name
    rot,info=optimize_one(bank,"teacher_jd_probability",cfg())
    assert np.allclose(rot@rot.T,np.eye(3),atol=1e-7)
    assert "identity_loss" in info


def test_exact_decoder_tangent_matches_probability_chain_rule():
    rng=np.random.default_rng(9)
    k_native,k=5,3
    z=rng.normal(size=(12,k_native)).astype("float32")
    x=rng.normal(size=(12,11,2)).astype("float32")
    mask=np.ones((12,11),dtype=bool)
    source=dict(z_fit=z,x_fit=x,mask_fit=mask,
                y_fit=rng.integers(0,2,size=mask.shape),
                rotation_dim=k,
                pca=SimpleNamespace(components_=np.eye(k_native)[:k],
                                    explained_variance_=np.ones(k)),
                whiten_floor=1e-5)
    model=DiscreteNeuralProcess(DiscreteNPConfig(
        x_dim=2,latent_dim=k_native,decoder_hidden_dims=(12,12)))
    model.eval().requires_grad_(False)
    soft,summary=evaluate_soft_teacher(model,torch.device("cpu"),source,cfg())
    assert np.all((soft>=0)&(soft<=1))
    assert summary["teacher_valid_trials"]==mask.sum()
    gradients=direct_fingerprints(model,torch.device("cpu"),source,cfg())
    assert gradients["probability"].shape==(10,k,8)
    assert gradients["logit"].shape==(10,k,8)
    # Independent finite differences check the logit tangent of the first axis.
    sid=int(gradients["anchor_fit_subject_ids"][0])
    x0=torch.tensor(gradients["common_x"])
    step=1e-3
    direction=gradients["dz_dq"][0]
    def logodds(vec):
        a=model.decode(x0[None],torch.tensor(vec[None],dtype=torch.float32))[0]
        return (a[:,1]-a[:,0]).detach().numpy()
    approx=(logodds(z[sid]+step*direction)-logodds(z[sid]-step*direction))/(2*step)
    assert np.allclose(gradients["logit"][0,0],approx,rtol=.08,atol=5e-3)
