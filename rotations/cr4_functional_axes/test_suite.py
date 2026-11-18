"""CR4 suite function-based rotation, AD and invariance regression tests."""
import numpy as np
import torch
from rotations.cr4_functional_axes.objective import (
    gain_axis,fit_functional_axes,curvature_value,perpendicular_frame)
from rotations.cr4_functional_axes.ux_audit import ux_diagnostics,ux_field
from rotations.cr4_functional_axes.fields import extract


def toy(seed,n=80,t=80,Q=None):
    """Eta=tau*(g-lambda*l + 1.2*tanh(gamma*p)); theta never enters fit."""
    rng=np.random.default_rng(seed)
    lam=rng.uniform(.6,2.3,(n,1))
    gam=rng.uniform(.55,1.4,(n,1))
    tau=rng.uniform(.9,3.2,(n,1))
    x=rng.uniform(-2.5,2.5,(t,3))
    g,l,p=x.T
    tanh=np.tanh(gam*p[None,:])
    dt=1-tanh*tanh
    eta=tau*(g[None,:]-lam*l[None,:]+1.2*tanh)
    G=np.stack((-tau*np.broadcast_to(l,eta.shape),
                1.2*tau*dt*p[None,:],eta/tau),axis=-1)
    H=np.zeros((n,t,3,3))
    H[...,0,2]=H[...,2,0]=-l[None,:]
    H[...,1,1]=1.2*tau*(-2*tanh*dt)*(p[None,:]**2)
    H[...,1,2]=H[...,2,1]=1.2*dt*p[None,:]
    if Q is not None:
        G=np.einsum("ij,stj->sti",Q,G)
        H=np.einsum("ia,stab,jb->stij",Q,H,Q)
    return dict(eta=eta,G=G,H=H)


def so3(seed):
    q,r=np.linalg.qr(np.random.default_rng(seed).normal(size=(3,3)))
    q=q@np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q)<0:q[:,0]*=-1
    return q


def config():
    return dict(gain_ridge=1e-4,gain_fit_max_ratio=.2,
        gain_valid_max_ratio=.3,gain_min_gap=.04,
        gain_min_energy_share=.02,min_curvature_gap=.0001,curvature_grid=91)


def test_ideal_analytic_function_recovers_three_rotated_axes():
    Q=so3(17)
    f=toy(1,Q=Q)
    v=toy(2,n=38,t=48,Q=Q)
    gain=gain_axis(f["eta"],f["G"])
    assert abs(gain["axis"]@Q[:,2])>.999
    R,diag,landscape=fit_functional_axes(f,v,config())
    assert np.allclose(R@R.T,np.eye(3),atol=1e-8)
    assert np.linalg.det(R)>0
    assert abs(R[2]@Q[:,2])>.999
    assert abs(R[0]@Q[:,0])>.98
    assert abs(R[1]@Q[:,1])>.98
    assert diag["gain_gate_pass"]
    assert diag["fit_low_curvature"]<diag["fit_complement_curvature"]
    assert len(landscape["angle_deg"])==91


def test_degenerate_weak_gain_axis_is_not_accepted():
    rng=np.random.default_rng(7)
    eta=rng.normal(size=(20,30))
    G=np.zeros((20,30,3))
    G[...,0]=eta
    G[...,1]=2*eta
    g=gain_axis(eta,G)
    assert g["energy_share"]<.02 or g["gap"]<.04


def test_curvature_and_ux_orthogonal_equivariance():
    rng=np.random.default_rng(8)
    H=rng.normal(size=(8,7,3,3));H=(H+H.swapaxes(-1,-2))/2
    R=so3(4)
    rotated=np.einsum("ia,...ab,jb->...ij",R,H,R)
    v=rng.normal(size=3);v/=np.linalg.norm(v)
    assert np.isclose(curvature_value(H,v),curvature_value(rotated,R@v),rtol=1e-9)
    assert np.allclose(perpendicular_frame(v).T@v,0,atol=1e-10)
    audit=ux_diagnostics(rng.normal(size=(5,9,3,6)),R)
    assert audit["X_energy_invariant_to_R"]
    assert np.allclose(audit["X_energy_fraction_rotated"],audit["X_energy_fraction_identity"],atol=1e-10)


class ToyDecoder(torch.nn.Module):
    def decode(self,x,z):
        eta=(z[:,0,None]*x[...,0]*x[...,1] + z[:,1,None]*x[...,1]**2 +
             z[:,2,None]*(x[...,2]+x[...,3]))
        return torch.stack((torch.zeros_like(eta),eta),dim=-1)


def test_frozen_nn_forward_mode_gradients_hessian_and_ux():
    torch.set_num_threads(1)
    rng=np.random.default_rng(55)
    z=rng.normal(size=(9,3));x=rng.uniform(-1,1,size=(25,6))
    src=dict(z=z,B=np.eye(3),bank=x)
    model=ToyDecoder()
    a=extract(model,torch.device("cpu"),src,np.arange(6),np.arange(11),
              np.zeros(6),np.ones(6),batch_subjects=3)
    assert a["G"].shape==(6,11,3) and a["H"].shape==(6,11,3,3)
    assert np.max(np.abs(a["H"]))<1e-7
    expected=np.stack((x[:11,0]*x[:11,1],x[:11,1]**2,x[:11,2]+x[:11,3]),axis=-1)
    np.testing.assert_allclose(a["G"],np.broadcast_to(expected,a["G"].shape),atol=1e-6)
    ux=ux_field(model,torch.device("cpu"),src,np.arange(4),np.arange(7),
                np.zeros(6),np.ones(6),batch_subjects=2)
    assert ux.shape==(4,7,3,6)
    np.testing.assert_allclose(ux[...,0,0],np.broadcast_to(x[:7,1],(4,7)),atol=1e-6)
    np.testing.assert_allclose(ux[...,0,1],np.broadcast_to(x[:7,0],(4,7)),atol=1e-6)
    np.testing.assert_allclose(ux[...,1,1],np.broadcast_to(2*x[:7,1],(4,7)),atol=1e-6)
    np.testing.assert_allclose(ux[...,2,2],np.ones((4,7)),atol=1e-6)
    np.testing.assert_allclose(ux[...,2,3],np.ones((4,7)),atol=1e-6)
