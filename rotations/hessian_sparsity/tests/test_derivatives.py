"""Frozen smooth NN derivatives remain computable at CR3/CR5 X endpoints.

This tests what the algorithm ACTUALLY computes (neural derivatives),
not the mathematically singular analytic CPT derivative.
"""
import numpy as np
import torch
from rotations.hessian_sparsity.derivatives import extract


class Toy(torch.nn.Module):
    def decode(self,x,z):
        eta=(z[:,0,None]*x[...,0]*x[...,2]+
             z[:,1,None]*x[...,1]**2+
             z[:,2,None]*x[...,4]*x[...,5])
        return torch.stack((torch.zeros_like(eta),eta),dim=-1)


def test_ad_uu_ux_boundary_and_original_x_unchanged():
    rng=np.random.default_rng(133)
    z=rng.normal(size=(5,3))
    x=rng.normal(size=(12,6))
    x[0,2]=0.
    x[1,2]=1.
    x[2,5]=0.
    x[3,5]=1.
    src=dict(z=z,bank_x=x,B=np.eye(3),group_codes=np.zeros(12,dtype=int))
    d=extract(Toy(),torch.device("cpu"),src,np.arange(4),np.arange(8),
              np.zeros(6),np.ones(6),include_ux=True,batch_size=2)
    assert d["UU"].shape==(4,8,3,3)
    assert d["UX"].shape==(4,8,3,6)
    assert np.allclose(d["UU"],0,atol=1e-6)
    np.testing.assert_allclose(
        d["UX"][...,0,0],np.broadcast_to(x[:8,2],(4,8)),atol=1e-6)
    np.testing.assert_allclose(
        d["UX"][...,0,2],np.broadcast_to(x[:8,0],(4,8)),atol=1e-6)
    np.testing.assert_allclose(
        d["UX"][...,1,1],np.broadcast_to(2*x[:8,1],(4,8)),atol=1e-6)
    np.testing.assert_allclose(
        d["UX"][...,2,4],np.broadcast_to(x[:8,5],(4,8)),atol=1e-6)
    np.testing.assert_allclose(
        d["UX"][...,2,5],np.broadcast_to(x[:8,4],(4,8)),atol=1e-6)
    assert x[0,2]==0 and x[1,2]==1
