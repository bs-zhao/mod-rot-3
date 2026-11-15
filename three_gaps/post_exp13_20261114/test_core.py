"""Numerical guards for transformations and the p-Hessian identity."""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import core as C
from torch.func import hessian,jacrev

class Algebra(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4); self.v=torch.randn(80,4,dtype=torch.float64)
    def test_probability_decomposition_and_rotation(self):
        w=torch.randn(7,dtype=torch.float64)
        eta=lambda w: (w[:4].sin().sum()+w[0]*w[4]+w[1]*w[2]*w[5])
        e=eta(w); g=jacrev(eta)(w); hh=hessian(eta)(w)
        p=e.sigmoid(); a=p*(1-p)*hh; b=p*(1-p)*(1-2*p)*g[:,None]*g[None,:]
        torch.testing.assert_close(hessian(lambda w:eta(w).sigmoid())(w),a+b,rtol=1e-10,atol=1e-10)
        r=C.rotations(4,1)[0]; sd=torch.diag(r@C.cov(self.v)@r.T).sqrt(); A=sd[:,None]*r
        f=lambda ux: eta(torch.cat([r.T@(sd*ux[:4]),ux[4:]]))
        u=torch.cat([(r@w[:4])/sd,w[4:]])
        self.assertLess(float((f(u)-eta(w)).abs()),1e-12)
        torch.testing.assert_close(hessian(f)(u)[:4,:4],A@hh[:4,:4]@A.T)
        torch.testing.assert_close(hessian(f)(u)[:4,4:],A@hh[:4,4:])
    def test_moments_and_signed_permutation(self):
        tx=torch.randn(3,5,4,3,dtype=torch.float64)
        tt=torch.randn(3,5,4,4,dtype=torch.float64); tt=(tt+tt.transpose(-1,-2))/2
        r=C.rotations(4,1)[0]; A=C.effective(r,C.cov(self.v)); ux,uu=C.energies(A,C.moments(tx,tt))
        torch.testing.assert_close(ux,torch.einsum('ia,skaj->skij',A,tx).square().mean((0,1)))
        torch.testing.assert_close(uu,torch.einsum('ia,skab,jb->skij',A,tt,A).square().mean((0,1)))
        perm=torch.eye(4,dtype=torch.float64)[[2,0,3,1]]; perm[0]*=-1
        vals=C.terms(A,C.moments(tx,tt)); vals2=C.terms(perm@A,C.moments(tx,tt))
        torch.testing.assert_close(vals,vals2)
    def test_no_truth_in_optimization_data(self):
        path=C.ROOT/'outputs/three_gaps/exp13'/f"cache2_C1_{C.condition('C1',1)}_prob.pt"
        if not path.exists(): self.skipTest('Local archived-cache integration check')
        c=C.load('C1',1,'prob',8,8)
        self.assertFalse(any(k in c for k in ('Tst','T','te','truth','oracle')))
    def test_X_standardization_cancels_unit_changes(self):
        path=C.ROOT/'outputs/three_gaps/exp13'/f"cache2_C1_{C.condition('C1',1)}_prob.pt"
        if not path.exists(): self.skipTest('Local archived-cache integration check')
        c=C.load('C1',1,'prob',8,8); factors=C.tensor([100.,.01,7.])
        original=C.moments(c['Htx']*c['X'].std(0),c['Htt'])
        rescaled=C.moments((c['Htx']/factors)*(c['X']*factors).std(0),c['Htt'])
        for a,b in zip(original,rescaled): torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)
    def test_batched_objective_gradient_and_zero(self):
        z=torch.zeros(4,4,dtype=torch.float64)
        self.assertTrue(torch.isfinite(C.neff(z))); self.assertEqual(float(C.share(z)),0)
        m=C.moments(torch.randn(2,3,4,3),torch.randn(2,3,4,4))
        o=C.Objective(self.v,lambda a:C.terms(a,m))
        r=C.rotations(4,3).requires_grad_(); loss=o(r); loss.sum().backward()
        self.assertTrue(torch.isfinite(r.grad).all())
        for i in range(3): torch.testing.assert_close(loss[i],o(r[i]))
    def test_finite_responses_batch_matches_scalar(self):
        sys.path.insert(0,str(C.HERE/'exp15_finite_scale'))
        from run import Finite
        from scipy.spatial.distance import cdist
        f=Finite.__new__(Finite); f.d=4; f.scales=(.1,.25,.5,1.)
        f.dev=torch.device('cpu'); f.v=self.v[:3]; f.x=torch.randn(8,3,dtype=torch.float64)
        f.cov=C.cov(self.v); f.train_cov_inv=torch.eye(4,dtype=torch.float64); f.support_threshold=1e9; f.support={}
        xs=f.x.numpy(); dis=cdist(xs,xs); np.fill_diagonal(dis,np.inf); f.pair=dis.argmin(1)
        f.distance=C.tensor(dis[np.arange(8),f.pair]); f.direction=C.tensor((xs[f.pair]-xs)/f.distance.numpy()[:,None])
        f.query=lambda v,x:(v[...,0]*v[...,1]*x[...,0]+v[...,2].square()+v[...,3].sin()*x[...,1]).reshape(-1)
        rr=C.rotations(4,3); batched=f.raw(rr)
        scalar=torch.stack([f.raw_one(r) for r in rr])
        torch.testing.assert_close(batched,scalar,rtol=1e-9,atol=1e-9)
        f.active_scales=[1]; selected=f.raw(rr)
        torch.testing.assert_close(selected[:,4:8],batched[:,4:8],rtol=1e-9,atol=1e-9)
        self.assertEqual(float(selected[:,:4].abs().sum()+selected[:,8:].abs().sum()),0.)

if __name__=='__main__': unittest.main()
