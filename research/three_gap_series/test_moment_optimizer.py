import unittest
import numpy as np
import torch
from strict_hessian import HessianLoss,CATALOG
from single_metric_atlas import BatchLoss
from moment_optimizer import MomentObjective,rotation_torch


class MomentOptimizerTests(unittest.TestCase):
    def test_same_single_losses_and_correct_rotation_gradient(self):
        rng=np.random.default_rng(32);n=24;d=4;p=3
        h=rng.normal(size=(n,d+p,d+p));h=(h+h.transpose(0,2,1))/2
        fields=dict(h=h,strict_x=np.tile(rng.normal(size=(12,p)),(2,1)),strict_subject_groups=np.repeat([0,1],12))
        fields['strict_rank']=np.repeat([.1,.8],12)
        fields['strict_z']=np.repeat(rng.normal(size=(2,d)),12,axis=0)
        names=[n for n in CATALOG if n.startswith(tuple(f'hs_b{i:02d}_' for i in range(1,15)))]+['hs_ux_rms_soft003']
        base=rng.normal(size=(d,d));cov=base@base.T+np.eye(d)
        batch=BatchLoss(fields,cov,d,names);angles=rng.normal(size=d*(d-1)//2)
        r=rotation_torch(torch.tensor(angles),d).numpy()
        for name,loss in batch.losses.items():
            obj=MomentObjective(loss);value,gradient=obj.scipy(angles)
            self.assertAlmostEqual(value,loss(r),places=9,msg=name)
            for j in [0,3]:
                delta=np.eye(len(angles))[j]*1e-6
                approx=(loss(rotation_torch(torch.tensor(angles+delta),d).numpy())-loss(rotation_torch(torch.tensor(angles-delta),d).numpy()))/2e-6
                self.assertAlmostEqual(gradient[j],approx,places=5,msg=name)
        for loss in batch.losses.values():
            value,gradient=MomentObjective(loss).scipy(np.zeros(len(angles)))
            self.assertTrue(np.isfinite(gradient).all())

    def test_exact_zero_edges_keep_the_original_sparsity_value(self):
        rng=np.random.default_rng(24);d=4;p=3;n=12
        h=rng.normal(size=(n,d+p,d+p));h=(h+h.transpose(0,2,1))/2;h[:,0,:]=0;h[:,:,0]=0
        fields=dict(h=h,strict_x=np.tile(rng.normal(size=(6,p)),(2,1)),strict_subject_groups=np.repeat([0,1],6))
        fields['strict_rank']=np.repeat([.1,.8],6)
        fields['strict_z']=np.repeat(rng.normal(size=(2,d)),6,axis=0)
        names=[n for n in CATALOG if n.startswith(tuple(f'hs_b{i:02d}_' for i in range(1,15)))]+['hs_ux_rms_soft003']
        batch=BatchLoss(fields,np.eye(d),d,names)
        for name,loss in batch.losses.items():
            value,gradient=MomentObjective(loss).scipy(np.zeros(6))
            self.assertAlmostEqual(value,loss(np.eye(d)),places=9,msg=name)
            self.assertTrue(np.isfinite(gradient).all(),msg=name)


if __name__=='__main__':unittest.main()
