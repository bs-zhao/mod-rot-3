import unittest
import numpy as np
from strict_hessian import HessianLoss,CATALOG,penalty


class StrictHessianTests(unittest.TestCase):
    def test_unique_graph_counts_one_bilinear_edge_vs_two_self_curvatures(self):
        n=10;h=np.zeros((n,3,3));h[:,0,1]=h[:,1,0]=1.
        fields=dict(h=h,strict_x=np.linspace(-1,1,n)[:,None])
        loss=HessianLoss(fields,np.eye(2),2,'hs_b02_unique_edges_l1')
        r=np.array([[1.,1.],[-1.,1.]])/np.sqrt(2)
        self.assertAlmostEqual(loss(r),2*loss(np.eye(2)),places=12)

    def test_batched_single_metrics_match_independent_losses_across_rotations(self):
        from single_metric_atlas import BatchLoss,canonical_singles
        rng=np.random.default_rng(21);n=12;d=4;p=3
        h=rng.normal(size=(n,d+p,d+p));h=(h+h.transpose(0,2,1))/2
        fields=dict(h=h,uu_h=h[:7,:d,:d],uu_t=rng.normal(size=(7,d,d,d)),
            strict_x=rng.normal(size=(n,p)),strict_subject_groups=np.repeat([0,1],6))
        fields.update(strict_rank=np.repeat([.1,.8],6),uu_rank=np.linspace(.05,.95,7))
        fields['strict_z']=np.repeat(rng.normal(size=(2,d)),6,axis=0)
        names,aliases=canonical_singles();cov=np.eye(d);batch=BatchLoss(fields,cov,d,names)
        for _ in range(2):
            r,_=np.linalg.qr(rng.normal(size=(d,d)))
            shared=batch(r)
            for name in names:
                self.assertAlmostEqual(shared[name],HessianLoss(fields,cov,d,name)(r),places=11,msg=name)
        self.assertTrue(aliases)

    def test_full_hessian_uses_common_points_and_transforms_before_summarizing(self):
        rng=np.random.default_rng(31);d=4;p=3;n=12
        h=rng.normal(size=(n,d+p,d+p));h=(h+h.transpose(0,2,1))/2
        fields=dict(h=h,uu_h=h[:5,:d,:d],strict_x=rng.normal(size=(n,p)))
        loss=HessianLoss(fields,np.eye(d),d,'hs_full_q90_l1')
        r,_=np.linalg.qr(rng.normal(size=(d,d)))
        uu=np.einsum('ia,nab,jb->nij',r,h[:,:d,:d],r)
        ux=np.einsum('ia,nap->nip',r,h[:,:d,d:])
        expected=np.quantile(abs(np.concatenate((uu,ux),axis=-1)),.9,axis=0).mean()/loss.common_scale
        self.assertAlmostEqual(loss(r),expected,places=12)

    def test_tail_matches_the_history_examples(self):
        a=np.array([2.,2.,.01,.001]);b=np.array([1.,.5,.01,.01])
        for score in ['tail1','tail2sq','relative_tail2']:
            self.assertLess(penalty(a,score,k=2),penalty(b,score,k=2))

    def test_entire_catalogue_scale_sign_and_permutation_invariance(self):
        rng=np.random.default_rng(14);n=24;d=4;p=3
        h=rng.normal(size=(n,d+p,d+p));h=(h+h.transpose(0,2,1))/2
        fields=dict(h=h,strict_x=rng.normal(size=(n,p)),third=rng.normal(size=(n,d+p,d+p,d+p)),strict_subject_groups=np.zeros(n,dtype=int))
        fields['strict_rank']=np.full(n,.1)
        fields['strict_z']=np.zeros((n,d))
        q,_=np.linalg.qr(rng.normal(size=(d,d)));base=rng.normal(size=(d,d));cov=base@base.T+np.eye(d)
        changed=(np.array([.001,-20.,.3,-100.])[:,None]*q)[[3,1,0,2]]
        for name in CATALOG:
            loss=HessianLoss(fields,cov,d,name)
            self.assertAlmostEqual(loss(q),loss(changed),places=9,msg=name)

    def test_interaction_can_turn_into_self_curvature(self):
        d=2;n=20;h=np.zeros((n,3,3));h[:,0,1]=h[:,1,0]=1
        fields=dict(h=h,strict_x=np.linspace(-1,1,n)[:,None])
        r=np.array([[1.,1.],[-1.,1.]])/np.sqrt(2)
        off=HessianLoss(fields,np.eye(d),d,'hs_uu_off_rms_l1')
        own=HessianLoss(fields,np.eye(d),d,'hs_uu_self_rms_l1')
        self.assertLess(off(r),off(np.eye(d)))
        self.assertGreater(own(r),own(np.eye(d)))

    def test_curvature_modulation_detects_bilinear_rotation_that_own_third_misses(self):
        # f=z0*z1*z2: rotate only z0/z1. Pure own third remains zero,
        # but d_z2 H_ii exposes the self curvature produced by that rotation.
        n=8;d=3;h=np.zeros((n,4,4));h[:,0,1]=h[:,1,0]=1
        t=np.zeros((n,3,3,3))
        import itertools
        for ijk in itertools.permutations(range(3)):t[(slice(None),*ijk)]=1
        fields=dict(h=h,third=t,strict_x=np.linspace(-1,1,n)[:,None])
        r=np.eye(d);r[:2,:2]=np.array([[1.,1.],[-1.,1.]])/np.sqrt(2)
        own=HessianLoss(fields,np.eye(d),d,'hs_uu_third_self_l1')
        modulation=HessianLoss(fields,np.eye(d),d,'hs_uu_third_repeat_l1')
        self.assertAlmostEqual(own(r),own(np.eye(d)),places=12)
        self.assertGreater(modulation(r),modulation(np.eye(d)))

    def test_subject_averaging_before_and_after_outer_product_are_distinct(self):
        n=24;h=np.zeros((n,3,3));h[:12,0,2]=h[:12,2,0]=1.;h[12:,1,2]=h[12:,2,1]=1.
        fields=dict(h=h,strict_x=np.tile(np.linspace(-1,1,12),2)[:,None],strict_subject_groups=np.repeat([0,1],12))
        late=HessianLoss(fields,np.eye(2),2,'hs_jd_profile1_strength0_selfl1_w0')
        early=HessianLoss(fields,np.eye(2),2,'hs_jd_mean_profile1_strength0')
        r=np.array([[1.,1.],[-1.,1.]])/np.sqrt(2)
        self.assertAlmostEqual(late(np.eye(2)),late(r),places=10)
        self.assertGreater(early(np.eye(2)),early(r)+.1)


if __name__=='__main__':unittest.main()
