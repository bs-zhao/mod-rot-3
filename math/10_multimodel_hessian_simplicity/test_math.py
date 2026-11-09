"""Independent math/10 smoke and mathematical invariance tests.

Execute:
python -m unittest discover -s math/10_multimodel_hessian_simplicity -p 'test_*.py' -v
"""
from __future__ import annotations
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from oracle import derivatives, values, DIMENSIONS
from metrics import tail, vector_scores, compute_metrics, aggregate
from run import models_from_args, rotation, rotate_h, main, load_samples
import run as atlas_run
from unittest import mock


class TailPreferenceTests(unittest.TestCase):
    def test_primary_user_preference(self):
        A=np.array([2,2,.01,.001])
        B=np.array([1,.5,.01,.01])
        for p in (1,2):
            self.assertLess(tail(A,2,p),tail(B,2,p))
            self.assertLess(tail(A,2,p,True),tail(B,2,p,True))
        # Total L1/L2 DO NOT satisfy the same preference.
        self.assertGreater(np.linalg.norm(A,1),np.linalg.norm(B,1))
        self.assertGreater(np.linalg.norm(A,2),np.linalg.norm(B,2))

    def test_small_residual_preference(self):
        A=[2,.011,.001]
        B=[2,.001,.001]
        self.assertGreater(tail(A,1,1),tail(B,1,1))
        self.assertGreater(tail(A,1,2),tail(B,1,2))

    def test_hoyer_and_tail_can_disagree(self):
        a=[2,2,.01,.001]
        b=[1,.5,.01,.01]
        sA=vector_scores(a,1)
        sB=vector_scores(b,1)
        self.assertLess(sA["tail2_k2_loss"],sB["tail2_k2_loss"])
        self.assertGreater(sA["hoyer_loss"],sB["hoyer_loss"])


class OracleDerivativeTests(unittest.TestCase):
    def test_all_models_exact_tx_and_tt(self):
        rng=np.random.default_rng(20261109)
        for model,(d,p) in DIMENSIONS.items():
            with self.subTest(model=model):
                t=(rng.normal(0,.2,(5,d)) if model!="CR3" else
                   np.tile([.82,1.8,.65,2.2],(5,1))+rng.normal(0,.02,(5,4)))
                x=(rng.uniform(-.9,.9,(5,p)) if model!="CR3" else
                   np.tile([25.,15.,.3,50.,10.,.7],(5,1)))
                a=derivatives(t,x,model)
                self.assertEqual(a["tt_logit"].shape,(5,d,d))
                self.assertEqual(a["tx_probability"].shape,(5,d,p))
                self.assertTrue(np.isfinite(a["tx_probability"]).all())
                h=2e-5
                for i in (0,d-1):
                    for j in (0,p-1):
                        tp=t.copy();tm=t.copy()
                        xp=x.copy();xm=x.copy()
                        tp[:,i]+=h;tm[:,i]-=h
                        xp[:,j]+=h;xm[:,j]-=h
                        num=(values(tp,xp,model)[0]-values(tp,xm,model)[0]
                             -values(tm,xp,model)[0]+values(tm,xm,model)[0])/(4*h*h)
                        np.testing.assert_allclose(num,a["tx_logit"][:,i,j],atol=1e-3,rtol=2e-3)
                if model in ("C1","C2","C3"):
                    self.assertTrue(np.all(a["tt_logit"]==0))

    def test_sigmoid_tt_chain(self):
        rng=np.random.default_rng(43)
        t=rng.normal(0,.5,(11,6))
        x=rng.uniform(-1,1,(11,4))
        a=derivatives(t,x,"C3")
        self.assertTrue(np.allclose(a["tt_probability"],a["tt_probability"].transpose(0,2,1)))
        self.assertTrue(np.any(np.abs(a["tt_probability"])>0))

    def test_rotation_isometry_and_signed_mean_equivariance(self):
        rng=np.random.default_rng(42)
        d,p,n=6,4,32
        h=rng.normal(size=(n,d,d))
        h=(h+h.transpose(0,2,1))/2
        g=rng.normal(size=(n,d,p))
        r=rotation(d,1,4,34)
        hr,gr=rotate_h(h,g,r)
        np.testing.assert_allclose(np.mean(np.sum(hr*hr,axis=(1,2))),
                                   np.mean(np.sum(h*h,axis=(1,2))),rtol=1e-12)
        np.testing.assert_allclose(np.mean(np.sum(gr*gr,axis=(1,2))),
                                   np.mean(np.sum(g*g,axis=(1,2))),rtol=1e-12)
        np.testing.assert_allclose(aggregate(hr)["signed"],
                                   r@aggregate(h)["signed"]@r.T,rtol=1e-12)

    def test_zero_logit_zz_is_flat_not_a_success(self):
        rng=np.random.default_rng(4)
        t=rng.normal(0,.5,(20,4))
        x=rng.uniform(-1,1,(20,3))
        a=derivatives(t,x,"C1")
        self.assertTrue(np.array_equal(a["tt_logit"],np.zeros((20,4,4))))
        h,g=rotate_h(a["tt_logit"],a["tx_logit"],rotation(4,0,1,45))
        score=compute_metrics(h,g)
        self.assertEqual(score["zz_rms_offdiag_l2_loss"],0)
        self.assertGreater(score["zx_rms_total_l2_loss"],0)


class IntegrationSmoke(unittest.TestCase):
    def test_cr1_alias_warns_without_claiming_a_new_generator(self):
        with self.assertWarns(UserWarning):
            models,alias=models_from_args(["C1","CR1"])
        self.assertEqual(models,["C1","CR3"])
        self.assertTrue(alias)

    def test_all_four_models_saved_s1_end_to_end(self):
        """Fake S1 files ONLY in temporary test root; no user's dataset touched."""
        rng=np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for model,(d,p) in DIMENSIONS.items():
                n=4
                if model=="CR3":
                    t=np.tile([.8,1.8,.7,2.0],(n,1))+rng.normal(0,.03,(n,d))
                    bank=np.column_stack((
                        rng.uniform(5,150,500),rng.uniform(5,150,500),
                        rng.uniform(.05,.95,500),rng.uniform(5,150,500),
                        rng.uniform(5,150,500),rng.uniform(.05,.95,500)))
                    bank[:35,0]=0   # undifferentiable amount trials
                    bank[:23,1]=0
                    x=np.tile(bank,(n,1))
                    trials=500
                    bp=root/"outputs"/"choice"/model/"s0_design"
                    bp.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(bp/"CR3_4P_fixed500.npz",X=bank)
                else:
                    t=rng.normal(0,.5,(n,d))
                    trials=9
                    x=rng.uniform(-1,1,(n*trials,p))
                pth=root/"outputs"/"choice"/model/"s1_generate"
                pth.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(
                    pth/"s1_generate__dataset.npz",
                    parameters=t,trial_counts=np.full(n,trials,dtype=int),
                    subject_index=np.repeat(np.arange(n),trials),
                    trial_index=np.tile(np.arange(trials),n),
                    x=x,y=np.zeros(len(x),int),predictive_true=np.full((len(x),1),.5))
            with mock.patch.object(atlas_run,"ROOT",root):
                cr3=load_samples("CR3",3,4,100)
                self.assertLess(cr3["valid_fraction"],1.)
                self.assertTrue(np.array_equal(
                    cr3["x"].reshape(3,4,6)[0],
                    cr3["x"].reshape(3,4,6)[1]))
                main(["--models","C1","C2","C3","CR3","--subjects","3",
                      "--trials-per-subject","3","--angles","0,45,90",
                      "--planes","12","--multiway","1","--no-higher","--plots","0",
                      "--output",str(root/"result")])
            out=root/"result"
            with (out/"cross_model_ranking.csv").open(newline="",encoding="utf-8") as f:
                global_ranks=list(csv.DictReader(f))
            self.assertGreater(len(global_ranks),10)
            self.assertTrue(any(int(r["full_coverage"]) for r in global_ranks))
            for model in DIMENSIONS:
                self.assertTrue((out/model/"curves.csv").is_file())
                self.assertTrue((out/model/"multiway_comparison.csv").is_file())

    def test_synthetic_c1_end_to_end_without_user_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            main(["--models","C1","--subjects","4","--trials-per-subject","3",
                  "--angles","0,45,90","--planes","12","--multiway","0",
                  "--no-higher","--plots","0","--synthetic-debug","--output",d])
            base=Path(d)/"C1"
            for file in ("curves.csv","screens.csv","ranking_by_model_channel.csv",
                         "metric_catalog.csv","metadata.json"):
                self.assertTrue((base/file).is_file(),file)
            with (base/"curves.csv").open(newline="",encoding="utf-8") as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),6)
            self.assertIn("zx_rms_row_tail2_k2_loss",rows[0])
            self.assertTrue((Path(d)/"all_models_summary.csv").exists())


if __name__=="__main__":
    unittest.main()
