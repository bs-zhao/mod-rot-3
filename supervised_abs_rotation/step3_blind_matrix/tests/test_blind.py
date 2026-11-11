"""Synthetic numeric-WY-only checks of Step-3 blind matrix discovery."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import metrics
import profiles
spec = importlib.util.spec_from_file_location("blind_matrix_step3",HERE/"run.py")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name]=mod
spec.loader.exec_module(mod)


class TestStructuralScores(unittest.TestCase):
    def test_linear_B_is_simple_at_true_axes_and_invariant_at_90_degrees(self):
        rng = np.random.default_rng(70)
        u = rng.normal(size=(1800,4))
        true_B=np.zeros((4,16))
        true_B[np.arange(16)%4,np.arange(16)] = 1.
        f=u@true_B+rng.normal(scale=.003,size=(len(u),16))
        train=np.arange(1100)
        original,_,b=metrics.linear_fingerprint(u,f,train,ridge=.01)
        angle=np.deg2rad(45.)
        g=np.eye(4)
        g[:2,:2] = [[np.cos(angle),-np.sin(angle)],
                     [np.sin(angle),np.cos(angle)]]
        rotated,_,bb=metrics.linear_fingerprint(u@g.T,f,train,ridge=.01)
        g90=np.eye(4)
        g90[:2,:2]=[[0.,-1.],[1.,0.]]
        ninety,_,_=metrics.linear_fingerprint(u@g90.T,f,train,ridge=.01)
        for name in ("B_column_l1_loss","B_column_hoyer_loss",
                     "B_column_entropy_loss","B_column_top1_tail_loss"):
            self.assertLess(original[name],rotated[name],
                            f"{name} must favor sparse original axes")
            self.assertAlmostEqual(original[name],ninety[name],places=10)
        np.testing.assert_allclose(
            original["control_B_fro_norm"],
            rotated["control_B_fro_norm"],rtol=1e-10)
        np.testing.assert_allclose(bb,g@b,atol=1e-9)

    def test_poly_full_quadratic_is_nearly_rotation_invariant(self):
        rng=np.random.default_rng(179)
        u=rng.normal(size=(400,4))
        f=np.column_stack([
            u[:,0]*u[:,1],u[:,2]**2,
            u[:,0]**3, np.sin(u[:,3])])
        train=np.arange(270)
        test=np.arange(270,400)
        a,_=metrics.functional_scores(u,f,train,test,ridge=1e-4)
        phi=.6
        g=np.eye(4)
        g[:2,:2]=[[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]]
        b,_=metrics.functional_scores(u@g.T,f,train,test,ridge=1e-4)
        self.assertAlmostEqual(a["function_full_quad_mse"],
                               b["function_full_quad_mse"],delta=1e-5)
        self.assertAlmostEqual(a["function_full_cubic_mse"],
                               b["function_full_cubic_mse"],delta=1e-5)
        self.assertGreater(a["function_quad_interaction_gap"],.05)
        self.assertTrue(np.isfinite(b["function_budget_auc_loss"]))

    def test_no_signal_is_not_a_fake_low_complexity(self):
        f=np.ones((100,4))
        used,meta=metrics.normalize_response(f,np.arange(70))
        self.assertIsNone(used)
        self.assertEqual(meta["selected"],0)


class TestBlindEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.step2=self.root/"step2_pairwise_sweep"
        (self.step2/"cases").mkdir(parents=True)
        self.out=self.root/"step3_blind_matrix"
        rng=np.random.default_rng(355)
        self.n=72
        self.n_trials=7
        self.dz=4
        self.dx=6
        self.counts=np.full(self.n,self.n_trials,dtype=np.int64)
        self.sid=np.repeat(np.arange(self.n),self.n_trials)
        self.tid=np.tile(np.arange(self.n_trials),self.n)
        self.xbank=rng.normal(size=(self.n_trials,self.dx)).astype(np.float32)
        self.x=np.tile(self.xbank,(self.n,1))
        zraw=rng.normal(size=(self.n,self.dz))
        self.u0=(zraw-zraw.mean(0))/zraw.std(0)
        q,_=np.linalg.qr(rng.normal(size=(self.dz,self.dz)))
        self.r_best=q
        self.baseline=self.u0@q.T
        coef=rng.normal(size=(self.dz,self.n_trials))
        coef[1:,:3]=0
        response=self.baseline@coef
        response += .03*rng.normal(size=response.shape)
        flat=response.reshape(-1)
        p=1/(1+np.exp(-flat))
        np.save(self.step2/"subject_index.npy",self.sid)
        np.save(self.step2/"trial_index.npy",self.tid)
        np.save(self.step2/"trial_counts.npy",self.counts)
        np.savez_compressed(self.step2/"Y.npz",
            choice_observed=rng.binomial(1,p).astype(np.uint8),
            logit_nn=flat.astype(np.float32),
            p_nn=p.astype(np.float32),
            choice_nn=(p>=.5).astype(np.uint8),
            logits_nn_2class=np.column_stack([np.zeros(len(flat)),flat]).astype(np.float32))
        cases=[{"id":"baseline","pair_1based":None,"angle_deg":0}]
        for i in range(self.dz):
            for j in range(i+1,self.dz):
                for angle in (15,30,45,60,75,90):
                    name=f"u{i+1}_u{j+1}_{angle:03d}"
                    cases.append({"id":name,"pair_1based":[i+1,j+1],
                                  "angle_deg":angle})
        assert len(cases)==37
        for case in cases:
            angle=case["angle_deg"]
            g=np.eye(self.dz)
            if angle:
                i,j=[v-1 for v in case["pair_1based"]]
                rad=np.deg2rad(angle)
                c,s=np.cos(rad),np.sin(rad)
                g[i,i],g[i,j],g[j,i],g[j,j]=c,-s,s,c
            # W is the ONLY case-specific input used by Step 3.
            u=self.baseline@g.T
            W=np.column_stack([u[self.sid],self.x]).astype(np.float32)
            folder=self.step2/"cases"/case["id"]
            folder.mkdir()
            np.save(folder/"W.npy",W)
        (self.step2/"index.json").write_text(json.dumps({
            "model":"CR3","Z_dim":4,"X_dim":6,"n_subjects":self.n,
            "n_trials":len(self.sid),"complete":True,"cases":cases}),
            encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_profiles_auto_detect_identical_stimulus_bank(self):
        F,info,_=profiles.load_profiles(
            self.step2,self.step2/"cases"/"u1_u2_045",
            channels=("logit_nn","p_nn"))
        self.assertEqual(info["profile_kind"],"shared_stimulus_bank")
        self.assertEqual(F["logit_nn"].shape,(self.n,self.n_trials))
        for ch in ("logit_nn","p_nn"):
            self.assertEqual(profiles.profile_views(F[ch],
                             info["profile_kind"])["contrast"].shape,
                             (self.n,self.n_trials))

    def test_completely_blind_sweep_and_output(self):
        result=mod.analyze(model="CR3",step2=self.step2,output=self.out,
                           channels=("logit_nn","p_nn"),seed=81,
                           train_fraction=.65,max_profile_outputs=5,
                           overwrite=False,plots=False)
        self.assertEqual(result["n_cases"],37)
        with open(self.out/"blind_scores.csv",newline="",encoding="utf-8") as f:
            rows=list(csv.DictReader(f))
        self.assertGreater(len(rows),500)
        self.assertTrue(all("case" not in r for r in rows))
        self.assertTrue(all(r["blind_id"].startswith("B") for r in rows))
        with open(self.out/"blind_key.csv",newline="",encoding="utf-8") as f:
            names=list(csv.DictReader(f))
        self.assertEqual(len(names),37)
        self.assertEqual({r["case"] for r in names},
                         {p.name for p in (self.step2/"cases").iterdir()})
        with open(self.out/"plane_curves.csv",newline="",encoding="utf-8") as f:
            curves=list(csv.DictReader(f))
        self.assertTrue(any(x["angle_deg"]=="0" and
                            x["case"]=="baseline" for x in curves))
        self.assertTrue(any(x["angle_deg"]=="90" for x in curves))
        report=json.loads((self.out/"summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["prohibited_sources_used"],[])
        self.assertEqual(report["n_cases"],37)
        self.assertEqual(report["profile"]["profile_kind"],
                         "shared_stimulus_bank")
        self.assertTrue((self.out/"rankings.csv").exists())
        self.assertTrue((self.out/"metric_summary.csv").exists())
        with self.assertRaises(FileExistsError):
            mod.analyze(model="CR3",step2=self.step2,output=self.out,
                        channels=("logit_nn",),plots=False)

    def test_incomplete_input_fails_closed(self):
        payload=json.loads((self.step2/"index.json").read_text())
        payload["complete"]=False
        (self.step2/"index.json").write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError,"incomplete"):
            mod.analyze(model="CR3",step2=self.step2,output=self.out,
                        channels=("logit_nn",),plots=False)

    def test_reject_cross_case_X_mutation(self):
        path=self.step2/"cases"/"baseline"/"W.npy"
        w=np.load(path)
        w[:,self.dz]+=8.
        np.save(path,w)
        with self.assertRaisesRegex(ValueError,"Stimulus X differs"):
            mod.analyze(model="CR3",step2=self.step2,output=self.out,
                        channels=("logit_nn",),plots=False)


class TestIrregularXProfiles(unittest.TestCase):
    def test_non_shared_stimulus_profiles_are_subject_X_regressions(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/"cases"/"X").mkdir(parents=True)
            rng=np.random.default_rng(58)
            counts=np.array([17,20,19,22,18,19,21,20,19],int)
            sid=np.repeat(np.arange(len(counts)),counts)
            tid=np.concatenate([np.arange(c) for c in counts])
            z=rng.normal(size=(len(counts),4))
            x=rng.uniform(-1,1,size=(len(sid),3))
            Y=2+z[sid,0]*x[:,0]+z[sid,2]*x[:,1]**2
            np.save(root/"subject_index.npy",sid)
            np.save(root/"trial_index.npy",tid)
            np.save(root/"trial_counts.npy",counts)
            np.save(root/"cases"/"X"/"W.npy",np.column_stack([z[sid],x]).astype(np.float32))
            np.savez_compressed(root/"Y.npz",logit_nn=Y)
            (root/"index.json").write_text(json.dumps({"Z_dim":4,"X_dim":3}))
            f,meta,_=profiles.load_profiles(
                root,root/"cases"/"X",channels=("logit_nn",))
            self.assertEqual(meta["profile_kind"],
                             "per_subject_degree2_X_response_coefficients")
            self.assertEqual(f["logit_nn"].shape,(len(counts),10))
            self.assertEqual(profiles.profile_views(f["logit_nn"],
                             meta["profile_kind"])["contrast"].shape,
                             (len(counts),9))


if __name__=="__main__":
    unittest.main()
