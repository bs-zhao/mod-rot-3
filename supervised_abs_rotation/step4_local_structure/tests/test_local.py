"""Numeric synthetic tests: local B's, equivariance, score symmetry and E2E."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))
from local import (orthogonal_map, build_local_operators, scores,
                   global_fingerprint, select_anchors)
import importlib.util
spec = importlib.util.spec_from_file_location("step4_local_structure_runner", HERE/"run.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
analyze = module.analyze


def givens(dim,pair,angle):
    g=np.eye(dim)
    if pair is None:return g
    i,j=pair
    t=np.deg2rad(angle);c=np.cos(t);s=np.sin(t)
    g[i,i],g[j,j],g[i,j],g[j,i]=c,c,-s,s
    return g


class LocalMathTests(unittest.TestCase):
    def test_exact_local_equivariance_and_90_degree_invariance(self):
        rng=np.random.default_rng(43)
        u=rng.normal(size=(105,4))
        profiles=np.column_stack([
            2*u[:,0]+.5*u[:,1], u[:,2]**2,
            u[:,3]*u[:,0],u[:,1]**2-u[:,3]])
        train=np.arange(70)
        anchors=np.arange(70,90)
        op=build_local_operators(u,train,anchors,neighbors=25,ridge_fraction=.025)
        b,mse=op.apply(profiles)
        self.assertEqual(b.shape,(20,4,4))
        self.assertGreater(mse,0)
        g=givens(4,(0,2),45)
        u_rot=u@g.T
        h,rel=orthogonal_map(u,u_rot)
        self.assertLess(rel,1e-12)
        np.testing.assert_allclose(h,g,atol=1e-12)
        op_rot=build_local_operators(u_rot,train,anchors,neighbors=25,ridge_fraction=.025)
        b_rot,mse_rot=op_rot.apply(profiles)
        np.testing.assert_allclose(b_rot,
                                   np.einsum("ij,ajp->aip",g,b),
                                   rtol=1e-9,atol=1e-9)
        self.assertAlmostEqual(mse,mse_rot,places=12)

        a=scores(b)
        ninety=scores(np.einsum("ij,ajp->aip",givens(4,(0,2),90),b))
        for name in a:
            self.assertAlmostEqual(a[name],ninety[name],delta=1e-8,
                                   msg=f"{name} must be sign/axis permutation invariant")
        self.assertAlmostEqual(
            a["control_local_gradient_fro"],
            scores(np.einsum("ij,ajp->aip",g,b))["control_local_gradient_fro"],
            delta=1e-9)

    def test_local_sparsity_can_distinguish_mixed_axes(self):
        rng=np.random.default_rng(4)
        b=np.zeros((60,4,22))
        for j in range(22):
            b[:,j%4,j]=rng.lognormal(0,.15,size=60)
        g=givens(4,(0,1),45)
        plain=scores(b);rot=scores(np.einsum("ij,ajp->aip",g,b))
        for key in ("local_point_entropy","local_point_tail1",
                    "local_rms_entropy","local_rms_tail1",
                    "local_meanabs_entropy","nonlinear_local_point_entropy"):
            self.assertLess(plain[key],rot[key],key)
        self.assertAlmostEqual(plain["control_local_gradient_fro"],
                               rot["control_local_gradient_fro"],places=8)

    def test_constant_field_has_defined_flat_nonlinearity(self):
        rng=np.random.default_rng(6)
        b=np.broadcast_to(rng.normal(size=(1,4,8)),(30,4,8)).copy()
        vals=scores(b)
        self.assertEqual(vals["nonlinear_local_point_entropy"],0.)
        self.assertEqual(vals["nonlinear_local_rms_entropy"],0.)
        self.assertAlmostEqual(vals["control_local_variation_fraction"],0.)

    def test_nonorthogonal_geometry_rejected(self):
        rng=np.random.default_rng(9)
        u=rng.normal(size=(50,4))
        scaled=u@np.diag([1.,1.2,.8,1.])
        with self.assertRaisesRegex(ValueError,"not an orthogonal transform"):
            orthogonal_map(u,scaled)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.src=self.root/"step2_pairwise_sweep"
        (self.src/"cases").mkdir(parents=True)
        self.out=self.root/"step4_local_structure"
        rng=np.random.default_rng(189)
        self.n=90;self.d=4;self.dx=6;self.t=9
        ids=np.repeat(np.arange(self.n),self.t)
        ti=np.tile(np.arange(self.t),self.n)
        counts=np.full(self.n,self.t,int)
        bank=rng.uniform(0,1,size=(self.t,self.dx)).astype("float32")
        x=np.tile(bank,(self.n,1))
        raw=rng.normal(size=(self.n,self.d))
        latent=(raw-raw.mean(0))/raw.std(0)
        r,_=np.linalg.qr(rng.normal(size=(self.d,self.d)))
        self.baseline=latent@r.T
        b=np.zeros((self.d,self.t))
        b[0,:3]=[.7,-1.,.4]
        b[1,3:5]=[-.8,.9]
        b[2,5:7]=[.6,-.4]
        b[3,7:9]=[.3,-1.1]
        logits=self.baseline@b+.02*rng.normal(size=(self.n,self.t))
        flat=logits.reshape(-1)
        prob=1/(1+np.exp(-flat))
        np.save(self.src/"subject_index.npy",ids)
        np.save(self.src/"trial_index.npy",ti)
        np.save(self.src/"trial_counts.npy",counts)
        np.savez_compressed(self.src/"Y.npz",logit_nn=flat.astype("float32"),
                            p_nn=prob.astype("float32"),
                            choice_nn=(prob>.5).astype("uint8"),
                            choice_observed=rng.binomial(1,prob).astype("uint8"))
        catalog=[dict(id="baseline",pair_1based=None,angle_deg=0)]
        for i in range(4):
            for j in range(i+1,4):
                for deg in (15,30,45,60,75,90):
                    catalog.append(dict(id=f"u{i+1}_u{j+1}_{deg:03}",
                                        pair_1based=[i+1,j+1],angle_deg=deg))
        assert len(catalog)==37
        for c in catalog:
            pair=([v-1 for v in c["pair_1based"]]
                  if c["pair_1based"] is not None else None)
            g=givens(4,pair,c["angle_deg"])
            W=np.column_stack([(self.baseline@g.T)[ids],x]).astype("float32")
            dir=self.src/"cases"/c["id"]
            dir.mkdir()
            np.save(dir/"W.npy",W)
        (self.src/"index.json").write_text(json.dumps(dict(
            model="CR3",n_subjects=self.n,n_trials=len(ids),Z_dim=4,
            X_dim=self.dx,complete=True,cases=catalog)),encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_all_cases_blind_scores_comparison(self):
        result=analyze(step2=self.src,output=self.out,model="CR3",
                       channels=("logit_nn","p_nn"),k_grid=(12,25),
                       anchors=12,seed=199,train_fraction=.7,plots=False)
        self.assertEqual(result["n_cases"],37)
        with (self.out/"blind_scores.csv").open(newline="",encoding="utf-8") as f:
            blind=list(csv.DictReader(f))
        self.assertGreater(len(blind),1500)
        self.assertTrue(all("case" not in row for row in blind))
        self.assertTrue(all(row["blind_id"].startswith("B") for row in blind))
        with (self.out/"blind_key.csv").open(newline="",encoding="utf-8") as f:
            key=list(csv.DictReader(f))
        self.assertEqual(len(key),37)
        self.assertEqual(set(row["case"] for row in key),set(
            p.name for p in (self.src/"cases").iterdir()))
        with (self.out/"metric_summary.csv").open(newline="",encoding="utf-8") as f:
            stats=list(csv.DictReader(f))
        self.assertTrue(any(r["metric"]=="local_rms_entropy"
                            for r in stats))
        self.assertTrue(any(r["metric"]=="nonlinear_local_point_entropy"
                            for r in stats))
        self.assertTrue(any(r["metric"]=="global_local_point_entropy"
                            for r in stats))
        with (self.out/"local_fit_quality.csv").open(newline="",encoding="utf-8") as f:
            quality=list(csv.DictReader(f))
        self.assertEqual(len(quality),8)
        record=json.loads((self.out/"summary.json").read_text())
        self.assertEqual(record["disallowed_inputs_used"],[])
        self.assertEqual(record["neighborhood_sizes"],[12,25])
        self.assertEqual(record["n_anchor_subjects"],12)
        with self.assertRaises(FileExistsError):
            analyze(step2=self.src,output=self.out,model="CR3",
                    channels=("logit_nn",),k_grid=(12,),anchors=9,
                    plots=False)

    def test_c1_variable_X_uses_generic_subject_response_profiles(self):
        rng=np.random.default_rng(821)
        # C1's subjects have different stimulus arrays, so the response
        # profile is a generic per-subject quadratic X regression, not a
        # simulator equation or a 500-stimulus shared-bank lookup.
        x=rng.uniform(-1,1,size=(self.n*self.t,3)).astype("float32")
        for case_dir in (self.src/"cases").iterdir():
            path=case_dir/"W.npy"
            existing=np.load(path)
            np.save(path,np.column_stack([existing[:,:self.d],x]).astype("float32"))
        idx=json.loads((self.src/"index.json").read_text())
        idx["model"]="C1"
        idx["X_dim"]=3
        (self.src/"index.json").write_text(json.dumps(idx))
        result=analyze(step2=self.src,output=self.out,model="C1",
                       channels=("logit_nn",),k_grid=(12,25),anchors=10,
                       train_fraction=.7,plots=False)
        self.assertEqual(result["n_cases"],37)
        report=json.loads((self.out/"summary.json").read_text())
        self.assertEqual(report["subject_profile"]["profile_kind"],
                         "per_subject_degree2_X_response_coefficients")
        self.assertEqual(report["subject_profile"]["n_profile_features"],10)
        self.assertTrue((self.out/"metric_summary.csv").is_file())

    def test_incomplete_step2_rejected(self):
        index=json.loads((self.src/"index.json").read_text())
        index["complete"]=False
        (self.src/"index.json").write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError,"Step 2 must be complete"):
            analyze(step2=self.src,output=self.out,model="CR3",
                    channels=("logit_nn",),k_grid=(12,),anchors=9,
                    plots=False)

    def test_changed_X_rejected(self):
        path=self.src/"cases"/"baseline"/"W.npy"
        data=np.load(path)
        data[:,self.d]+=13
        np.save(path,data)
        with self.assertRaisesRegex(ValueError,"X columns changed"):
            analyze(step2=self.src,output=self.out,model="CR3",
                    channels=("logit_nn",),k_grid=(12,),anchors=9,
                    plots=False)


if __name__=="__main__":
    unittest.main()
