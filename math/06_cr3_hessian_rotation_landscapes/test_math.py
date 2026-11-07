"""Unit and integration tests for 06 independent SO(4) Hessian landscapes.

Run at repo root:
 python -m unittest discover -s math/06_cr3_hessian_rotation_landscapes -p "test_*.py" -v
"""
from __future__ import annotations
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np

from rotation import (fit_affine,pair_rotation,candidate_mapping,
                      original_z_to_candidate_u,star_to_rotated,rotated_to_star,
                      parse_planes,parse_angles,PLANES)
from fast_tensor import RotationTensor
from run import run_nine,main
from metrics import structure,equivariance_error,metrics_for_source
from loader import load_source,CHANNELS
from plots import plot_dashboard
from sparsity import vector_losses,rich_sparsity,metric_catalog
from screen import select_sparsity_metrics
from plots import plot_xz_sparsity


def fixture(seed=41,n=75,m=35):
    rng=np.random.default_rng(seed)
    Z=rng.normal(size=(n,4))
    X=rng.uniform(.1,.9,size=(m,6))
    A=np.array([[.12,.24,-.03,.01],
                [.04,-.22,.30,.01],
                [.22,.05,.03,.18],
                [.05,.07,-.11,-.29]])
    b=np.array([.8,2.,.7,2.2])
    T=Z@A.T+b
    # A quadratic function in the rotatable U and the fixed X.
    y=(T[:,0,None]*X[None,:,2]+
       .4*T[:,1,None]**2*X[None,:,5]+
       T[:,3,None]*X[None,:,0]**2)
    Y=np.stack([(k+1)*.1*y for k in range(6)],axis=-1)
    return Z,T,X,Y,A,b


class GeometryTests(unittest.TestCase):
    def test_fit_regression_and_rotation(self):
        Z,T,X,Y,A,b=fixture()
        Ah,bh=fit_affine(Z,T)
        np.testing.assert_allclose(Ah,A,atol=1e-12)
        np.testing.assert_allclose(bh,b,atol=1e-12)
        self.assertEqual(len(PLANES),6)
        for i,j in PLANES:
            R=pair_rotation(i,j,90)
            np.testing.assert_allclose(R.T@R,np.eye(4),atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(R),1.,places=12)
            Ac,R,U=candidate_mapping(Ah,bh,Z,i,j,35.)
            np.testing.assert_allclose(Ac,R@Ah,atol=1e-14)
            np.testing.assert_allclose(U-b,(T-b)@R.T,atol=1e-12)

    def test_analytic_chain_rule_and_invariance(self):
        Z,T,X,Y,A,b=fixture()
        rng=np.random.default_rng(10)
        H=rng.normal(size=(10,10))
        H=(H+H.T)/2
        Hstar=original_z_to_candidate_u(H,A)
        for i,j in PLANES:
            for angle in (0.,13.5,45.,90.):
                Anew,R,_=candidate_mapping(A,b,Z,i,j,angle)
                test=original_z_to_candidate_u(H,Anew)
                exact=star_to_rotated(Hstar,R)
                np.testing.assert_allclose(test,exact,atol=1e-11)
                np.testing.assert_allclose(rotated_to_star(test,R),Hstar,atol=1e-11)
                for block in ("ZZ","ZX","XX","FULL"):
                    self.assertAlmostEqual(structure(test,block)["frob_norm"],
                                           structure(Hstar,block)["frob_norm"],places=9)
                self.assertLess(equivariance_error(test,Hstar,R),1e-10)

    def test_angle_parsing(self):
        np.testing.assert_allclose(parse_angles("0:90:5"),np.arange(0,91,5))
        np.testing.assert_allclose(parse_angles("0,30,90"),[0,30,90])
        self.assertEqual(parse_planes("all"),PLANES)
        self.assertEqual(parse_planes("alpha:lambda,02"),((0,1),(0,2)))
        with self.assertRaises(ValueError):parse_angles("5:90:5")
        with self.assertRaises(ValueError):parse_planes("3:1")


class DataOnlyTests(unittest.TestCase):
    def test_six_y_regression_hessian_at_two_rotations(self):
        Z,T,X,Y,A,b=fixture()
        model=RotationTensor(X,Y,np.arange(55),degree_u=2,degree_x=2,
                             ridge_u=1e-9,ridge_x=1e-9,
                             heldout_ids=np.arange(55,75),validation_limit=5)
        M0,mask,rmse=model.fit_at(A,Z,b)
        self.assertEqual(M0.shape,(6,10,10))
        np.testing.assert_allclose(M0,M0.transpose(0,2,1),atol=1e-13)
        self.assertEqual(rmse.shape,(6,))
        self.assertTrue(np.isfinite(M0).all())
        self.assertTrue(np.isfinite(rmse).all())
        # Analytic f = .1*(U0 X2+.4 U1^2 X5+U3 X0^2)
        self.assertAlmostEqual(M0[0,0,6],.1,delta=.005)
        self.assertAlmostEqual(M0[0,1,1],.08*np.mean(X[:,5]),delta=.005)
        self.assertAlmostEqual(M0[0,9,9],0.,delta=.005)
        Anew,R,_=candidate_mapping(A,b,Z,0,2,90.)
        M1,_,_=model.fit_at(Anew,Z,b)
        self.assertTrue(np.isfinite(M1).all())
        self.assertEqual(M1.shape,M0.shape)

    def test_all_nine_source_response_combinations(self):
        Z,T,X,Y,A,b=fixture()
        h=np.eye(10)
        oracle={"logit":{"mean_ZX":h,"fraction_defined":np.ones_like(h)},
                "probability":{"mean_ZX":.5*h,"fraction_defined":np.ones_like(h)}}
        model=RotationTensor(X,Y,np.arange(55),degree_u=2,degree_x=2,
                             ridge_u=1e-6,ridge_x=1e-6,
                             heldout_ids=np.arange(55,75),validation_limit=3)
        nine,reference,mask,err=run_nine(A,Z,X,Y,oracle,model,b)
        self.assertEqual(len(nine),9)
        np.testing.assert_allclose(nine[("analytic","choice")],
                                   nine[("analytic","probability")])
        self.assertTrue(np.isfinite(nine[("nn_y","logit")]).all())
        self.assertTrue(np.isfinite(nine[("formula_y","choice")]).all())
        blocks=metrics_for_source(nine[("nn_y","logit")],reference["logit"],mask)
        self.assertEqual(blocks["ZZ_diag"]["n_elements"],4)
        self.assertEqual(blocks["ZZ_offdiag"]["n_elements"],6)
        self.assertEqual(blocks["ZX"]["n_elements"],24)
        self.assertEqual(blocks["XX_offdiag"]["n_elements"],15)


class PlotTests(unittest.TestCase):
    def test_nine_method_dashboard_png(self):
        curves=[]
        for angle in (0.,45.,90.):
            for method in ("analytic","formula_y","nn_y"):
                for target in ("choice","logit","probability"):
                    for block in ("ZZ","ZX","XX"):
                        curves.append(dict(plane="01_alpha_lambda",angle_degrees=angle,
                                           source=method,target=target,block=block,
                                           normalized_l1=.5+.001*angle))
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"dashboard.png"
            plot_dashboard(curves,"01_alpha_lambda",
                           np.array([0.,45.,90.]),"normalized_l1",output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size,1000)


class RichSparsityTests(unittest.TestCase):
    def test_generic_one_and_two_dominant_effect_cases(self):
        # Examples reflect a general property: any small residual
        # should be penalized whether 1, 2 or more major effects remain.
        for large_a,small_a,large_b,small_b,k in [
            ([2.],[.011,.001],[2.],[.001,.001],1),
            ([2.,2.],[.01,.001],[2.,2.],[.001,.001],2),
            ([3.,1.,2.],[.02,.003],[3.,1.,2.],[.002,.003],3),
        ]:
            a=np.array(large_a+small_a,dtype=float)
            b=np.array(large_b+small_b,dtype=float)
            one=vector_losses(a)
            two=vector_losses(b)
            self.assertGreater(one[f"tail2sq_k{k}_loss"],
                               two[f"tail2sq_k{k}_loss"])
            self.assertGreater(one["softl0_t0p005_loss"],
                               two["softl0_t0p005_loss"])
            self.assertGreater(one["logsum_t0p005_loss"],
                               two["logsum_t0p005_loss"])
            self.assertGreater(one["capped_l2_t0p005_loss"],
                               two["capped_l2_t0p005_loss"])
            self.assertGreater(one["penalized_support_t0p005_loss"],
                               two["penalized_support_t0p005_loss"])

    def test_scale_sign_and_permutation_invariance(self):
        rng=np.random.default_rng(5)
        a=np.array([2.,2.,.012,.001,0.,.00005])
        baseline=vector_losses(a)
        for factor in (1e-8,1e7):
            perturbed=factor*a[rng.permutation(len(a))]
            perturbed[0]*=-1
            current=vector_losses(perturbed)
            for key in baseline:
                np.testing.assert_allclose(current[key],baseline[key],
                                           rtol=1e-9,atol=1e-10,
                                           err_msg=key)

    def test_rowwise_sparsity_and_unit_scaling(self):
        H=np.zeros((4,6))
        H[0,:4]=[2,2,.01,.001]
        H[1,:4]=[2,2,.001,.001]
        raw=rich_sparsity(H,"ZX",x_std=np.array([.2,1.5,.5,3.,2.,1.]))
        self.assertIn("zx_row_mean_tail2sq_k2_loss",raw)
        self.assertIn("zx_row_max_tail2sq_k2_loss",raw)
        self.assertIn("zx_xstd_row_mean_logsum_t0p005_loss",raw)
        self.assertIn("xstd_flat_capped_l2_t0p005_loss",raw)
        self.assertGreater(raw["zx_row_max_tail2sq_k2_loss"],
                           raw["zx_row_mean_tail2sq_k2_loss"])
        self.assertNotAlmostEqual(raw["zx_row_mean_logsum_t0p005_loss"],
                                  raw["zx_xstd_row_mean_logsum_t0p005_loss"])
        # Row reorder/sign is an exact 90-degree coordinate exchange.
        R=np.eye(4)
        R[:2,:2]=[[0,-1],[1,0]]
        changed=rich_sparsity(R@H,"ZX",x_std=np.array([.2,1.5,.5,3.,2.,1.]))
        for key,val in raw.items():
            np.testing.assert_allclose(changed[key],val,atol=1e-10,err_msg=key)

    def test_all_hessian_blocks_get_sparsity_metrics(self):
        H=np.eye(10)
        H[0,6]=H[6,0]=.04
        oracle=H.copy()
        mask=np.ones((10,10))
        result=metrics_for_source(H,oracle,mask,x_std=np.ones(6))
        for block in ("ZZ","ZX","XX","FULL","ZZ_diag","ZZ_offdiag","XX_diag","XX_offdiag"):
            self.assertIn("flat_lp025_loss",result[block])
            self.assertIn("flat_tail1_k1_loss",result[block])
        self.assertGreater(len(metric_catalog()),35)

    def test_minimum_is_essential_not_just_an_extremum(self):
        angles=[0.,15.,45.,75.,90.]
        good=[1.0,2.,3.,2.2,1.]
        rows=[]
        for source,off in (("analytic",0.),("formula_y",.05),("nn_y",.02)):
            for ang,v in zip(angles,good):
                rows.append(dict(plane="01_alpha_lambda",target="logit",
                                 block="ZX",source=source,angle_degrees=ang,
                                 zx_row_mean_capped_l2_t0p005_loss=v+off))
        a,b,r=select_sparsity_metrics(rows,min_correlation=.9)
        self.assertEqual(len(a),1)
        self.assertTrue(a[0]["qualifies"])
        self.assertEqual(r[0]["max_planes_in_one_group"],1)
        for row in rows:
            row["zx_row_mean_capped_l2_t0p005_loss"]*= -1
        neg,_,_=select_sparsity_metrics(rows,min_correlation=.9)
        self.assertFalse(neg[0]["qualifies"])  # 0/90 becomes a MAXIMUM

    def test_plot_xz_sparsity_renders(self):
        rows=[]
        for target in ("choice","logit","probability"):
            for source in ("analytic","formula_y","nn_y"):
                for angle in (0.,45.,90.):
                    rows.append(dict(plane="01_alpha_lambda",target=target,
                                     block="ZX",source=source,angle_degrees=angle,
                                     zx_row_mean_capped_l2_t0p005_loss=angle/90+1))
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/"xz.png"
            plot_xz_sparsity(rows,"01_alpha_lambda",[0.,45.,90.],
                             "zx_row_mean_capped_l2_t0p005_loss",dest)
            self.assertTrue(dest.is_file())
            self.assertGreater(dest.stat().st_size,500)


class EndToEndTests(unittest.TestCase):
    def test_step04_directory_and_zip_end_to_end(self):
        Z,T,X,Y,A,b=fixture()
        H=np.eye(10)*.1
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/"repeat01"
            d.mkdir()
            np.savez_compressed(d/"all_Y_and_mapping.npz",
                Z_all_trial=Z,T_true=T,X_fixed=X,A=A,b=b,
                T_hat_AZ_plus_b=T,**dict(zip(CHANNELS,[Y[:,:,j] for j in range(6)])))
            np.savez_compressed(d/"all_matrices.npz",
                map_fit_ids=np.arange(len(Z)),sieve_train_ids=np.arange(55),
                sieve_test_ids=np.arange(55,len(Z)),
                analytic_logit__conditional_mean=H,
                analytic_logit__fraction_defined=np.ones((10,10)),
                analytic_probability__conditional_mean=.3*H,
                analytic_probability__fraction_defined=np.ones((10,10)))
            (d/"summary.json").write_text(json.dumps({"no_nn_hessian":True}),
                                          encoding="utf-8")
            loaded=load_source(d)
            self.assertEqual(loaded["Y"].shape,(len(Z),len(X),6))
            arc=Path(tmp)/"source.zip"
            with zipfile.ZipFile(arc,"w") as z:
                for p in d.iterdir():z.write(p,arcname="repeat01/"+p.name)
            np.testing.assert_allclose(load_source(arc)["A_star"],A,atol=1e-12)
            out=Path(tmp)/"result"
            result=main(["--source",str(arc),"--output",str(out),
                         "--planes","01","--angles","0,90",
                         "--degree-u","2","--degree-x","2",
                         "--ridge-u","1e-7","--ridge-x","1e-7",
                         "--validation-limit","3","--no-plots"])
            self.assertEqual(result["total_hessians"],18)
            self.assertTrue((out/"all_curves.csv").is_file())
            self.assertTrue((out/"best_angles.csv").is_file())
            self.assertTrue((out/"hessians__01_alpha_lambda.npz").is_file())
            with np.load(out/"hessians__01_alpha_lambda.npz") as f:
                self.assertEqual(f["H"].shape,(2,9,10,10))
                self.assertEqual(f["A_candidates"].shape,(2,4,4))


if __name__=="__main__":
    unittest.main()
