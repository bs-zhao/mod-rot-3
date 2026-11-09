"""Sequencing and provenance guards for one-click Gap1 -> paired Gap2+3."""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_pipeline
from run_gaps import cross_summary


class PipelineTests(unittest.TestCase):
    def test_runs_gap1_then_both_numeric_gaps(self):
        calls=[]
        with tempfile.TemporaryDirectory() as td:
            def gap1(args):
                calls.append(("gap1",args))
            def gaps(args):
                calls.append(("paired_gaps23",args))
            with mock.patch.object(run_pipeline,"valid_gap1",return_value=False), \
                 mock.patch.object(run_pipeline,"run_gap1",side_effect=gap1), \
                 mock.patch.object(run_pipeline,"run_gaps23",side_effect=gaps):
                run_pipeline.main(["--models","C1","--gap1-dir",td,
                                   "--angles","0,45,90","--top-global","3"])
        self.assertEqual([n for n,_ in calls],["gap1","paired_gaps23"])
        self.assertIn("--top-global",calls[1][1])

    def test_reuse_gap1_does_not_recompute(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(run_pipeline,"valid_gap1",return_value=True), \
                 mock.patch.object(run_pipeline,"run_gap1") as a, \
                 mock.patch.object(run_pipeline,"run_gaps23") as b:
                run_pipeline.main(["--models","C2","--gap1-dir",td,"--only-gaps23"])
                a.assert_not_called()
                b.assert_called_once()

    def test_hash_failure_aborts_instead_of_claiming_reuse(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(run_pipeline,"valid_gap1",return_value=False), \
                 mock.patch.object(run_pipeline,"run_gap1") as a, \
                 mock.patch.object(run_pipeline,"run_gaps23") as b:
                with self.assertRaises(FileNotFoundError):
                    run_pipeline.main(["--models","CR3","--gap1-dir",td,"--only-gaps23"])
                a.assert_not_called()
                b.assert_not_called()

    def test_cross_model_ranking_does_not_promote_c1_only(self):
        def row(model,metric,plane,ch,method):
            return dict(model=model,metric=metric,comparison="gap3_vs_analytic",
                        method=method,channel=ch,plane=plane,
                        curve_r=.99 if metric=="C1_only" else .80,
                        normalized_shape_rmse=.01,
                        a_near_true_axis=1,b_near_true_axis=1)
        rows=[]
        for method in ("sieve","rbf"):
            for ch in ("logit","probability"):
                rows.append(row("C1","C1_only","12",ch,method))
                for model in ("C1","C2","C3","CR3"):
                    pairs=("12","13","14","15","16","23","24","25","26",
                           "34","35","36","45","46","56") if model=="C3" else (
                           "12","13","14","23","24","34")
                    for plane in pairs:
                        rows.append(row(model,"all_model",plane,ch,method))
        ranks=cross_summary(rows,["C1","C2","C3","CR3"])
        self.assertEqual(ranks[0]["metric"],"all_model")
        self.assertEqual(ranks[0]["full_coverage"],1)
        partial=next(r for r in ranks if r["metric"]=="C1_only")
        self.assertEqual(partial["models_covered"],1)
        self.assertEqual(partial["full_coverage"],0)
        self.assertLess(partial["condition_coverage_fraction"],1)

    def test_cr1_is_rejected_as_unsupported_actual_generator(self):
        with self.assertRaisesRegex(ValueError,"no separate CR1"):
            run_pipeline.main(["--models","CR1"])


if __name__=="__main__":
    unittest.main()
