"""Fixture-based check that Step4 C1/CR3 comparison is post-hoc and audited."""
from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[1]/"compare.py"
spec=importlib.util.spec_from_file_location("step4_local_compare",SCRIPT)
compare=importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class TestCompare(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.a=self.root/"C1"
        self.b=self.root/"CR3"
        self.a.mkdir();self.b.mkdir()
        for folder,model in ((self.a,"C1"),(self.b,"CR3")):
            (folder/"summary.json").write_text(json.dumps({"model":model}))
            rows=[
                {
                    "channel":"logit_nn","view":"contrast",
                    "k":"96","metric":"local_point_entropy","is_control":"0",
                    "baseline_rank": "1" if model=="C1" else "2",
                    "endpoint_favored_planes": "6" if model=="C1" else "5",
                    "winner_case":"baseline" if model=="C1" else "u1_u2_045",
                    "median_interior_minus_endpoint": ".03",
                    "max_endpoint_gap":"0",
                },
                {
                    "channel":"logit_nn","view":"contrast",
                    "k":"global","metric":"global_local_point_entropy",
                    "is_control":"0",
                    "baseline_rank":"1" if model=="C1" else "27",
                    "endpoint_favored_planes":"6" if model=="C1" else "1",
                    "winner_case":"baseline" if model=="C1" else "u1_u2_045",
                    "median_interior_minus_endpoint": ".02",
                    "max_endpoint_gap":"0",
                },
                {
                    "channel":"logit_nn","view":"contrast",
                    "k":"96","metric":"control_local_gradient_fro",
                    "is_control":"1",
                    "baseline_rank":"1",
                    "endpoint_favored_planes":"0",
                    "winner_case":"baseline",
                    "median_interior_minus_endpoint":"0",
                    "max_endpoint_gap":"0",
                },
            ]
            with (folder/"metric_summary.csv").open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]))
                w.writeheader();w.writerows(rows)
    def tearDown(self):
        self.tmp.cleanup()

    def test_posthoc_joint_and_global_comparison(self):
        output=self.root/"compare"
        report=compare.analyze_results(self.a,self.b,output)
        self.assertEqual(report["n_common_metrics"],3)
        self.assertEqual(report["n_noncontrol_metrics"],2)
        self.assertEqual(report["n_joint_strict_success"],0)
        self.assertEqual(report["n_joint_relaxed_success"],1)
        with (output/"local_vs_global.csv").open(newline="") as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),2)
        r=next(row for row in rows if row["model"]=="CR3")
        self.assertEqual(int(r["global_baseline_rank"]),27)
        self.assertEqual(int(r["local_baseline_rank"]),2)
        self.assertEqual(int(r["improvement_planes"]),4)
        self.assertTrue((output/"cross_model_shortlist.csv").exists())

    def test_reject_mismatched_channel_or_model(self):
        p=self.b/"summary.json"
        p.write_text(json.dumps({"model":"C1"}))
        with self.assertRaisesRegex(ValueError,"genuine C1 and CR3"):
            compare.analyze_results(self.a,self.b,self.root/"other")


if __name__=="__main__":
    unittest.main()
