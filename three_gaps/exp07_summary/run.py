"""EXP07 - collect EXP06 'final' results into the three-gap pass table.

Fixed criteria (set before looking at the table):
  Gap 1 (analytic) / Gap 2 (formula_sieve): worst-axis angle to the true Tstd axes < 5 deg
  Gap 3 (decoder_sieve): TEST matched |corr| >= oracle - 0.02
  decoder_ad is reported as the diagnostic (network's own derivatives).

python three_gaps/exp07_summary/run.py --tag final --variant AB --loss absL1
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="final")
    ap.add_argument("--variant", default="AB")
    ap.add_argument("--loss", default="absL1")
    a = ap.parse_args()
    files = glob.glob(str(ROOT / "outputs" / "three_gaps" / "exp06" / a.tag / "*" / "results.csv"))
    df = pd.concat([pd.read_csv(f) for f in files])
    df = df[(df.variant == a.variant) & (df.loss == a.loss)].copy()
    gap = {"analytic": "Gap1", "formula_sieve": "Gap2", "decoder_ad": "Gap3-ad(diag)", "decoder_sieve": "Gap3"}
    df["gap"] = df.source.map(gap)

    def verdict(r):
        if r.source in ("analytic", "formula_sieve"):
            return f"{'PASS' if r.angle_to_truth < 5 else 'fail'} {r.angle_to_truth:.1f}deg"
        ok = r.test_mcc >= r.oracle_mcc - 0.02
        return f"{'PASS' if ok else 'fail'} {r.test_mcc:.3f}/{r.oracle_mcc:.3f}"
    df["verdict"] = df.apply(verdict, axis=1)
    tab = df.pivot_table(index=["model", "channel"], columns="gap", values="verdict", aggfunc="first")
    tab = tab[[c for c in ["Gap1", "Gap2", "Gap3-ad(diag)", "Gap3"] if c in tab.columns]]
    out = ROOT / "outputs" / "three_gaps" / "exp07"
    out.mkdir(parents=True, exist_ok=True)
    md = tab.to_markdown()
    (out / f"table_{a.tag}_{a.variant}_{a.loss}.md").write_text(md, encoding="utf-8")
    df.to_csv(out / f"rows_{a.tag}_{a.variant}_{a.loss}.csv", index=False)
    print(md)


if __name__ == "__main__":
    sys.exit(main())
