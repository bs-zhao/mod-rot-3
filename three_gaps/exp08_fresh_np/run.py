"""EXP08 - confirmation of the frozen EXP06/EXP07 method on a freshly trained NP.

Nothing in the method is changed: same view (GCat; = GP for C models), same
stage-A thresholds, same stage-B loss (absL1), same seeds/starts/steps.  The
only change is the NP checkpoint (np/s3_train.py --repeat 2: a new training
seed, everything else identical) and therefore the frozen Z and decoder.

Gap 1 / Gap 2 do not depend on the NP (they use the true Tstd coordinates), so
only the decoder sources are rerun.  Pre-registered pass rule (same as EXP07):
  Gap 3: TEST matched |corr| >= oracle(rep) - 0.02

Steps (Windows CMD, after training + s5 extraction of repeat 2):
  python three_gaps/exp00_oracle/run.py    --model C1 --beta 0.0005 --zdim 4 --pca 0 --repeat 2
  python three_gaps/exp06_two_stage/run.py --model C1 --beta 0.0005 --zdim 4 --pca 0 --repeat 2 ^
         --view GCat --losses absL1,rmsL1 --sources decoder_ad,decoder_sieve --tag confirm
  ... (C2, C3 zdim 6, CR3 beta 0.00005)
  python three_gaps/exp08_fresh_np/run.py
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = {"decoder_ad": "Gap3-ad(diag)", "decoder_sieve": "Gap3"}


def collect(tag, rep, variant, loss):
    files = glob.glob(str(ROOT / "outputs" / "three_gaps" / "exp06" / tag / "*" / "results.csv"))
    if not files:
        raise FileNotFoundError(f"no exp06 results for tag {tag}")
    df = pd.concat([pd.read_csv(f) for f in files])
    df = df[(df.variant == variant) & (df.loss == loss) & df.source.isin(SRC)].copy()
    df["rep"] = rep
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag_ref", default="final")
    ap.add_argument("--tag_new", default="confirm")
    ap.add_argument("--variant", default="AB")
    ap.add_argument("--loss", default="absL1")
    a = ap.parse_args()
    df = pd.concat([collect(a.tag_ref, 1, a.variant, a.loss), collect(a.tag_new, 2, a.variant, a.loss)])
    df["pass"] = df.test_mcc >= df.oracle_mcc - 0.02
    df["cell"] = df.apply(lambda r: f"{'PASS' if r['pass'] else 'fail'} {r.test_mcc:.3f}/{r.oracle_mcc:.3f}", axis=1)
    df["col"] = df.source.map(SRC) + " rep" + df.rep.astype(str)
    tab = df.pivot_table(index=["model", "channel"], columns="col", values="cell", aggfunc="first")
    order = [f"{g} rep{r}" for g in ("Gap3", "Gap3-ad(diag)") for r in (1, 2)]
    tab = tab[[c for c in order if c in tab.columns]]
    ref = df.drop_duplicates(["model", "rep"]).pivot_table(index="model", columns="rep",
                                                           values=["oracle_mcc", "identity_mcc"])
    ang = df[df.source == "decoder_sieve"].pivot_table(index=["model", "channel"], columns="rep",
                                                       values=["angle_to_oracle", "stageA"], aggfunc="first")
    out = ROOT / "outputs" / "three_gaps" / "exp08"
    out.mkdir(parents=True, exist_ok=True)
    md = ("## Gap 3 pass table (test_mcc / oracle_mcc)\n\n" + tab.to_markdown() +
          "\n\n## Oracle / identity TEST MCC per repeat\n\n" + ref.round(3).to_markdown() +
          "\n\n## decoder_sieve: stage-A axes and angle to the oracle rotation\n\n" + ang.to_markdown(floatfmt=".1f") + "\n")
    (out / f"table_{a.variant}_{a.loss}.md").write_text(md, encoding="utf-8")
    df.to_csv(out / f"rows_{a.variant}_{a.loss}.csv", index=False)
    print(md)


if __name__ == "__main__":
    sys.exit(main())
