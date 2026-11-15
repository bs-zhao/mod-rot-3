"""Step 0: protect and reproduce the frozen EXP13.

(a) SHA256 of every file in the local archive manifest (baseline/results/archive_manifest.json), and of the live source files;
(b) frozen code unchanged since the tag exp13-frozen-doc05-20261114 (git diff on exp13_pure_simplicity, exp11 harness, common/);
(c) historical 300-subject result reproduced from the frozen cache with the NEW moment-based loss (A0), vs baseline/results/metrics_all.csv.
python verify_frozen.py [--settings all|cr3]
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x7core as X
import losses2 as L2

RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)
TAG = "exp13-frozen-doc05-20261114"


def check_archive():
    m = json.load(open(X.ROOT / "three_gaps" / "post_exp13_20261114" / "baseline" / "results" / "archive_manifest.json"))
    rows = []
    for f in m["files"]:
        a, s = Path(f["archive"]), Path(f["source"])
        ok_a = a.exists() and X.sha256(a) == f["sha256"]
        ok_s = s.exists() and X.sha256(s) == f["sha256"]
        rows.append(dict(archive=str(a), exists=a.exists(), archive_sha_ok=ok_a, source_sha_ok=ok_s, bytes=f["bytes"]))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "step0_archive_check.csv", index=False)
    print(f"archive files {len(df)}: archive sha ok {int(df.archive_sha_ok.sum())}, live source sha ok {int(df.source_sha_ok.sum())}, bytes {df.bytes.sum()/1e9:.2f} GB")
    return df


def check_code():
    out = subprocess.run(["git", "diff", "--stat", TAG, "HEAD", "--", "three_gaps/exp13_pure_simplicity/losses2.py",
                          "three_gaps/exp13_pure_simplicity/combos.py", "three_gaps/exp13_pure_simplicity/cache2.py",
                          "three_gaps/exp11_cr3_search/harness.py", "three_gaps/common"], cwd=X.ROOT, capture_output=True, text=True).stdout.strip()
    print("frozen-code diff vs tag:", out if out else "(none)")
    (RES / "step0_code_diff.txt").write_text(out or "no differences in frozen code versus " + TAG)
    return out == ""


def reproduce(settings):
    hist = pd.read_csv(X.ROOT / "three_gaps" / "post_exp13_20261114" / "baseline" / "results" / "metrics_all.csv")
    rows = []
    for model, rep, ch in settings:
        t0 = time.time()
        st = X.Setting(model, rep)
        c = torch.load(X.ROOT / "outputs" / "three_gaps" / "exp13" / f"cache2_{model}_{st.cond}_{ch}.pt", weights_only=False)
        SX, T4 = L2.moments(c)
        Cref = X.population_cov(st.V[c["ids"]])
        ml = X.MomentLoss(SX.numpy(), T4.numpy(), Cref, st.d)
        fin = ml.search()
        mcc, leak, lmax = X.evaluate(st, fin[0][1])
        h = hist[(hist.model == model) & (hist.repeat == rep) & (hist.channel == ch)].iloc[0]
        rows.append(dict(model=model, rep=rep, channel=ch, mcc=mcc, leak=leak, leak_max=lmax, hist_mcc=h.mcc, hist_leak=h.leakage,
                         hist_leak_max=h.max_leakage, d_mcc=mcc - h.mcc, d_leak=leak - h.leakage, loss=fin[0][0], hist_loss=h.loss, secs=time.time() - t0))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(RES / "step0_reproduction.csv", index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    check_code()
    check_archive()
    sets = [(m, r, "prob") for m in ("CR3", "C1", "C2", "C3") for r in (1, 2)]
    if which == "cr3":
        sets = [s for s in sets if s[0] == "CR3"]
    df = reproduce(sets)
    print("max |d_mcc| =", float(df.d_mcc.abs().max()), " max |d_leak| =", float(df.d_leak.abs().max()))
