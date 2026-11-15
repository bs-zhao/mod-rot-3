"""EXP24 analysis of the screening: development-set ranking (NP repeat 1, four models).

dev_score = mean over models of (leak - oracle_leak) / (identity_leak - oracle_leak)   (0 = Oracle-level, 1 = no better than no rotation)
Also reported: mean MCC gap, CR3 leakage, worst-model score, per-model numbers.  Ranking never uses repeat 2.
python analyze.py [REP=1]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
rep = int(sys.argv[1]) if len(sys.argv) > 1 else 1
d = pd.concat([pd.read_csv(f) for f in sorted(RES.glob(f"screen_rep{rep}_shard*of8.csv"))], ignore_index=True)
d["cand"] = np.where(d.kind == "combo", "UX:" + d.ux.fillna("") + " + UU:" + d.uu.fillna(""),
                     np.where(d.kind == "single_UX", "UX:" + d.ux.fillna("") + " only", np.where(d.kind == "single_UU", "UU:" + d.uu.fillna("") + " only", "JOINT:" + d.ux.fillna(""))))
ok = d[d.status == "ok"].copy()
ok["leak_pos"] = (ok.leak - ok.oracle_leak) / (ok.identity_leak - ok.oracle_leak)
ok["mcc_pos"] = (ok.oracle_mcc - ok.mcc) / (ok.oracle_mcc - ok.identity_mcc)
g = ok.groupby("cand")
agg = pd.DataFrame(dict(n_models=g.model.nunique(), dev_score=g.leak_pos.mean(), worst_model=g.leak_pos.max(), mcc_gap=g.mcc_pos.mean(), kind=g.kind.first()))
for m in ("CR3", "C1", "C2", "C3"):
    s = ok[ok.model == m].set_index("cand")
    agg[f"{m}_mcc"] = s.mcc
    agg[f"{m}_leak"] = s.leak
    agg[f"{m}_lmax"] = s.leak_max
agg = agg[agg.n_models == 4].sort_values("dev_score")
agg.to_csv(RES / f"ranking_rep{rep}.csv")
base = agg.loc["UX:neff + UU:diagshare"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 100)
cols = ["dev_score", "worst_model", "mcc_gap", "CR3_mcc", "CR3_leak", "CR3_lmax", "C1_mcc", "C2_mcc", "C3_mcc"]
print(f"candidates with all 4 models: {len(agg)} of {d.groupby('cand').ngroups}; no_signal/nan rows: {(d.status != 'ok').sum()}")
print("\n== frozen EXP13 (UX:neff + UU:diagshare)\n", base[cols].astype(float).round(4).to_string())
print("\n== top 25 by dev_score\n", agg[cols].head(25).round(4).to_string())
for kind in ("single_UX", "single_UU", "joint"):
    print(f"\n== best {kind}\n", agg[agg.kind == kind][cols].head(8).round(4).to_string())
print("\n== rank of the frozen baseline:", int((agg.dev_score < base.dev_score).sum()) + 1, "of", len(agg))
# marginal effects: mean dev_score by UX measure and by UU measure over the combo grid
cb = ok[ok.kind == "combo"].groupby(["ux", "uu"]).leak_pos.mean().unstack()
cb.to_csv(RES / f"combo_grid_devscore_rep{rep}.csv")
print("\n== mean dev_score by UX measure (over all UU partners), best 10\n", cb.mean(1).sort_values().head(10).round(3).to_string())
print("\n== mean dev_score by UU measure (over all UX partners), best 10\n", cb.mean(0).sort_values().head(10).round(3).to_string())
