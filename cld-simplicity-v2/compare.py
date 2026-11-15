"""Live comparison against the frozen EXP13 with the user's rule: leakage (mean OR max) must drop by >= 0.03 on CR3, on both repeats,
with C1-C3 MCC not lower than EXP13 by more than 0.01.   Writes RESULTS_LIVE.md.
python compare.py PHASE_DIR [PHASE_DIR ...]   (reads <phase>/results/{dev,confirm}_rep*_shard*.csv)
"""
import glob
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "common"))
import cldlib as L

phases = sys.argv[1:] or ["phase1_ctx"]
lines = ["# RESULTS_LIVE — 与冻结 EXP13 对比（自动生成）", "",
         "**判据（用户）**：CR3 的泄漏均值或最大泄漏相对 EXP13 下降 ≥ 0.03，且在两个 NP 重复上都成立，C1–C3 的 MCC 不低于 EXP13 超过 0.01，才算真正提升。",
         "EXP13 参考：CR3 重复 1 = MCC .8173 / 泄漏 .1646 / 最大 .3677；重复 2 = .8122 / .1832 / .3781。", ""]
for ph in phases:
    for stage in ("dev", "confirm"):
        fs = glob.glob(str(HERE / ph / "results" / f"{stage}_rep*_shard*.csv"))
        if not fs:
            continue
        d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        d = d[d.status == "ok"]
        recs = []
        for cand, g in d.groupby("cand"):
            rows = g.to_dict("records")
            v = L.verdict(rows) if stage == "confirm" else {}
            r = dict(cand=cand, n=len(g))
            for rep in (1, 2):
                cr = g[(g.model == "CR3") & (g.rep == rep)]
                if len(cr):
                    r[f"CR3_mcc_r{rep}"], r[f"CR3_leak_r{rep}"], r[f"CR3_lmax_r{rep}"] = cr.mcc.iloc[0], cr.leak.iloc[0], cr.leak_max.iloc[0]
                    b = L.EXP13[("CR3", rep)]
                    r[f"d_mean_r{rep}"], r[f"d_max_r{rep}"] = b[1] - cr.leak.iloc[0], b[2] - cr.leak_max.iloc[0]
                for m in ("C1", "C2", "C3"):
                    cm = g[(g.model == m) & (g.rep == rep)]
                    if len(cm):
                        r[f"{m}_dmcc_r{rep}"] = cm.mcc.iloc[0] - L.EXP13[(m, rep)][0]
            r["dev_pass_r1"] = bool(r.get("d_mean_r1", -1) >= L.THRESH or r.get("d_max_r1", -1) >= L.THRESH)
            r["C_ok_r1"] = all(r.get(f"{m}_dmcc_r1", 0) >= -0.01 for m in ("C1", "C2", "C3"))
            r["REAL"] = v.get("real")
            recs.append(r)
        t = pd.DataFrame(recs)
        key = "d_mean_r1" if "d_mean_r1" in t else "d_mean_r2"
        t = t.sort_values(key, ascending=False)
        t.to_csv(HERE / ph / "results" / f"table_{stage}.csv", index=False)
        n_pass = int(((t.dev_pass_r1) & (t.C_ok_r1)).sum())
        lines += [f"## {ph} — {stage}（{len(t)} 个候选；重复 1 上满足 ≥0.03 且 C1–C3 不退化的：{n_pass}）", ""]
        cols = ["cand", "CR3_mcc_r1", "CR3_leak_r1", "CR3_lmax_r1", "d_mean_r1", "d_max_r1", "C1_dmcc_r1", "C2_dmcc_r1", "C3_dmcc_r1"]
        if stage == "confirm":
            cols += ["CR3_leak_r2", "CR3_lmax_r2", "d_mean_r2", "d_max_r2", "REAL"]
        lines += [t[[c for c in cols if c in t]].head(15).round(4).to_markdown(index=False), ""]
(HERE / "RESULTS_LIVE.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines)[:6000])
