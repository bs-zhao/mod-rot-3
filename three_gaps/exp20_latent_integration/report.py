"""Summarise EXP20A/B CSVs into markdown tables (MCC, leakage mean/max, angle to Oracle [diagnostic], angle to reference). Only actually-run results.

python report.py > results/summary_tables.md
"""
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent / "results"
pd.set_option("display.width", 250)


def md(df):
    return df.round(4).to_markdown()


print("# EXP20 汇总表（由 report.py 从 CSV 生成）\n")
for f in sorted(R.glob("exp20a_*.csv")):
    d = pd.read_csv(f)
    m, rep, ch = d.model[0], d.rep[0], d.channel[0]
    s = d[d.tag == "sub"].groupby(["N", "scale"]).agg(mcc=("mcc", "mean"), mcc_sd=("mcc", "std"), leak=("leak", "mean"), leak_max=("leak_max", "mean"),
                                                      oracle_angle_max=("oracle_angle_deg", "mean"), err_SX=("err_SX", "mean"), spearman=("landscape_spearman", "mean"),
                                                      angle_to_full=("angle_to_full_deg", "mean"))
    o = d[d.tag != "sub"].set_index(["tag", "N", "scale"])[["mcc", "leak", "leak_max", "oracle_angle_deg", "landscape_spearman", "angle_to_full_deg"]]
    print(f"## EXP20A {m} rep{rep} {ch}\n\n子集（5 个嵌套种子均值）：\n\n{md(s)}\n\n单次：\n\n{md(o)}\n")
for f in sorted(R.glob("exp20b_*.csv")):
    d = pd.read_csv(f)
    m, rep, ch = d.model[0], d.rep[0], d.channel[0]
    p1 = d[d.phase == "P1"].groupby(["integrator", "budget"]).agg(n=("mcc", "size"), mcc=("mcc", "mean"), mcc_sd=("mcc", "std"), leak=("leak", "mean"), leak_max=("leak_max", "mean"),
                                                                  oracle_angle_max=("oracle_angle_deg", "mean"), err_SX=("err_SX", "mean"), err_T4=("err_T4", "mean"),
                                                                  spearman=("spearman_vs_ref", "mean"), angle_to_ref=("angle_to_ref_deg", "mean"))
    p2 = d[d.phase == "P2"].set_index("name")[["mcc", "leak", "leak_max", "oracle_angle_deg", "err_T4", "spearman_vs_ref", "angle_to_ref_deg"]]
    print(f"## EXP20B {m} rep{rep} {ch}\n\nP1 积分器收敛：\n\n{md(p1)}\n\n测度比较（converged RQMC）：\n\n{md(p2)}\n")
