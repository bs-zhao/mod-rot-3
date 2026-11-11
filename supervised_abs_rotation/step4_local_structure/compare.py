"""Post-hoc C1/CR3 comparison of already FROZEN Step-4 blind results.

No score computation and no parameter/formula/decoder access. This is a
descriptive, after-the-fact display, NOT a fitting or selection procedure
used inside the blind method.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    with Path(path).open("r",newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key(row):
    return (row["channel"],row["view"],row["k"],row["metric"])


def analyze_results(c1_dir,cr3_dir,output):
    c1_dir=Path(c1_dir);cr3_dir=Path(cr3_dir);out=Path(output)
    meta1=json.loads((c1_dir/"summary.json").read_text(encoding="utf-8"))
    meta2=json.loads((cr3_dir/"summary.json").read_text(encoding="utf-8"))
    if meta1["model"]!="C1" or meta2["model"]!="CR3":
        raise ValueError("Expected genuine C1 and CR3 outputs, in that order")
    a=read_rows(c1_dir/"metric_summary.csv")
    b=read_rows(cr3_dir/"metric_summary.csv")
    aa={key(r):r for r in a}
    bb={key(r):r for r in b}
    if len(aa)!=len(a) or len(bb)!=len(b):
        raise ValueError("Duplicate metric/channel/view/neighborhood identifiers")
    if set(aa)!=set(bb):
        left=sorted(set(aa)-set(bb))
        right=sorted(set(bb)-set(aa))
        raise ValueError(f"Model metric grids differ: C1 only {left[:3]}; CR3 only {right[:3]}")

    rows=[]
    for ky in sorted(aa):
        x,y=aa[ky],bb[ky]
        ctrl=bool(int(x["is_control"]) or int(y["is_control"]))
        c1_plane=int(x["endpoint_favored_planes"])
        cr3_plane=int(y["endpoint_favored_planes"])
        c1_rank=int(x["baseline_rank"])
        cr3_rank=int(y["baseline_rank"])
        rows.append(dict(
            channel=ky[0],view=ky[1],k=ky[2],metric=ky[3],
            is_control=int(ctrl),
            c1_rank=c1_rank,c1_planes=c1_plane,c1_winner=x["winner_case"],
            cr3_rank=cr3_rank,cr3_planes=cr3_plane,cr3_winner=y["winner_case"],
            worst_rank=max(c1_rank,cr3_rank),
            min_favored_planes=min(c1_plane,cr3_plane),
            joint_strict_success=int(
                not ctrl and c1_rank==1 and cr3_rank==1
                and c1_plane>=5 and cr3_plane>=5),
            joint_relaxed_success=int(
                not ctrl and c1_rank<=3 and cr3_rank<=3
                and c1_plane>=4 and cr3_plane>=4),
            c1_median_interior_diff=float(x["median_interior_minus_endpoint"]),
            cr3_median_interior_diff=float(y["median_interior_minus_endpoint"]),
            c1_max_endpoint_gap=float(x["max_endpoint_gap"]),
            cr3_max_endpoint_gap=float(y["max_endpoint_gap"]),
        ))
    out.mkdir(parents=True,exist_ok=True)
    columns=list(rows[0])
    def write(path,records,fieldnames):
        with path.open("w",newline="",encoding="utf-8") as handle:
            w=csv.DictWriter(handle,fieldnames=fieldnames)
            w.writeheader();w.writerows(records)
    write(out/"cross_model_all_metrics.csv",rows,columns)

    eligible=[r for r in rows if not r["is_control"]]
    prioritized=sorted(
        eligible,key=lambda r:(-r["joint_strict_success"],
                               -r["joint_relaxed_success"],
                               -r["min_favored_planes"],
                               r["worst_rank"],
                               r["channel"],r["view"],r["metric"],r["k"]))
    write(out/"cross_model_shortlist.csv",prioritized,columns)
    # The same common metrics in one model: compare their baseline rank
    # and 6-plane recovery under global B vs local B at each k.
    by_model={"C1":aa,"CR3":bb}
    benchmark=[]
    for model,mapping in by_model.items():
        for k in sorted(set(row[2] for row in mapping if row[2]!="global")):
            for ky,x in mapping.items():
                ch,view,setting,metric=ky
                if setting!=k or not metric.startswith("local_"):
                    continue
                global_key=(ch,view,"global","global_"+metric)
                if global_key not in mapping:
                    continue
                baseline=mapping[global_key]
                benchmark.append(dict(
                    model=model,channel=ch,view=view,k=k,metric=metric,
                    global_baseline_rank=int(baseline["baseline_rank"]),
                    local_baseline_rank=int(x["baseline_rank"]),
                    improvement_rank=int(baseline["baseline_rank"])-
                                     int(x["baseline_rank"]),
                    global_favored_planes=int(baseline["endpoint_favored_planes"]),
                    local_favored_planes=int(x["endpoint_favored_planes"]),
                    improvement_planes=int(x["endpoint_favored_planes"])-
                                       int(baseline["endpoint_favored_planes"]),
                ))
    if benchmark:
        write(out/"local_vs_global.csv",benchmark,list(benchmark[0]))
    report={
        "scope":"AFTER unblinding: descriptive C1 vs CR3 metrics only",
        "c1_source":str(c1_dir),
        "cr3_source":str(cr3_dir),
        "n_common_metrics":len(rows),
        "n_noncontrol_metrics":len(eligible),
        "n_joint_strict_success":sum(int(x["joint_strict_success"]) for x in rows),
        "n_joint_relaxed_success":sum(int(x["joint_relaxed_success"]) for x in rows),
        "n_C1_baseline_rank1":sum(r["c1_rank"]==1 for r in eligible),
        "n_CR3_baseline_rank1":sum(r["cr3_rank"]==1 for r in eligible),
        "n_C1_6of6":sum(r["c1_planes"]==6 for r in eligible),
        "n_CR3_6of6":sum(r["cr3_planes"]==6 for r in eligible),
        "caution":"All candidate losses are exploratory and correlated. "
                  "Joint success is not evidence of independent replication; "
                  "Step1 R is supervised and S5 Z uses all observed labels.",
        "top_15_by_declared_joint_criteria":prioritized[:15],
    }
    (out/"comparison.json").write_text(json.dumps(
        report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"Compared {len(rows)} scores in both models: "
          f"strict joint {report['n_joint_strict_success']}, "
          f"relaxed joint {report['n_joint_relaxed_success']}; output={out}")
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c1",type=Path,
                   help="Override default C1 Step4 output folder")
    p.add_argument("--cr3",type=Path,
                   help="Override default CR3 Step4 output folder")
    p.add_argument("--output",type=Path,default=Path("outputs")/"supervised_abs_rotation"/
                   "step4_C1_CR3_comparison")
    args=p.parse_args(argv)
    repo=Path(__file__).resolve().parents[2]
    root=repo/"outputs"/"supervised_abs_rotation"
    common="repeat01/abs_mse/step4_local_structure"
    c1=args.c1 or (root/"C1"/"beta0p0005_lambda0_ctx0p25to0p8_L4"/common)
    cr3=args.cr3 or (root/"CR3"/"beta5em05_lambda0_ctx0p25to0p8_L4"/common)
    analyze_results(c1,cr3,args.output)


if __name__=="__main__":
    main()
