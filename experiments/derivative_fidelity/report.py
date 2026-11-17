"""Summarize a derivative_fidelity run; never choose winners using real theta."""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def write_csv(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=sorted(set().union(*(set(r) for r in rows)))
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def collect(root):
    root=Path(root)
    rows=[]
    for path in root.rglob("field_comparisons.json"):
        raw=json.loads(path.read_text(encoding="utf-8"))
        for block,record in raw.items():
            rows.append({"folder":str(path.parent.relative_to(root)),
                         "block":block,**record})
    return rows


def make_report(root):
    root=Path(root)
    rows=collect(root)
    write_csv(root/"summary_field_accuracy.csv",rows)
    metrics=[]
    for path in root.rglob("metric_comparisons.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item=json.loads(line)
                item["folder"]=str(path.parent.relative_to(root))
                metrics.append(item)
    write_csv(root/"summary_metric_accuracy.csv",metrics)
    manifest=[]
    for path in root.rglob("run_manifest.json"):
        obj=json.loads(path.read_text(encoding="utf-8"))
        manifest.append({"folder":str(path.parent.relative_to(root)),**obj})
    (root/"all_run_manifests.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    text=["# Derivative fidelity experiment report","",
          f"- Field comparisons: {len(rows)}",
          f"- Metric comparisons: {len(metrics)}",
          f"- Runs: {len(manifest)}","",
          "## Field RMSE by method and block","",
          "| Method | Block | N | Mean RMSE | Mean relative L2 |","|---|---|---:|---:|---:|"]
    groups=defaultdict(list)
    for r in rows:
        if r.get("status")=="ok":
            method=r["folder"].split("/")[-1]
            groups[(method,r["block"])].append(r)
    for (method,block),records in sorted(groups.items()):
        rr=[float(x["relative_l2"]) for x in records if x.get("relative_l2") is not None]
        text.append(f"| {method} | {block} | {len(records)} | {np.mean([float(x['rmse']) for x in records]):.5g} | {np.mean(rr):.5g} |" if rr else
                    f"| {method} | {block} | {len(records)} | {np.mean([float(x['rmse']) for x in records]):.5g} | null |")
    failures=[r for r in manifest if r.get("status") not in ("complete","reference")]
    text+=["","## Incomplete / failed jobs","",f"{len(failures)} run manifests are not complete.", "",
           "Use run_manifest.json to distinguish failed, not_run and not_applicable.",
           "The matched-coordinate analytic proxy is not exact theta(V) ground truth.",
           "Compare raw truth-coordinate results separately.", ""]
    (root/"REPORT.md").write_text("\n".join(text),encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        methods=sorted(set(r["folder"].split("/")[-1] for r in rows if r.get("status")=="ok"))
        blocks=["U","X","UU","UX","XX","UUU","UUX","UXX","XXX"]
        img=np.full((len(blocks),len(methods)),np.nan)
        for bi,b in enumerate(blocks):
            for mi,m in enumerate(methods):
                v=[float(r["rmse"]) for r in rows if r.get("status")=="ok" and
                   r["block"]==b and r["folder"].split("/")[-1]==m]
                if v: img[bi,mi]=np.median(v)
        if len(methods):
            fig,ax=plt.subplots(figsize=(max(8,len(methods)*.8),5))
            show=np.log10(np.maximum(img,1e-12))
            im=ax.imshow(np.ma.masked_invalid(show),aspect="auto")
            ax.set_xticks(range(len(methods)),methods,rotation=50,ha="right")
            ax.set_yticks(range(len(blocks)),blocks)
            ax.set_title("Median field RMSE (log10); pooled across recorded runs")
            fig.colorbar(im,ax=ax,label="log10 RMSE")
            fig.tight_layout(); fig.savefig(root/"summary_rmse.png",dpi=180)
            plt.close(fig)
    except ImportError:
        pass
    return len(rows),len(metrics)
