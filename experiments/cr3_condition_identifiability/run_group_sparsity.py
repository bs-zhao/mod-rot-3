"""Run condition-wise UNIVERSAL dimension-sparsity and weighted rotation scans.

Reads *existing* analytic CR3 moments.npz; does not retrain any NN, recompute
derivatives, or modify older experiments.

  python -m experiments.cr3_condition_identifiability.run_group_sparsity \\
      --source-tag analytic_cr3_64 --tag universal_group_v1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .design import ROOT, NAMES
from .run import load_moments
from .objectives import alignment, candidate_rotations, matrix_from_params
from .dimension_sparsity import (
    COMPONENTS, METRICS, AGGREGATIONS, evaluate, prepare, mixing_weights,
)

BASE = ROOT / "outputs" / "cr3_condition_identifiability"
DEFAULT_RECIPES = list(COMPONENTS)
DEFAULT_METRICS = list(METRICS)


def json_dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def checksum(path):
    h = hashlib.sha256()
    with open(path,"rb") as src:
        for chunk in iter(lambda: src.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def make_config(recipe, metric, balance, aggregation):
    return (recipe, metric, float(balance), aggregation)


def config_key(config):
    recipe, metric, balance, aggregation = config
    return f"{recipe}__{metric}__balance{balance:g}__{aggregation}"


def _one_row(label, kind, R, config, result):
    recipe, metric, balance, aggregation=config
    al=alignment(R)  # post-hoc only; NEVER used inside objective/optimizer
    rec=dict(
        rotation=label,kind=kind,recipe=recipe,metric=metric,
        balance=balance,aggregation=aggregation,
        score=float(result["score"]),
        stratified_score=float(result["stratified_score"]),
        pooled_score=float(result["pooled_score"]),
        axis_mean_abs_cos=al["axis_mean_abs_cos"],
        axis_min_abs_cos=al["axis_min_abs_cos"],
        axis_max_angle_deg=al["axis_max_angle_deg"],
        axis_mean_angle_deg=al["axis_mean_angle_deg"],
        **{f"condition_{s}_loss":float(z["loss"])
           for s,z in zip(NAMES,result["by_condition"])},
        **{f"condition_{s}_effective_dims":float(z["effective_dims"])
           for s,z in zip(NAMES,result["by_condition"])},
    )
    return rec


def write_csv(path, rows):
    if not rows:
        return
    with open(path,"w",newline="",encoding="utf-8") as dst:
        writer=csv.DictWriter(dst, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_and_optimize(M, args):
    balances=sorted(set(args.balances))
    configs=[make_config(recipe,metric,balance,aggregation)
             for recipe in args.recipes for metric in args.metrics
             for balance in balances for aggregation in AGGREGATIONS]
    rng=np.random.default_rng(args.seed+541)
    rotation_bank=list(candidate_rotations(seed=args.seed,
                                           random_count=args.random_rotations))
    score_rows=[]
    by_config={config:[] for config in configs}
    matrices={}
    for name,R,metadata in rotation_bank:
        matrices[name]=R
        for config in configs:
            recipe,metric,balance,agg=config
            result=evaluate(M,R,recipe,metric,balance,agg,args.power,
                            args.complexity_penalty)
            rec=_one_row(name,metadata["kind"],R,config,result)
            score_rows.append(rec)
            by_config[config].append(rec)

    if args.optimize:
        # Identical optimization budget for all generic recipes; for the
        # pooled control we DO NOT optimize, since the scientific target
        # is the sum of condition losses.
        target_configs=[c for c in configs if c[-1]=="stratified"]
        for idx,config in enumerate(target_configs,1):
            recipe,metric,balance,aggregation=config
            print(f"OPTIMIZE {idx}/{len(target_configs)} {config_key(config)}",
                  flush=True)
            for rep in range(args.restarts):
                start=rng.uniform(-1.6,1.6,size=6)
                def objective(params):
                    return evaluate(M,matrix_from_params(params),recipe,metric,
                                    balance,aggregation,args.power,
                                    args.complexity_penalty)["score"]
                # non-smooth near exact zero axis contributions; Powell is
                # derivative-free and does not require finite-differencing
                # the nonsmooth objective.
                fit=minimize(objective,start,method="Powell",
                             options={"maxiter":args.maxiter,
                                      "xtol":1e-3,"ftol":1e-7})
                R=matrix_from_params(fit.x)
                label=f"optimized_{config_key(config)}_{rep+1:02d}"
                matrices[label]=R
                # Every optimized R gets scored on every config, allowing
                # fair re-use of candidate rotations across objectives.
                for other in configs:
                    r,m,b,a=other
                    detail=evaluate(M,R,r,m,b,a,args.power,
                                    args.complexity_penalty)
                    rec=_one_row(label,"optimized",R,other,detail)
                    score_rows.append(rec)
                    by_config[other].append(rec)

    report={}
    for config in configs:
        items=by_config[config]
        reference=next(z for z in items if z["kind"]=="truth_reference")
        # The true R and oracle-centered pairwise sweeps are held-out
        # post-hoc diagnostics, NEVER candidate optimizers or recovery
        # selections.
        searched=[z for z in items if z["kind"] in ("haar","optimized")]
        best=min(searched,key=lambda x:x["score"]) if searched else None
        random=[z for z in items if z["kind"]=="haar"]
        key=config_key(config)
        chosen=best or reference
        R=matrices[chosen["rotation"]]
        recipe,metric,balance,agg=config
        report[key]=dict(
            recipe=recipe,metric=metric,balance=balance,aggregation=agg,
            identity_score=reference["score"],
            best_searched=best,
            best_matrix=R.tolist() if best is not None else None,
            identity_minus_best=float(reference["score"]-best["score"])
                   if best is not None else None,
            fraction_random_better_than_identity=(
                float(np.mean([z["score"] < reference["score"]-1e-10 for z in random]))
                if random else None),
            identity=evaluate(M,np.eye(4),recipe,metric,balance,agg,
                              args.power,args.complexity_penalty),
            best_details=evaluate(M,R,recipe,metric,balance,agg,
                                  args.power,args.complexity_penalty)
                if best is not None else None,
        )
    return score_rows,report


def plot_planes(rows, folder, recipes, metrics, balances):
    """Fixed-angle diagnostic landscapes (not optimization candidates)."""
    import matplotlib
    matplotlib.use("Agg",force=True)
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True,exist_ok=True)
    for recipe in recipes:
        for metric in metrics:
            for balance in sorted(set(balances)):
                for agg in AGGREGATIONS:
                    relevant=[r for r in rows if r["recipe"]==recipe and
                              r["metric"]==metric and r["balance"]==balance and
                              r["aggregation"]==agg]
                    ref=next((r["score"] for r in relevant
                              if r["rotation"]=="identity"),None)
                    if ref is None:continue
                    fig,ax=plt.subplots(figsize=(8,5))
                    for a in range(4):
                        for b in range(a+1,4):
                            prefix=f"plane_{a+1}{b+1}_"
                            part=sorted(
                                (r for r in relevant if
                                 r["rotation"].startswith(prefix)),
                                key=lambda z:int(z["rotation"].rsplit("_",1)[-1]))
                            degrees=[0]+[int(z["rotation"].rsplit("_",1)[-1])
                                         for z in part]
                            vals=[ref]+[z["score"] for z in part]
                            ax.plot(degrees,vals,marker="o",markersize=3,
                                    label=f"{a+1}-{b+1}")
                    ax.set(title=f"{recipe} | {metric} | balance={balance:g} | {agg}",
                           xlabel="Rotation from true reference (degrees)",
                           ylabel="Dimension sparsity loss (smaller is simpler)")
                    ax.grid(alpha=.25)
                    ax.legend(ncol=3,fontsize=8)
                    fig.tight_layout()
                    fig.savefig(folder/f"{recipe}_{metric}_balance{balance:g}_{agg}.png",
                                dpi=130)
                    plt.close(fig)


def report_markdown(output, channels, args):
    lines=[
        "# Universal condition-wise dimension sparsity, analytic CR3",
        "",
        "This experiment applies the same L(R;P_s) in all four conditions",
        "and compares SUM_s pi_s L_s(R) against L computed AFTER pooling moments.",
        "No CR3-specific axis or special zero condition is hardcoded in the loss.",
        "",
        "All R are shared, orthogonal 4x4; theta labels are used only for",
        "post-hoc alignment. Original U/UU and condition-tangent UX moments",
        "are loaded from the prior analytic A/B/C run, not recomputed.",
        "",
        f"Input tag: `{args.source_tag}`, output tag: `{args.tag}`.",
        f"p for the lp family: {args.power}; automatic top-k penalty: {args.complexity_penalty}.",
        f"UX source: {args.ux_kind}.",
        "Balance: 0=empirical shares, 1=four equal category shares.",
        "",
    ]
    for channel in channels:
        lines.extend([f"## {channel}","",
                     "| Generic L | Balance | Aggregation | Identity | Best blind search | Max axis angle | Random better |",
                     "|---|---:|---|---:|---:|---:|---:|"])
        result_file=output/channel/"best_rotations.json"
        if not result_file.exists():continue
        result=json.loads(result_file.read_text(encoding="utf-8"))
        for k,entry in result.items():
            best=entry["best_searched"]
            s=best["score"] if best else float("nan")
            angle=best["axis_max_angle_deg"] if best else float("nan")
            randombetter=entry["fraction_random_better_than_identity"]
            lines.append(
                f"| {entry['recipe']} / {entry['metric']} | "
                f"{entry['balance']:g} | {entry['aggregation']} | "
                f"{entry['identity_score']:.6g} | {s:.6g} | "
                f"{angle:.1f}° | {randombetter:.1%} |" if randombetter is not None
                else f"| {k} | missing random results |")
        lines+=[""]
    lines += [
        "## Scientific limitations",
        "",
        "- The identity is a held-out analytic reference, not a search seed.",
        "- General conditions may truly require all four U dimensions.",
        "- U gradient energy provides a coordinate-dependence proxy, not",
        "  a measured drop-k predictor test or intrinsic manifold dimension.",
        "- UX means only condition-valid X derivatives; zero amount's",
        "  non-existent X derivative is not silently replaced by zero.",
        "- This stage is pure analytic truth, not decoder fidelity.",
        "- Smaller loss alone cannot guarantee recovery of generating axes.",
    ]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def parse(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-tag",default="analytic_cr3_64")
    ap.add_argument("--tag",default="universal_group_v1")
    ap.add_argument("--source-dir",type=Path,default=None)
    ap.add_argument("--output-dir",type=Path,default=None)
    ap.add_argument("--channels",nargs="+",choices=("logit","prob"),
                    default=("logit","prob"))
    ap.add_argument("--ux-kind",
                    choices=("tangent","ambient","finite_tangent"),default="tangent")
    ap.add_argument("--recipes",nargs="+",choices=tuple(COMPONENTS),
                    default=DEFAULT_RECIPES)
    ap.add_argument("--metrics",nargs="+",choices=METRICS,default=DEFAULT_METRICS)
    ap.add_argument("--balances",nargs="+",type=float,default=[0.,1.])
    ap.add_argument("--power",type=float,default=.5,
                    help="0<power<1 exponent in sum_i q_i^power")
    ap.add_argument("--complexity-penalty",type=float,default=.3,
                    help="lambda for min_k(tail(k)+lambda*k/zdim)")
    ap.add_argument("--random-rotations",type=int,default=96)
    ap.add_argument("--restarts",type=int,default=2)
    ap.add_argument("--maxiter",type=int,default=80)
    ap.add_argument("--no-optimize",dest="optimize",action="store_false",default=True)
    ap.add_argument("--seed",type=int,default=20261117)
    ap.add_argument("--no-plots",dest="plots",action="store_false",default=True)
    return ap.parse_args(argv)


def main(argv=None):
    args=parse(argv)
    if not (0<args.power<1 and args.complexity_penalty>0):
        raise ValueError("power in (0,1) and complexity-penalty>0 required")
    if not args.balances or any(not 0<=x<=1 for x in args.balances):
        raise ValueError("all balances in [0,1]")
    if args.random_rotations<1:
        raise ValueError("Need at least one random blind rotation for meaningful assessment")
    if args.restarts<1 or args.maxiter<1:
        raise ValueError("restarts and maxiter must be positive")
    source=args.source_dir or BASE/args.source_tag
    target=args.output_dir or BASE/"group_sparsity"/args.tag
    source=Path(source)
    target=Path(target)
    if source.resolve()==target.resolve():
        raise ValueError("Output must not overwrite input analytic A/B/C artifacts")
    target.mkdir(parents=True,exist_ok=True)
    data_hashes={}
    for channel in args.channels:
        moment_path=source/channel/"moments.npz"
        if not moment_path.is_file():
            raise FileNotFoundError(f"Missing {moment_path}. Run A/B/C derivation first.")
        data_hashes[channel]=checksum(moment_path)
    conf=dict(source=str(source.resolve()),data_hashes=data_hashes,
              channels=list(args.channels),ux_kind=args.ux_kind,
              recipes=list(args.recipes),metrics=list(args.metrics),
              balances=sorted(set(args.balances)),power=args.power,
              complexity_penalty=args.complexity_penalty,
              random_rotations=args.random_rotations,
              restarts=args.restarts,maxiter=args.maxiter,
              optimize=args.optimize,seed=args.seed,plots=args.plots)
    confpath=target/"config.json"
    if confpath.exists():
        old=json.loads(confpath.read_text(encoding="utf-8"))
        if old!=conf:
            raise RuntimeError("This output tag already has a different config. "
                               "Use --tag NEW_TAG (existing results are protected).")
    json_dump(confpath,conf)

    for channel in args.channels:
        stat=load_moments(source/channel/"moments.npz")
        M=prepare(stat,ux_kind=args.ux_kind)
        print(f"GROUP SPARSITY {channel} | 4 condition counts {M['counts'].tolist()}",
              flush=True)
        rows,best=sample_and_optimize(M,args)
        folder=target/channel
        folder.mkdir(parents=True,exist_ok=True)
        write_csv(folder/"rotation_scores.csv",rows)
        json_dump(folder/"best_rotations.json",best)
        json_dump(folder/"moment_summaries.json",{
            "trial_counts":M["counts"].tolist(),
            "ux_source":M["ux_kind"],"channel":channel,"x_scale":M["x_scale"],
            "ux_coordinate_support":M["supports"].tolist(),
            "component_trace_by_condition":{
                name:np.trace(mats,axis1=-2,axis2=-1).tolist()
                for name,mats in M["components"].items()},
            "mixture_weights":{
                f"{b:g}":mixing_weights(M["counts"],b).tolist()
                for b in sorted(set(args.balances))}
        })
        if args.plots:
            plot_planes(rows,folder/"figures",args.recipes,args.metrics,args.balances)
        print(f"DONE {channel}: {len(rows)} rotation/config score rows; "
              f"{len(best)} objective+aggregation variants",flush=True)
    report_markdown(target,args.channels,args)
    print("REPORT",target/"REPORT.md",flush=True)
    return target


if __name__=="__main__":
    main()
