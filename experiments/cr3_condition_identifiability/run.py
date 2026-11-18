"""Analytic CR3 condition identifiability A/B/C benchmark and SO(4) scan.

Run as: python -m experiments.cr3_condition_identifiability.run --stage all
Only reads S0/S1 files; never edits checkpoints/latents/other experiment data.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .design import (
    ROOT,BANK,DATASET,NAMES,X_NAMES,PARAM_NAMES,EXPECTED,
    load_design,derivative_masks,physical_finite_probes
)
from .derivatives import compute_fields,save_fields,true_generator_selfcheck
from .objectives import moments,score_all,fisher_diagnostics,scan,METRICS

BASE=ROOT/"outputs/cr3_condition_identifiability"


def dump_json(path,obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=lambda x: x.tolist() if isinstance(x,np.ndarray) else str(x)),
                    encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_moments(path,stat):
    d=dict(F=stat["F"],G_cov=stat["G_cov"],H4=stat["H4"],
           counts=stat["counts"],channel=np.array(stat["channel"]),
           x_scale=np.array(stat["x_scale"]))
    for name in stat["C_cov"]:
        d[f"C_cov_{name}"]=stat["C_cov"][name]
        d[f"C_support_{name}"]=stat["C_support"][name]
    np.savez_compressed(path,**d)


def load_moments(path):
    from .objectives import C_KINDS
    with np.load(path,allow_pickle=False) as d:
        stat=dict(F=d["F"],G_cov=d["G_cov"],H4=d["H4"],
                  counts=d["counts"],channel=str(d["channel"]),
                  x_scale=str(d["x_scale"]))
        stat["C_cov"]={kind:d[f"C_cov_{kind}"] for kind in C_KINDS}
        stat["C_support"]={kind:d[f"C_support_{kind}"] for kind in C_KINDS}
    return stat


def diagnostic_summary(design,fields,stats):
    masks=fields["ambient_valid"]; tangent=fields["tangent_valid"]
    fg=fields.get("G_logit",fields["G"])
    gain=design.codes==0
    neutral=design.codes==1
    case={
        "subjects":len(design.theta),"stimuli":len(design.x),
        "category_counts":{name:int(stats["counts"][i]) for i,name in enumerate(NAMES)},
        "parameter_coordinates":"T=(theta-mu)/sd, true-parameter standardized",
        "X_coordinates":fields["x_scale"],
        "theory_axis_diagnostics_not_used_for_search":{
            "gain_only_gradient_lambda_max_abs":float(np.abs(fg[:,gain,1]).max()) if gain.any() else None,
            "gamma_neutral_gradient_gamma_max_abs":float(np.abs(fg[:,neutral,2]).max()) if neutral.any() else None,
        },
        "per_category_derivative_support":{},
        "fisher_by_category":fisher_diagnostics(stats),
        "identity_rotation_scores":{
            mix:score_all(stats,np.eye(4),mix) for mix in ("empirical","balanced")
        },
        "warning":"X gradients at zero amount and p endpoints are UNDEFINED and excluded by masks; finite contrasts there are not point derivatives.",
    }
    for i,name in enumerate(NAMES):
        m=design.codes==i
        case["per_category_derivative_support"][name]=dict(
            n_trials=int(m.sum()),
            ambient_per_X=np.sum(masks[m],axis=0).astype(int).tolist(),
            tangent_per_X=np.sum(tangent[m],axis=0).astype(int).tolist(),
            finite_tangent_per_X=np.sum(fields["finite_kind"][m]==0,axis=0).astype(int).tolist(),
            finite_boundary_per_X=np.sum(fields["finite_kind"][m]==1,axis=0).astype(int).tolist(),
            finite_cross_condition_per_X=np.sum(fields["finite_kind"][m]==2,axis=0).astype(int).tolist(),
            number_all_X_interior=int(np.count_nonzero(masks[m].all(axis=1))),
        )
    return case


def derive(args,design,channel,out):
    out.mkdir(parents=True,exist_ok=True)
    protocol=dict(
        bank_sha256=design.bank_hash, dataset_sha256=design.dataset_hash,
        subject_ids=design.subject_ids.tolist(),
        trial_ids=design.trial_ids.tolist(),
        amount_step=float(args.amount_step),
        probability_step=float(args.probability_step),
        x_scale=args.x_scale,channel=channel,
    )
    protocol_path=out/"protocol.json"
    if protocol_path.exists() and not args.force:
        previous=read_json(protocol_path)
        if previous!=protocol:
            raise RuntimeError("Existing output tag uses a different data/step/scaling protocol: "+
                               str(protocol_path)+
                               ". Choose a NEW --tag, or explicitly --force to recompute.")
    dump_json(protocol_path,protocol)
    fields_path=out/"fields.npz"
    moments_path=out/"moments.npz"
    if fields_path.exists() and moments_path.exists() and not args.force:
        print("REUSE derivative fields",channel,out,flush=True)
        return load_moments(moments_path)
    print("DERIVE",channel,flush=True)
    fields=compute_fields(
        design,channel=channel,device=args.device,batch=args.batch,
        amount_step=args.amount_step,probability_step=args.probability_step,
        x_scale=args.x_scale,finite=True)
    stat=moments(design,fields)
    save_fields(fields_path,design,fields,save_dense=True)
    save_moments(moments_path,stat)
    diag=diagnostic_summary(design,fields,stat)
    dump_json(out/"diagnostics.json",diag)
    print("DERIVE COMPLETE",channel,flush=True)
    return stat


def write_csv(path,rows):
    if not rows:return
    cols=list(rows[0])
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def make_plane_plots(rows,out):
    """Six plane-sweep curves per objective, identity marked at 0 degrees."""
    import matplotlib
    matplotlib.use("Agg",force=True)
    import matplotlib.pyplot as plt
    selected=("A_jointdiag_plus_null","B_H_offdiag",
              "B_H_top4_tail","C_tangent_top1",
              "C_finite_tangent_top1","C_finite_boundary_top1")
    out.mkdir(parents=True,exist_ok=True)
    for mix in ("empirical","balanced"):
        for metric in selected:
            fig,ax=plt.subplots(figsize=(9,5))
            for i in range(4):
                for j in range(i+1,4):
                    prefix=f"plane_{i+1}{j+1}_"
                    subset=[r for r in rows if r["mixture"]==mix and
                            r["metric"]==metric and r["rotation"].startswith(prefix)]
                    subset.sort(key=lambda r:int(r["rotation"].rsplit("_",1)[-1]))
                    baseline=next((r["score"] for r in rows if r["rotation"]=="identity" and
                                   r["mixture"]==mix and r["metric"]==metric),float("nan"))
                    angles=[0]+[int(r["rotation"].rsplit("_",1)[-1]) for r in subset]
                    values=[baseline]+[r["score"] for r in subset]
                    ax.plot(angles,values,marker="o",markersize=2.8,
                            label=f"theta{i+1}-theta{j+1}")
            ax.set(xlabel="Pairwise plane rotation (degrees)",
                   ylabel="loss (lower is simpler)",
                   title=f"{metric}, {mix}: analytic truth at 0 / 90")
            ax.grid(alpha=.2)
            ax.legend(ncol=3,fontsize=8)
            fig.tight_layout()
            fig.savefig(out/f"{mix}__{metric}.png",dpi=140)
            plt.close(fig)


def search_one(args,stat,channel,out):
    print("SEARCH",channel,flush=True)
    rows,best=scan(stat,mixtures=("empirical","balanced"),
                   random_count=args.random_rotations,seed=args.seed,
                   optimize=args.optimize,restarts=args.restarts,maxiter=args.maxiter)
    write_csv(out/"rotations_all_scores.csv",rows)
    dump_json(out/"best_rotations.json",best)
    make_plane_plots(rows,out/"figures")
    print("SEARCH COMPLETE",channel,len(rows),"score rows",flush=True)
    return best


def report(out,channels):
    sections=["# CR3 analytic conditional identifiability: A / B / C",
              "",
              "All X are from the unchanged original CR3 fixed bank. Analytic-only:",
              "no neural decoder, no latent Z, no learned model family.",
              "",
              "A: conditional choice-Fisher information and two distinct null axes.",
              "B: U-only gradient and Hessian sparsity across four trial conditions.",
              "C: interior UX, condition-tangent UX, plus finite X interventions at boundaries.",
              "",
              "All rotations are SO(4); true theta axes appear only in post-hoc evaluation.",
              "0 and 90 degree plane cases are signed-permutation-equivalent baselines.",
              ""]
    for channel in channels:
        path=out/channel
        sections+=["## "+channel,""]
        if (path/"diagnostics.json").exists():
            d=read_json(path/"diagnostics.json")
            sections+=["Subjects: "+str(d["subjects"])+", stimuli: "+str(d["stimuli"])+".",
                       "Conditions: "+str(d["category_counts"])+".",""]
            sections+=["Structural-zero checks (gradient w.r.t standardized true parameters):",
                       "gain-only lambda max abs: "+str(d["theory_axis_diagnostics_not_used_for_search"]["gain_only_gradient_lambda_max_abs"]),
                       "gamma-neutral gamma max abs: "+str(d["theory_axis_diagnostics_not_used_for_search"]["gamma_neutral_gradient_gamma_max_abs"]),
                       ""]
            sections+=["X derivative support per category:"]
            for kind,info in d["per_category_derivative_support"].items():
                sections+=[f"- {kind}: interior X counts {info['ambient_per_X']}; tangent X counts {info['tangent_per_X']}; all-X interior trials {info['number_all_X_interior']}"]
            sections+=[""]
        if (path/"best_rotations.json").exists():
            b=read_json(path/"best_rotations.json")
            sections+=["| Mixture/objective | Identity loss | Best searched loss | Best alignment mean | Max axis angle |",
                       "|---|---:|---:|---:|---:|"]
            for key,item in b.items():
                sections.append(
                    f"| {key} | {item['identity_score']:.6g} | {item['best']['score']:.6g} | "
                    f"{item['best']['axis_mean_abs_cos']:.3f} | "
                    f"{item['best']['axis_max_angle_deg']:.1f}° |")
            sections+=[""]
        else:
            sections+=["Search not run yet.",""]
    sections+=["## Interpretation",
               "",
               "- A full-rank total Fisher matrix establishes local theta identifiability;",
               "  it does not select a rotation of an unrestricted decoder.",
               "- Near-zero gain-only lambda and gamma-neutral gamma gradients",
               "  are analytic properties of the generator, not training-based discoveries.",
               "- A low simplicity loss does NOT prove recovery of the generating axes.",
               "- Boundary finite contrasts and the classic UX Hessian measure",
               "  different estimands. Their results must never be pooled without labels.",
               "- Compare per-condition and empirical/balanced mixtures separately.",
               "- The NN-fidelity gap is outside this analytic-only stage.",
               ""]
    (out/"REPORT.md").write_text("\n".join(sections),encoding="utf-8")
    return out/"REPORT.md"


def parse(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage",choices=("all","precheck","derive","search","report"),default="all")
    ap.add_argument("--tag",default="analytic_v1")
    ap.add_argument("--subjects",type=int,default=32)
    ap.add_argument("--stimuli",type=int,default=500)
    ap.add_argument("--seed",type=int,default=20261117)
    ap.add_argument("--channels",nargs="+",choices=("logit","prob"),default=["logit","prob"])
    ap.add_argument("--device",default="auto")
    ap.add_argument("--batch",type=int,default=128)
    ap.add_argument("--amount-step",type=float,default=5.0)
    ap.add_argument("--probability-step",type=float,default=.05)
    ap.add_argument("--x-scale",choices=("raw","standardized"),default="standardized")
    ap.add_argument("--random-rotations",type=int,default=96)
    ap.add_argument("--restarts",type=int,default=2)
    ap.add_argument("--maxiter",type=int,default=90)
    ap.add_argument("--no-optimize",action="store_false",dest="optimize",default=True)
    ap.add_argument("--bank",type=Path,default=BANK)
    ap.add_argument("--dataset",type=Path,default=DATASET)
    ap.add_argument("--force",action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args=parse(argv)
    if args.device=="auto":
        args.device="cuda" if torch.cuda.is_available() else "cpu"
    if args.subjects<1 or args.batch<1 or args.random_rotations<0 or args.restarts<1:
        raise ValueError("subjects/batch/restarts must be positive, random_rotations nonnegative")
    out=BASE/args.tag
    out.mkdir(parents=True,exist_ok=True)
    if args.stage in ("precheck","derive","all"):
        dump_json(out/"config.json",{**vars(args),"bank":str(args.bank),"dataset":str(args.dataset)})
    elif args.stage=="search":
        dump_json(out/"search_config.json",{
            "tag":args.tag,"channels":args.channels,"seed":args.seed,
            "random_rotations":args.random_rotations,
            "restarts":args.restarts,"maxiter":args.maxiter,
            "optimize":args.optimize
        })
    if args.stage in ("precheck","derive","all"):
        d=load_design(args.subjects,args.stimuli,args.seed,args.bank,args.dataset)
        check=true_generator_selfcheck(d)
        ambient,tangent=derivative_masks(d.x,d.codes)
        probes=physical_finite_probes(d.x,d.codes,args.amount_step,args.probability_step)
        dump_json(out/"precheck.json",{
            "status":"passed","exact_generator_max_logit_difference":check,
            "bank_sha256":d.bank_hash,"dataset_sha256":d.dataset_hash,
            "counts":{name:int((d.codes==i).sum()) for i,name in enumerate(NAMES)},
            "all_six_X_interior_count":int(np.count_nonzero(ambient.all(axis=1))),
            "tangent_valid_counts":np.sum(tangent,axis=0).tolist(),
            "finite_one_sided_count":int(np.sum(probes[-1])),
            "original_data_unchanged":True,
            "warning":"analytic-only: original NN decoder and its X normalization are not modified",
        })
        print("PRECHECK PASS",out,flush=True)
        if args.stage=="precheck":return
        for channel in args.channels:
            stat=derive(args,d,channel,out/channel)
            if args.stage=="all":
                search_one(args,stat,channel,out/channel)
    if args.stage=="search":
        for channel in args.channels:
            cache=out/channel/"moments.npz"
            if not cache.exists():
                raise FileNotFoundError(f"derive first: {cache}")
            search_one(args,load_moments(cache),channel,out/channel)
    result=report(out,args.channels)
    print("COMPLETE",result,flush=True)


if __name__=="__main__":
    main()
