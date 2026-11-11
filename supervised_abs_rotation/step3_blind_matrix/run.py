"""Step 3 — BLIND, WY-only search for simple latent coordinates (C1, CR3).

READ PERMITTED: Step-2 W.npy, common Y.npz, trial/subject IDs and structural
metadata for alignment. NEVER read true theta, generator formula, trained NN,
R_best, per-case R, or per-case G_relative when evaluating metrics.

Case identities and angles are revealed only AFTER all scores have been
calculated and frozen. Metrics compare a subject x response-feature matrix F
against U extracted from each trial W. There is NO per-angle NN retraining.

Examples:
  python supervised_abs_rotation/step3_blind_matrix/run.py --model C1
  python supervised_abs_rotation/step3_blind_matrix/run.py --model CR3
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Explicit local import path for both direct Windows runs and pytest importlib.
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))
from profiles import CHANNELS, load_profiles, profile_views
from metrics import (MATRIX_METRICS, CONTROL_METRICS, FUNCTION_METRICS,
                     normalize_response, linear_fingerprint, functional_scores, _poly)

ALL_METRICS = (*MATRIX_METRICS, *CONTROL_METRICS, *FUNCTION_METRICS)
MAIN_PLOTS = (
    "B_column_entropy_loss",
    "B_column_top1_tail_loss",
    "function_quad_interaction_gap",
    "function_cubic_interaction_gap",
    "function_budget_auc_loss",
)


def tag(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p").replace("+","")


def condition(beta, lam, cmin, cmax, dim):
    return (f"beta{tag(beta)}_lambda{tag(lam)}_"
            f"ctx{tag(cmin)}to{tag(cmax)}_L{dim}")


def digest(path):
    h = hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda:stream.read(2*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def save_csv(path, rows, columns):
    with Path(path).open("w",newline="",encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,fieldnames=columns,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    Path(path).write_text(
        json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
        encoding="utf-8")


def inspect_cases(step2, model, require_complete=True):
    """Metadata only for input integrity; do NOT return angle/plane to scorer."""
    step2 = Path(step2)
    index_path = step2 / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"No Step 2 index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("model") != model:
        raise ValueError(f"Step 2 data are {index.get('model')} not {model}")
    expected = 37 if int(index["Z_dim"]) == 4 else 91
    if require_complete and not index.get("complete"):
        raise ValueError("Step 2 incomplete: rerun step2_pairwise_sweep/run.py")
    cases_dir = step2/"cases"
    paths = sorted(x for x in cases_dir.glob("*/W.npy") if x.is_file())
    if require_complete and len(paths) != expected:
        raise ValueError(f"Expected {expected} full W cases, found {len(paths)}")
    if len(paths)<2:
        raise ValueError("Need at least two W cases")
    documented = {case["id"] for case in index["cases"]}
    found = {p.parent.name for p in paths}
    if documented != found and require_complete:
        raise ValueError("Actual W cases do not match Step-2 index")
    n = int(index["n_subjects"]);m = int(index["n_trials"])
    dz = int(index["Z_dim"]);dx = int(index["X_dim"])
    if n < 8 or m < n or dz < 2 or dx < 1:
        raise ValueError("Invalid Step 2 geometry")
    return paths, index


def _case_u(wpath, sid, offsets, *, n, m, dz, dx,
            x_reference, check_rows):
    """Extract U from W, not R or U_subject, verifying per-trial consistency."""
    w = np.load(wpath,mmap_mode="r",allow_pickle=False)
    if w.shape != (m,dz+dx) or w.dtype.kind != "f":
        raise ValueError(f"Wrong W dimensions: {wpath}")
    u = np.asarray(w[offsets[:-1],:dz],dtype=np.float64)
    # Same person's trials must share U; X and trial alignment must be fixed.
    if not np.isfinite(u).all():
        raise ValueError(f"Nonfinite U: {wpath}")
    verify = np.r_[check_rows, offsets[:-1], offsets[1:]-1]
    verify = np.unique(verify)
    if not np.allclose(w[verify,:dz], u[sid[verify]],rtol=1e-6,atol=1e-6):
        raise ValueError(f"Trial rows and subject-level U disagree: {wpath}")
    if not np.array_equal(w[verify,dz:],x_reference[verify,dz:]):
        raise ValueError(f"Stimulus X differs across rotation candidates: {wpath}")
    return u


def _plot_curve(path, records, model, channel, view, metric):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(9.0,5.1))
    planes = sorted({r["plane"] for r in records})
    for plane in planes:
        arr = sorted((r for r in records if r["plane"]==plane),
                     key=lambda q:q["angle_deg"])
        ax.plot([r["angle_deg"] for r in arr],
                [r["loss"] for r in arr],marker="o",lw=1.3,label=plane)
    ax.set_xticks([0,15,30,45,60,75,90])
    ax.set_xlabel("Extra rotation relative to fitted R* (degrees)")
    ax.set_ylabel("Complexity loss (lower = simpler)")
    ax.set_title(f"{model} · {channel} · {view}\n{metric}")
    ax.grid(alpha=.25)
    ax.legend(ncol=3,fontsize=8,loc="best")
    fig.tight_layout()
    output = io.BytesIO()
    try:
        fig.savefig(output,format="png",dpi=155)
        path.write_bytes(output.getvalue())
    finally:
        plt.close(fig)


def _plot_B(path, b, model, channel, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d,p = b.shape
    # Downsample image columns only; the exact full B is saved separately.
    use=np.linspace(0,p-1,min(p,100),dtype=int)
    vals=b[:,use]
    bound=max(float(np.max(np.abs(vals))),1e-8)
    fig,ax=plt.subplots(figsize=(11,max(3.5,d*.65+1.2)))
    im=ax.imshow(vals,aspect="auto",cmap="coolwarm",vmin=-bound,vmax=bound)
    ax.set_yticks(range(d),[f"U{i+1}" for i in range(d)])
    ax.set_xlabel(f"Response-profile features (sampled {len(use)} of {p})")
    ax.set_ylabel("Latent axis")
    ax.set_title(f"{model} · {channel} · {title}")
    fig.colorbar(im,ax=ax,shrink=.8)
    fig.tight_layout()
    buf=io.BytesIO()
    try:
        fig.savefig(buf,format="png",dpi=155)
        path.write_bytes(buf.getvalue())
    finally:
        plt.close(fig)


def _summarize_curves(revealed, index, *, output):
    """Reveal case identities only after blinded score calculation is complete."""
    base = {k:r for r in revealed if r["case"]=="baseline"
            for k in [(r["channel"],r["view"],r["metric"])]}
    grouped=defaultdict(dict)
    for row in revealed:
        key = (row["channel"],row["view"],row["metric"])
        grouped[key][row["case"]] = float(row["loss"])

    plane_map=defaultdict(dict)
    for case in index["cases"]:
        pair=case.get("pair_1based")
        if pair is not None:
            key=f"U{pair[0]}-U{pair[1]}"
            plane_map[key][int(case["angle_deg"])]=case["id"]

    curve_rows=[]
    stats=[]
    ranking_rows=[]
    for key, loss_by_case in sorted(grouped.items()):
        channel,view,metric=key
        if "baseline" not in loss_by_case:
            raise RuntimeError("Missing baseline case in frozen scores")
        baseline=loss_by_case["baseline"]
        sorted_losses=sorted(loss_by_case.items(),key=lambda x:(x[1],x[0]))
        winner=sorted_losses[0][0]
        base_rank=1+sum(
            v < baseline-1e-9 for v in loss_by_case.values())
        for rank,(case,score) in enumerate(sorted_losses,1):
            ranking_rows.append(dict(
                channel=channel,view=view,metric=metric,
                rank=rank,case=case,loss=score,
                is_baseline=int(case=="baseline"),
                is_control=int(metric in CONTROL_METRICS)))
        pass_planes=0
        positive_planes=0
        endpoint_deviation=[]
        inner_contrasts=[]
        per_metric_curves=[]
        for plane,amap in sorted(plane_map.items()):
            if not all(a in amap for a in (15,30,45,60,75,90)):
                continue
            ninety=loss_by_case[amap[90]]
            inner=np.array([loss_by_case[amap[a]] for a in (15,30,45,60,75)])
            endpoint_mean=.5*(baseline+ninety)
            diff=float(np.median(inner)-endpoint_mean)
            spread=max(float(np.ptp(np.r_[baseline,ninety,inner])),1e-12)
            endpoint_shift=abs(float(baseline-ninety))
            # Substantial positive contrast, with 0/90 both near the
            # best middle-angle value. Flat curves are NOT successes.
            passed = (diff > .05*spread
                      and max(baseline,ninety)<float(np.min(inner))+.05*spread
                      and endpoint_shift<=.05*spread)
            pass_planes+=int(passed)
            positive_planes+=int(diff>.05*spread)
            endpoint_deviation.append(endpoint_shift)
            inner_contrasts.append(diff)
            for angle in (0,15,30,45,60,75,90):
                case_name = "baseline" if angle==0 else amap[angle]
                row=dict(channel=channel,view=view,metric=metric,
                         plane=plane,angle_deg=angle,case=case_name,
                         loss=float(loss_by_case[case_name]),
                         delta_from_baseline=float(loss_by_case[case_name]-baseline),
                         plane_endpoint_favored=int(passed))
                curve_rows.append(row);per_metric_curves.append(row)
        stats.append(dict(
            channel=channel,view=view,metric=metric,
            baseline_loss=baseline,baseline_rank=base_rank,
            blind_winner_revealed=winner,n_candidates=len(sorted_losses),
            n_planes=len(endpoint_deviation),n_planes_endpoint_favored=pass_planes,
            n_planes_positive_contrast=positive_planes,
            median_interior_minus_endpoints=float(np.median(inner_contrasts)) if inner_contrasts else None,
            max_endpoint_0vs90_gap=max(endpoint_deviation) if endpoint_deviation else None,
            score_class="invariant_control" if metric in CONTROL_METRICS else
                        ("function_decomposition" if metric in FUNCTION_METRICS else
                         "linear_fingerprint"),
        ))
    save_csv(output/"plane_curves.csv",curve_rows,
             ["channel","view","metric","plane","angle_deg","case","loss",
              "delta_from_baseline","plane_endpoint_favored"])
    save_csv(output/"rankings.csv",ranking_rows,
             ["channel","view","metric","rank","case","loss",
              "is_baseline","is_control"])
    save_csv(output/"metric_summary.csv",stats,[
        "channel","view","metric","score_class","baseline_loss",
        "baseline_rank","blind_winner_revealed","n_candidates",
        "n_planes","n_planes_endpoint_favored",
        "n_planes_positive_contrast","median_interior_minus_endpoints",
        "max_endpoint_0vs90_gap"])
    return curve_rows,stats


def analyze(*, model, step2, output, channels=CHANNELS,
            seed=31, train_fraction=.7,
            fingerprint_ridge=1e-2, nonlinear_ridge=.1,
            max_profile_outputs=64, profile_ridge=1e-3,
            overwrite=False, plots=True):
    """Entire scoring phase reads ONLY anonymous (W,Y) matrices."""
    if not (0.4<train_fraction<.9):
        raise ValueError("Train fraction must be between .4 and .9")
    if (fingerprint_ridge<0 or nonlinear_ridge<=0
            or profile_ridge<=0 or max_profile_outputs<1):
        raise ValueError("Invalid regularization/profile settings")
    step2=Path(step2)
    output=Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Existing Step 3 results: {output}; use --overwrite")
    case_paths,index=inspect_cases(step2,model)
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(case_paths))
    # Only W paths and anonymous IDs are passed to numerical computations.
    # Angles, pair names, R_best and T remain inaccessible to scorer.
    blind_cases=[(f"B{i+1:03d}",case_paths[k]) for i,k in enumerate(order)]
    profiles,profile_meta,(counts,sid,tid,offsets)=load_profiles(
        step2,blind_cases[0][1].parent,channels=channels,ridge=profile_ridge)
    n=len(counts)
    m=len(sid)
    dz=int(index["Z_dim"]);dx=int(index["X_dim"])
    if int(index["n_subjects"])!=n or int(index["n_trials"])!=m:
        raise ValueError("Profile sample sizes disagree with Step2 index")
    shuffled=rng.permutation(n)
    split=int(train_fraction*n)
    train,test=np.sort(shuffled[:split]),np.sort(shuffled[split:])
    # Fix profile normalization ONCE across all angles; exclude uninformative
    # stimulus columns consistently (not independently per rotation).
    cached={}
    cache_meta={}
    for ch,profile in profiles.items():
        for view,array in profile_views(profile,profile_meta["profile_kind"]).items():
            normalized,keep=normalize_response(array,train)
            if normalized is None:
                raise ValueError(f"No variable response features: {ch} {view}")
            cached[(ch,view)]=normalized
            cache_meta[f"{ch}::{view}"]=keep

    check_rows=np.unique(np.r_[0,m-1, rng.choice(m,min(m,1024),replace=False)])
    reference=np.load(blind_cases[0][1],mmap_mode="r",allow_pickle=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".step3_blind_build_",
                                     dir=output.parent) as temporary:
        folder=Path(temporary)/"step3_blind_matrix"
        folder.mkdir()
        (folder/"plots").mkdir()
        # Compute every score before reading per-case angle metadata.
        frozen=[]
        audit=[]
        for idx,(blind_id,path) in enumerate(blind_cases,1):
            u=_case_u(path,sid,offsets,n=n,m=m,dz=dz,dx=dx,
                      x_reference=reference,check_rows=check_rows)
            polynomial = _poly(u,train)  # shared across Y channels and views
            for (channel,view),f_std in cached.items():
                linear,linfo,_=linear_fingerprint(
                    u,f_std,train,ridge=fingerprint_ridge,
                    pre_normalized=True)
                nonlinear,ninfo=functional_scores(
                    u,f_std,train,test,ridge=nonlinear_ridge,
                    max_profile_outputs=max_profile_outputs,
                    pre_normalized=True,precomputed_poly=polynomial)
                if linear is None or nonlinear is None:
                    raise ValueError(f"Missing scores for {channel} {view}")
                values={**linear,**nonlinear}
                if set(values) != set(ALL_METRICS):
                    raise AssertionError("Scoring table does not match frozen metrics")
                if not all(np.isfinite(v) for v in values.values()):
                    raise ArithmeticError(f"Nonfinite score: {blind_id}, {channel}/{view}")
                for metric,value in values.items():
                    frozen.append({
                        "blind_id":blind_id,"channel":channel,"view":view,
                        "metric":metric,"loss":float(value),
                        "is_control":int(metric in CONTROL_METRICS),
                    })
            if idx==1 or idx%5==0 or idx==len(blind_cases):
                print(f"[Step3 blind] {model}: scored {idx}/{len(blind_cases)} "
                      f"anonymous W matrices",flush=True)

        save_csv(folder/"blind_scores.csv",frozen,
                 ["blind_id","channel","view","metric","loss","is_control"])
        # FROZEN BLIND SCORING END: angle/case identities are read only now.
        reveal={blind_id:path.parent.name for blind_id,path in blind_cases}
        save_csv(folder/"blind_key.csv",
                 [{"blind_id":b,"case":name} for b,name in reveal.items()],
                 ["blind_id","case"])
        enriched=[{**row,"case":reveal[row["blind_id"]]} for row in frozen]
        save_csv(folder/"revealed_scores.csv",enriched,
                 ["blind_id","case","channel","view","metric","loss","is_control"])
        curves,stats=_summarize_curves(enriched,index,output=folder)

        plot_errors={}
        if plots:
            for ch in channels:
                if ch not in ("logit_nn","p_nn"):continue
                for metric in MAIN_PLOTS:
                    view="contrast" if metric.startswith("B_") else "full"
                    rows=[r for r in curves if r["channel"]==ch and
                          r["view"]==view and r["metric"]==metric]
                    if rows:
                        filepath=folder/"plots"/f"curve_{ch}_{metric}.png"
                        try:
                            _plot_curve(filepath,rows,model,ch,view,metric)
                        except OSError as error:
                            plot_errors[filepath.name]=str(error)

            # Export full B for baseline and all six 45-degree single-plane
            # perturbations for the top two NN channels; this is done AFTER
            # blind scoring, only for interpretation.
            diag=folder/"diagnostic_B"
            diag.mkdir()
            show_cases=["baseline"]+[c["id"] for c in index["cases"]
                                     if c.get("angle_deg")==45]
            for case in show_cases:
                path=step2/"cases"/case/"W.npy"
                u=_case_u(path,sid,offsets,n=n,m=m,dz=dz,dx=dx,
                          x_reference=reference,check_rows=check_rows)
                for ch in channels:
                    if ch not in ("logit_nn","p_nn"):continue
                    score,_,b=linear_fingerprint(
                        u,cached[(ch,"contrast")],train,ridge=fingerprint_ridge,
                        pre_normalized=True)
                    np.save(diag/f"B_{ch}_{case}.npy",b)
                    try:
                        _plot_B(diag/f"B_{ch}_{case}.png",b,model,ch,case)
                    except OSError as error:
                        plot_errors[f"B_{ch}_{case}.png"]=str(error)
        source_fingerprints={
            "step2_index_sha256":digest(step2/"index.json"),
            "step2_shared_Y_sha256":digest(step2/"Y.npz"),
        }
        save_json(folder/"summary.json",{
            "model":model, "n_subjects":n, "n_trials":m, "z_dim":dz,
            "x_dim":dx,"n_cases":len(blind_cases),
            "n_training_subjects":len(train),"n_test_subjects":len(test),
            "channels":list(channels),"score_formulae":list(ALL_METRICS),
            "control_metrics":list(CONTROL_METRICS),
            "seed":seed,"train_fraction":train_fraction,
            "profile":profile_meta,"response_columns_kept":cache_meta,
            "ridge_fingerprint":fingerprint_ridge,
            "ridge_nonlinear":nonlinear_ridge,
            "max_profile_outputs_for_function_scores":max_profile_outputs,
            "source_files":source_fingerprints,
            "blind_scoring_policy": (
                "Case order seeded and anonymized; numerical scorer receives "
                "only per-trial W and fixed common Y-derived subject F. "
                "Case identity / rotation angles unblinded only after every "
                "metric has been saved in blind_scores.csv."
            ),
            "data_sources": ["step2/cases/*/W.npy", "step2/Y.npz",
                             "step2/subject_index.npy", "step2/trial_index.npy",
                             "step2/trial_counts.npy", "step2/index.json for geometry"],
            "prohibited_sources_used": [],
            "scope": "Exploratory blind structure screening; not proof baseline is optimal.",
            "function_models":(
                "degree-2 and degree-3 polynomial in subject-level U; "
                "train-only response standardization; train subject fit, "
                "separate held-out subject scoring. U is reconstructed from "
                "W and no decoder is trained or differentiated."
            ),
            "comparison_note":(
                "All W are invertible rotations; population Bayes-optimal "
                "predictive accuracy and B Frobenius norm are invariant. "
                "Only axis-dependent sparsity, additive decomposition and "
                "fixed feature-budget accuracy can favor baseline. "
                "Original S5 posterior uses all outcomes: this is NOT "
                "independent trial-level generalization."
            ),
            "plot_errors":plot_errors,
        })
        train_set = set(train.tolist())
        save_csv(folder/"subject_splits.csv",
                 [{"subject_index":int(i),
                   "split":"train" if i in train_set else "test"}
                  for i in range(n)],["subject_index","split"])
        # Publish completed directory without leaving half-written results.
        backup=None
        if output.exists():
            backup=output.with_name(output.name+"_previous")
            if backup.exists():
                raise FileExistsError(f"Existing Step 3 backup: {backup}")
            os.replace(output,backup)
        try:
            os.replace(folder,output)
        except BaseException:
            if backup is not None and backup.is_dir() and not output.exists():
                os.replace(backup,output)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    print(f"[Step3] {model}: {len(blind_cases)} cases, "
          f"{len(stats)} channel/view/metric summaries -> {output}",flush=True)
    return {"output":str(output),"n_cases":len(blind_cases),
            "n_metrics_summary":len(stats)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices=("C1","CR3"),required=True)
    p.add_argument("--beta",type=float,default=None)
    p.add_argument("--latent-dim",type=int,default=4)
    p.add_argument("--lambda-sparse",type=float,default=0.)
    p.add_argument("--context-min",type=float,default=.25)
    p.add_argument("--context-max",type=float,default=.8)
    p.add_argument("--repeat",type=int,default=1)
    p.add_argument("--objective",choices=("abs_mse","signed_mse"),default="abs_mse")
    p.add_argument("--step2-dir",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--seed",type=int,default=31)
    p.add_argument("--train-fraction",type=float,default=.7)
    p.add_argument("--channels",default=",".join(CHANNELS))
    p.add_argument("--max-profile-outputs",type=int,default=64)
    p.add_argument("--fingerprint-ridge",type=float,default=.01)
    p.add_argument("--nonlinear-ridge",type=float,default=.1)
    p.add_argument("--profile-ridge",type=float,default=.001)
    p.add_argument("--no-plots",action="store_true")
    p.add_argument("--overwrite",action="store_true")
    args=p.parse_args(argv)
    if args.beta is None:
        args.beta = .00005 if args.model=="CR3" else .0005
    if args.latent_dim!=4 or args.repeat<1 or args.beta<0 or not (
            0<args.context_min<=args.context_max<=1):
        p.error("Invalid NP geometry or condition")
    channels=tuple(s.strip() for s in args.channels.split(",") if s.strip())
    if not channels or len(channels)!=len(set(channels)) or any(
            s not in CHANNELS for s in channels):
        p.error(f"--channels must be unique subset of {CHANNELS}")
    step1=(ROOT/"outputs"/"supervised_abs_rotation"/args.model/
           condition(args.beta,args.lambda_sparse,args.context_min,
                     args.context_max,args.latent_dim)/
           f"repeat{args.repeat:02d}"/args.objective)
    step2=args.step2_dir or step1/"step2_pairwise_sweep"
    output=args.output or step1/"step3_blind_matrix"
    analyze(model=args.model,step2=step2,output=output,
            channels=channels,seed=args.seed,
            train_fraction=args.train_fraction,
            fingerprint_ridge=args.fingerprint_ridge,
            nonlinear_ridge=args.nonlinear_ridge,
            max_profile_outputs=args.max_profile_outputs,
            profile_ridge=args.profile_ridge,
            plots=not args.no_plots,overwrite=args.overwrite)


if __name__=="__main__":
    main()
