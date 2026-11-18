"""Post-freeze ALL-SUBJECT Oracle R vs unsupervised R* on identical Hessian losses.

Run retroactively on an existing hessian_sparsity folder WITHOUT
recomputing decoder AD fields or rerunning the SO(d) blind optimizer.

Oracle definitions:
  (1) R_oracle_procrustes: SO(d) orthogonal Procrustes mapping ALL q
      to standardized ALL theta.
  (2) R_oracle: optimize SO(d) to maximize mean Pearson² on ALL subjects.
      This is an approximate supervised oracle, not a Hessian fit.

True theta is read ONLY here, AFTER fit_complete.json and rotation.npz.
There is no TEST sample, and no parameter generalization estimate.
FIT-X and VALID-X use different stimulus sets but the SAME all-subject cohort.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"np"/"src")]

import numpy as np
import torch
from scipy.optimize import minimize,linear_sum_assignment
from rotations.hessian_sparsity.data import sha256
from rotations.hessian_sparsity.loss import summarize_fields,HessianLoss
from rotations.hessian_sparsity.optimizer import givens,pairs


def _corr(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    ac=a-a.mean(0);bc=b-b.mean(0)
    denom=np.sqrt(np.sum(ac*ac,axis=0)[:,None]*np.sum(bc*bc,axis=0)[None,:])
    return np.divide(ac.T@bc,denom,out=np.zeros((a.shape[1],b.shape[1])),
                     where=denom>1e-14)


def _is_so(R,tol=1e-8):
    return (np.isfinite(R).all() and
            np.allclose(R@R.T,np.eye(len(R)),atol=tol) and
            abs(float(np.linalg.det(R))-1)<tol)


def procrustes_oracle(q_train,theta_train):
    """Closest mapping q@R.T -> z-scored theta using ALL subjects."""
    q=np.asarray(q_train,dtype=float)
    theta=np.asarray(theta_train,dtype=float)
    if q.shape!=theta.shape or q.ndim!=2 or len(q)<q.shape[1]+1:
        raise ValueError("Oracle requires matched ALL-subject q/theta matrices")
    qc=q-q.mean(0); tc=theta-theta.mean(0)
    sd=tc.std(0,ddof=1)
    if np.any(sd<1e-10) or not np.isfinite(sd).all():
        raise ValueError("Oracle true parameter has no ALL-subject variance")
    # For R rows, solve max tr(R @ Q.T @ T_std), i.e.
    # R = U V^T for T_std.T @ Q = U diag(s) V^T.
    cross=(tc/sd).T@qc
    U,s,Vt=np.linalg.svd(cross)
    fix=np.eye(len(s))
    fix[-1,-1]=float(np.sign(np.linalg.det(U@Vt)))
    R=U@fix@Vt
    if not _is_so(R):
        raise RuntimeError("ALL-subject Procrustes R is not in SO(d)")
    return R,dict(all_theta_mean=theta.mean(0).tolist(),
                  all_theta_sd=sd.tolist(),
                  singular_values=s.tolist())


def correlation_oracle(q_train,theta_train,R_initial,restarts=20,maxiter=300,seed=981):
    """Approximate full-SO(d) ALL-subject Pearson² oracle; no Hessian role assumptions."""
    q=np.asarray(q_train,dtype=float)
    t=np.asarray(theta_train,dtype=float)
    R_initial=np.asarray(R_initial,dtype=float)
    d=q.shape[1]
    if q.shape!=t.shape or R_initial.shape!=(d,d) or not _is_so(R_initial):
        raise ValueError("Oracle shape/initial SO(d) mismatch")
    if not np.isfinite(q).all() or not np.isfinite(t).all():
        raise ValueError("Nonfinite ALL-subject q/theta")
    qc=q-q.mean(0);tc=t-t.mean(0)
    cov_q=qc.T@qc
    cross=tc.T@qc
    theta_energy=np.square(tc).sum(0)
    if np.min(np.linalg.eigvalsh(cov_q))<1e-10 or np.min(theta_energy)<1e-10:
        raise ValueError("Degenerate ALL-subject q or theta for correlation oracle")
    cq=torch.as_tensor(cov_q,dtype=torch.float64)
    ct=torch.as_tensor(cross,dtype=torch.float64)
    energy=torch.as_tensor(theta_energy,dtype=torch.float64)
    base=torch.as_tensor(R_initial,dtype=torch.float64)
    def scores(R):
        numerator=torch.einsum("ia,ia->i",R,ct)
        variance=torch.einsum("ia,ab,ib->i",R,cq,R)
        return (numerator*numerator/(energy*variance.clamp(min=1e-14)))
    def objective(angle):
        a=torch.tensor(angle,dtype=torch.float64,requires_grad=True)
        R=givens(a,d)@base
        vals=scores(R)
        val=-vals.mean()
        g=torch.autograd.grad(val,a)[0]
        return float(val.detach()),g.detach().numpy().astype(float)
    rng=np.random.default_rng(seed)
    zeros=np.zeros(len(pairs(d)),dtype=float)
    starts=[zeros]+[rng.uniform(-np.pi,np.pi,size=len(zeros))
                    for _ in range(max(int(restarts)-1,0))]
    records=[]
    best=None
    for i,initial in enumerate(starts):
        res=minimize(objective,initial,method="L-BFGS-B",jac=True,
                     bounds=[(-4*np.pi,4*np.pi)]*len(initial),
                     options=dict(maxiter=int(maxiter),ftol=1e-13,gtol=1e-9))
        angles=np.asarray(res.x,dtype=float)
        R=(givens(torch.as_tensor(angles,dtype=torch.float64),d)@base).detach().numpy()
        correlations=_corr(t,q@R.T)
        diag=np.diag(correlations)
        score=float(np.mean(diag**2))
        records.append(dict(start=i,all_mean_diag_r2=score,
                            scipy_success=bool(res.success),
                            iterations=int(res.nit)))
        if best is None or score>best["all_mean_diag_r2"]:
            best=dict(R=R,all_mean_diag_r2=score,start=i)
    if not _is_so(best["R"]):
        raise RuntimeError("ALL-subject correlation Oracle R not in SO(d)")
    return best["R"],dict(
        method="approximate_SO(d)_maximize_ALL_parameter_specific_Pearson_r2",
        mean_all_r2=best["all_mean_diag_r2"],
        selected_start=int(best["start"]),
        num_restarts=len(starts),all_starts=records,
        all_subjects=True,used_hessian_losses_for_oracle=False,
        no_test_split=True)


def _param_scores(theta,q,R,labels,oracle_axes=False):
    """Descriptive in-sample Pearson r² on ALL subjects, with no TEST."""
    u=q@R.T
    correlation=_corr(theta,u)
    if oracle_axes:
        matched={name:i for i,name in enumerate(labels)}
    else:
        i,j=linear_sum_assignment(-np.square(correlation))
        matched={labels[a]:int(b) for a,b in zip(i,j)}
    r2={name:float(correlation[a,matched[name]]**2)
        for a,name in enumerate(labels)}
    return dict(
        all_correlation=correlation.tolist(),
        all_matched_axis=matched,matched_all_r2=r2,
        mean_matched_all_r2=float(np.mean(list(r2.values()))),
        oracle_diagonal_all_r2=(float(np.mean(np.diag(correlation)**2))
                                if oracle_axes else None),
        n_subjects=len(q),
        scope="ALL subjects, no independent TEST; in-sample parameter alignment",
        labels=labels)


def _fields(path,include_ux):
    with np.load(path,allow_pickle=False) as f:
        H=np.asarray(f["H_UU"],float)
        J=np.asarray(f["H_UX"],float) if include_ux else None
        code=np.asarray(f["condition_codes"],dtype=int)
    return dict(UU=H,UX=J,group_codes=code)


def _load_old(run_dir):
    out=Path(run_dir).resolve()
    for p in ("protocol.json","fit_complete.json","rotation.npz",
              "fit_fields.npz","validation_fields.npz","loss_diagnostics.json"):
        if not (out/p).is_file():
            raise FileNotFoundError(f"Oracle post-fit analysis needs existing {out/p}")
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    frozen=json.loads((out/"fit_complete.json").read_text(encoding="utf-8"))
    if not frozen.get("rotation_frozen") or not (frozen.get("oracle_theta_used_for_optimization") is False):
        raise RuntimeError("Original R* has not been documented as theta-blind frozen")
    mode=protocol["config"]["mode"]
    grouping=protocol["config"]["grouping"]
    if (mode!=frozen.get("mode") or grouping!=frozen.get("grouping")):
        raise RuntimeError("Original run mode/grouping provenance inconsistent")
    model=protocol["model"];cond=protocol["condition"];repeat=int(protocol["repeat"])
    ck=ROOT/"outputs"/"choice"/model/"s3_train"/cond/f"repeat{repeat:02d}"/"s3_train__checkpoint.pt"
    latent=ROOT/"outputs"/"choice"/model/"s4_evaluate"/cond/f"repeat{repeat:02d}"/"s4_evaluate__latent.npz"
    dataset=ROOT/"outputs"/"choice"/model/"s1_generate"/"s1_generate__dataset.npz"
    banks={"CR3":"CR3_4P_fixed500.npz","CR4":"CR4_3P_random500.npz",
           "CR5":"CR5_3P_cr3bank500.npz"}
    bank=ROOT/"outputs"/"choice"/model/"s0_design"/banks[model]
    for key,path in (("checkpoint",ck),("latent",latent),
                     ("dataset",dataset),("bank",bank)):
        if not path.is_file() or sha256(path)!=protocol["hashes"][key]:
            raise RuntimeError(f"{key} source hash changed since R* optimization")
    with np.load(out/"rotation.npz",allow_pickle=False) as a:
        R=np.asarray(a["R"],float)
        q=np.asarray(a["q"],float)
        if "all_subject_ids" not in a.files:
            raise RuntimeError("Legacy subset rotation cannot be represented as ALL-subject R*. "
                               "Re-run hessian_sparsity/run.py with the new all-subject protocol.")
        all_ids=np.asarray(a["all_subject_ids"],int)
        fit_x=np.asarray(a["fit_x_ids"],int)
        valid_x=np.asarray(a["valid_x_ids"],int)
    if not _is_so(R):
        raise RuntimeError("Original optimizer R* is not SO(d)")
    if q.ndim!=2 or q.shape[1]!=len(R):
        raise ValueError("Original PCA whitened q does not match R*")
    if (protocol.get("rotation_subject_scope")!="all_subjects_no_test" or
        not np.array_equal(all_ids,np.arange(len(q))) or
        protocol.get("all_subject_count")!=len(q)):
        raise RuntimeError("Not a true ALL-subject rotation protocol")
    for ids,name in ((fit_x,"fit_x_ids"),(valid_x,"valid_x_ids")):
        if not np.array_equal(ids,np.asarray(protocol[name],dtype=int)):
            raise RuntimeError(f"{name} changed since R* fitting")
    with np.load(dataset,allow_pickle=False) as a:
        theta=np.asarray(a["parameters"],float)
    labels=(["alpha","lambda_loss","gamma","tau"] if model=="CR3"
            else ["lambda_loss","gamma","tau"])
    if theta.shape!=(len(q),len(labels)):
        raise RuntimeError("Source theta does not match original q")
    cfg=protocol["config"]
    include_ux=mode!="uu_only"
    f=_fields(out/"fit_fields.npz",include_ux)
    v=_fields(out/"validation_fields.npz",include_ux)
    options=dict(mode=mode,uu_weight=float(cfg["uu_weight"]),
                 ux_weight=float(cfg["ux_weight"]),epsilon=float(cfg["epsilon"]))
    fit_loss=HessianLoss(summarize_fields(f,grouping,include_ux),**options)
    valid_loss=HessianLoss(summarize_fields(v,grouping,include_ux),**options)
    prev=json.loads((out/"loss_diagnostics.json").read_text(encoding="utf-8"))
    for label,objective,field in (("FIT",fit_loss,"fit_loss"),
                                  ("VALID",valid_loss,"validation_loss")):
        if abs(objective.value(R)-float(prev[field]["total"]))>1e-9:
            raise RuntimeError(f"{label} Hessian loss no longer reproduces R* run")
    return dict(out=out,model=model,q=q,R=R,theta=theta,labels=labels,
                all_ids=all_ids,fit_loss=fit_loss,valid_loss=valid_loss,
                protocol=protocol)


def compare_existing(run_dir,*,force=False,oracle_restarts=20,oracle_maxiter=300,
                     oracle_seed=981):
    src=_load_old(run_dir)
    out=src["out"]
    result_path=out/"oracle_comparison.json"
    if result_path.is_file() and not force:
        print("Existing Oracle comparison preserved:",result_path,flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))
    q,theta,Rstar=src["q"],src["theta"],src["R"]
    d=len(Rstar)
    Rpro,pro_meta=procrustes_oracle(q,theta)
    Roracle,corr_meta=correlation_oracle(
        q,theta,Rpro,restarts=oracle_restarts,
        maxiter=oracle_maxiter,seed=oracle_seed)
    matrices={"identity_PCA":np.eye(d),"optimized_R_star":Rstar,
              "oracle_R_procrustes":Rpro,"oracle_R_correlation":Roracle}
    reports={}
    for label,R in matrices.items():
        fit=src["fit_loss"].scalar(R)
        valid=src["valid_loss"].scalar(R)
        reports[label]=dict(
            FIT=fit,VALID=valid,
            ALL=_param_scores(theta,q,R,src["labels"],
                               oracle_axes=label.startswith("oracle_R")))
    gap_fit=float(reports["oracle_R_correlation"]["FIT"]["total"]-
                  reports["optimized_R_star"]["FIT"]["total"])
    gap_val=float(reports["oracle_R_correlation"]["VALID"]["total"]-
                  reports["optimized_R_star"]["VALID"]["total"])
    alignment=Roracle@Rstar.T
    report=dict(
        model=src["model"],
        condition=src["protocol"]["condition"],
        mode=src["protocol"]["config"]["mode"],
        grouping=src["protocol"]["config"]["grouping"],
        oracle_definition="ALL-subject SO(d) maximize mean named-parameter Pearson r^2; approximate multistart solution",
        oracle_procrustes_definition="ALL-subject SO(d) orthogonal least-squares fit to standardized theta",
        original_R_star_remains_unchanged=True,
        original_source_hashes=src["protocol"]["hashes"],
        theta_used_in_blind_optimization=False,
        theta_used_for_oracle=True,
        oracle_subject_scope="ALL subjects, no held-out TEST",
        hessian_loss_evaluation_scope="Original saved FIT-X and VALID-X derivative fields; both include ALL subjects",
        procrustes_metadata=pro_meta,
        correlation_oracle_metadata=corr_meta,
        matrices={name:R.tolist() for name,R in matrices.items()},
        comparison=reports,
        oracle_minus_optimized=dict(FIT=gap_fit,VALID=gap_val),
        oracle_higher_hessian_loss_in_FIT=gap_fit>0,
        oracle_higher_hessian_loss_in_VALID=gap_val>0,
        optimized_to_oracle_axis_overlap=np.abs(alignment).tolist(),
        note="Oracle is an ALL-subject supervised orthogonal approximation; no independent test performance is claimed. Never use Oracle to choose blind loss.")
    # Write all diagnostics separately. No mutation of the frozen R or
    # original evaluation/loss protocol.
    np.savez_compressed(out/"oracle_rotations.npz",
                        R_optimized=Rstar,R_oracle=Roracle,
                        R_oracle_procrustes=Rpro,
                        R_identity=np.eye(d),
                        optimized_to_oracle_alignment=alignment)
    with (out/"oracle_loss_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        names=["rotation","FIT_total","VALID_total",
               "FIT_UU","VALID_UU","FIT_UX","VALID_UX",
               "ALL_mean_matched_r2"]
        writer=csv.DictWriter(f,fieldnames=names)
        writer.writeheader()
        for name,detail in reports.items():
            def part(region,key):
                items=detail[region]["by_group"]
                vals=[g[key] for g in items]
                return None if vals[0] is None else float(np.mean(vals))
            writer.writerow(dict(
                rotation=name,FIT_total=detail["FIT"]["total"],
                VALID_total=detail["VALID"]["total"],
                FIT_UU=part("FIT","uu"),VALID_UU=part("VALID","uu"),
                FIT_UX=part("FIT","ux"),VALID_UX=part("VALID","ux"),
                ALL_mean_matched_r2=detail["ALL"]["mean_matched_all_r2"]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4.1))
    keys=list(reports)
    xpos=np.arange(len(keys))
    labels=["PCA","R* (blind)","Oracle (Procrustes)","Oracle (corr)"]
    for ax,split in zip(axs,("FIT","VALID")):
        values=[reports[k][split]["total"] for k in keys]
        ax.bar(xpos,values)
        ax.set(xticks=xpos,xticklabels=labels,title=f"{src['model']} {split} Hessian loss",
               ylabel="Original Hessian sparsity loss")
        ax.tick_params(axis="x",labelrotation=25)
        for i,val in enumerate(values):
            ax.text(i,val,f"{val:.4f}",ha="center",va="bottom",fontsize=9)
    fig.tight_layout()
    fig.savefig(out/"oracle_loss_comparison.png",dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4.4))
    im=ax.imshow(alignment,vmin=-1,vmax=1,cmap="viridis")
    ax.set(xticks=range(d),yticks=range(d),
           xticklabels=[f"R*_axis{i+1}" for i in range(d)],
           yticklabels=src["labels"],
           xlabel="Blind Hessian R* axes",ylabel="Oracle named-parameter axes",
           title="Absolute cosine overlap between Oracle R and R*")
    for i in range(d):
        for j in range(d):
            ax.text(j,i,f"{abs(alignment[i,j]):.2f}",ha="center",va="center",
                    color="white" if abs(alignment[i,j])<.5 else "black",fontsize=9)
    fig.colorbar(im,ax=ax)
    fig.tight_layout()
    fig.savefig(out/"oracle_axis_overlap.png",dpi=160)
    plt.close(fig)
    result_path.write_text(json.dumps(report,indent=2,allow_nan=False),
                           encoding="utf-8")
    print("Oracle vs optimized R* on IDENTICAL Hessian FIT/VALID:",flush=True)
    for name in matrices:
        one=reports[name]
        print(f"  {name:26s} FIT={one['FIT']['total']:.6f} "
              f"VALID={one['VALID']['total']:.6f} "
              f"ALL_r2={one['ALL']['mean_matched_all_r2']:.4f}",flush=True)
    print("  Oracle - R*: FIT",gap_fit,"VALID",gap_val,flush=True)
    print("Saved post-fit Oracle comparison:",result_path,flush=True)
    return report


def _find_run(args):
    if args.run_dir:
        return Path(args.run_dir)
    if not args.model:
        raise ValueError("Choose --model or --run-dir")
    condition=args.condition
    if condition is None:
        condition=("beta0p0005_lambda0_ctx0p25to0p8_L4_xstd"
                   if args.model=="CR3" else
                   "beta0p0005_lambda0_ctx0p25to0p8_L3")
    if args.model=="CR3" and not condition.endswith("_xstd"):
        condition+="_xstd"
    parent=(ROOT/"outputs"/"choice"/args.model/"rotations"/"hessian_sparsity"/
            condition/f"repeat{args.repeat:02d}")
    folders=sorted(parent.glob(f"{args.tag}_{args.mode}_{args.grouping}_*"))
    folders=[f for f in folders if (f/"fit_complete.json").is_file()
             and json.loads((f/"protocol.json").read_text(encoding="utf-8")).get(
                 "rotation_subject_scope")=="all_subjects_no_test"]
    if len(folders)!=1:
        raise FileNotFoundError(
            f"Expected one completed run, found {len(folders)} at {parent}. "
            "Use --run-dir with an exact existing output folder.")
    return folders[0]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",default=None)
    p.add_argument("--model",choices=("CR3","CR4","CR5"))
    p.add_argument("--condition",default=None)
    p.add_argument("--repeat",type=int,default=1)
    p.add_argument("--tag",default="main")
    p.add_argument("--mode",choices=("joint","uu_only","ux_only"),default="joint")
    p.add_argument("--grouping",choices=("pooled","balanced"),default="pooled")
    p.add_argument("--oracle-restarts",type=int,default=20)
    p.add_argument("--oracle-maxiter",type=int,default=300)
    p.add_argument("--oracle-seed",type=int,default=981)
    p.add_argument("--force",action="store_true")
    args=p.parse_args(argv)
    if args.oracle_restarts<1 or args.oracle_maxiter<1:
        p.error("Oracle restarts and maxiter must be positive")
    return compare_existing(_find_run(args),force=args.force,
                            oracle_restarts=args.oracle_restarts,
                            oracle_maxiter=args.oracle_maxiter,
                            oracle_seed=args.oracle_seed)


if __name__=="__main__":
    main()
