"""Gaussian-integrated role-free rotation: NO observed subject Z is an AD node.

Examples:
  python rotations/gaussian_integrated_rotation/run.py --model CR3
  python rotations/gaussian_integrated_rotation/run.py --model CR4 --x-mode generative
  python rotations/gaussian_integrated_rotation/run.py --model CR5 --set z_cutoff=null
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"np"/"src")]
import numpy as np
import torch

from rotations.gaussian_integrated_rotation.data import load_source
from rotations.gaussian_integrated_rotation.quadrature import gaussian_nodes
from rotations.gaussian_integrated_rotation.x_integration import x_nodes
from rotations.gaussian_integrated_rotation.teacher import restore,logit
from rotations.gaussian_integrated_rotation.moments import accumulate,save_moments,load_moments
from rotations.gaussian_integrated_rotation.loss import GaussianHessianLoss
from rotations.gaussian_integrated_rotation.optimizer import optimize,plane_landscapes
from rotations.gaussian_integrated_rotation.oracle import postfreeze

HERE=Path(__file__).resolve().parent
CONDITIONS={"CR3":"beta0p0005_lambda0_ctx0p25to0p8_L4_xstd",
            "CR4":"beta0p0005_lambda0_ctx0p25to0p8_L3",
            "CR5":"beta0p0005_lambda0_ctx0p25to0p8_L3"}


def json_out(path,obj):
    def default(o):
        if isinstance(o,np.ndarray):return o.tolist()
        if isinstance(o,np.generic):return o.item()
        raise TypeError(str(type(o)))
    Path(path).write_text(json.dumps(obj,indent=2,default=default,
                                      allow_nan=False),encoding="utf-8")


def code_hash():
    h=hashlib.sha256()
    for p in sorted(HERE.glob("*.py"))+[HERE/"config.json"]:
        h.update(str(p.name).encode());h.update(p.read_bytes())
    return h.hexdigest()


def config_from_args(a):
    cfg=json.loads((HERE/"config.json").read_text(encoding="utf-8"))
    cfg.update(seed=a.seed,device=a.device,
               x_mode=a.x_mode,grouping=a.grouping,
               derivatives=a.derivatives)
    if a.x_spec_file:
        cfg["x_distribution_file"]=str(Path(a.x_spec_file).expanduser().resolve())
    for item in a.set:
        key,sep,value=item.partition("=")
        if not sep or key not in cfg:
            raise ValueError(f"Invalid --set {item}: allowed {sorted(cfg)}")
        try:cfg[key]=json.loads(value)
        except json.JSONDecodeError:
            if isinstance(cfg[key],str):cfg[key]=value
            else:raise ValueError(f"Value for {key} must be a JSON literal")
    if cfg["derivatives"] not in ("uu_only","ux_only","joint"):
        raise ValueError("Invalid derivatives")
    if cfg["grouping"] not in ("pooled","balanced"):
        raise ValueError("Invalid grouping")
    if cfg["x_mode"] not in ("bank","generative"):
        raise ValueError("Invalid X integration mode")
    for key in ("z_fit_points","z_validation_points","x_fit_points",
                "x_validation_points","x_batch","z_batch",
                "restarts","maxiter","oracle_restarts","oracle_maxiter"):
        if int(cfg[key])<1:raise ValueError(f"Config {key} must be positive")
    return cfg


def audit(model,device,source,R,x,mu,sd,q_nodes):
    """Reparametrizing q by R cannot change frozen decoder predictions."""
    q=q_nodes[:min(5,len(q_nodes))]
    z_original=source["mu"]+q@source["B"].T
    u=q@R.T
    z_rebuilt=source["mu"]+(u@R)@source["B"].T
    z_error=float(np.max(np.abs(z_original-z_rebuilt)))
    xx=(x[:min(8,len(x))]-mu)/sd
    xt=torch.as_tensor(xx,dtype=torch.float32,device=device).unsqueeze(0).expand(len(q),-1,-1).clone()
    with torch.no_grad():
        pred=logit(model,torch.as_tensor(z_original,dtype=torch.float32,device=device),xt)
        new=logit(model,torch.as_tensor(z_rebuilt,dtype=torch.float32,device=device),xt)
        p_error=float(torch.max(torch.abs(torch.sigmoid(pred)-torch.sigmoid(new))))
    if z_error>1e-9 or p_error>5e-6:
        raise RuntimeError("SO(d) rotation changed frozen decoder predictions")
    return dict(Z_max_abs_error=z_error,p_max_abs_error=p_error,
                det_R=float(np.linalg.det(R)),
                orthogonality_max_abs=float(np.max(np.abs(R@R.T-np.eye(len(R))))))


def _get_nodes(source,cfg):
    dim=source["B"].shape[1]
    method=cfg["z_method"]
    opt=dict(dim=dim,method=method,cutoff=cfg["z_cutoff"])
    fit_q,fit_w,fit_meta=gaussian_nodes(
        **opt,n=int(cfg["z_fit_points"]),
        order=int(cfg["z_grid_order"]),seed=int(cfg["seed"])+11)
    valid_q,valid_w,valid_meta=gaussian_nodes(
        **opt,n=int(cfg["z_validation_points"]),
        order=int(cfg.get("z_grid_validation_order",9)),
        seed=int(cfg["seed"])+37)
    xf,xwf,cf,xfmeta=x_nodes(source,cfg["x_mode"],
        n=int(cfg["x_fit_points"]),seed=int(cfg["seed"])+91,
        bank_limit=int(cfg["x_bank_limit"]),
        x_spec_file=cfg["x_distribution_file"])
    xv,xwv,cv,xvmeta=x_nodes(source,cfg["x_mode"],
        n=int(cfg["x_validation_points"]),seed=int(cfg["seed"])+173,
        bank_limit=int(cfg["x_bank_limit"]),
        x_spec_file=cfg["x_distribution_file"])
    if cfg["x_mode"]=="bank":
        # Both Q integrals use EXACTLY the same fixed empirical X measure.
        # No continuous interpolation or artificial subject trial sampling.
        xv,xwv,cv,xvmeta=xf.copy(),xwf.copy(),cf.copy(),dict(xfmeta)
    if cfg["grouping"]=="balanced" and source["model"] in ("CR3","CR5"):
        for codes in (cf,cv):
            if len(np.unique(codes))!=4:
                raise ValueError("Balanced grouping requires all four native CR3/CR5 X categories")
    return ((fit_q,fit_w,fit_meta,xf,xwf,cf,xfmeta),
            (valid_q,valid_w,valid_meta,xv,xwv,cv,xvmeta))


def _write_planes(path,records,dim):
    with (path/"plane_landscapes.csv").open("w",newline="",encoding="utf-8") as file:
        w=csv.DictWriter(file,fieldnames=list(records[0]))
        w.writeheader();w.writerows(records)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for i in range(1,dim+1):
        for j in range(i+1,dim+1):
            lines=[v for v in records if v["i"]==i and v["j"]==j]
            fig,ax=plt.subplots(figsize=(6.5,4.))
            for key in ("FIT","VALID"):
                ax.plot([v["angle_deg"] for v in lines],
                        [v[key] for v in lines],label=key)
            ax.set(title=f"Gaussian-integrated SO({dim}): U{i} / U{j}",
                   xlabel="Further rotation (degrees)",
                   ylabel="Hessian RMS sparsity")
            ax.legend();fig.tight_layout()
            fig.savefig(path/f"plane_u{i}_u{j}.png",dpi=145)
            plt.close(fig)


def run(args):
    if args.repeat<1 or args.seed<0 or not args.tag.strip():
        raise ValueError("Invalid repeat, seed or tag")
    cfg=config_from_args(args)
    condition=args.condition or CONDITIONS[args.model]
    if args.model=="CR3" and not condition.endswith("_xstd"):
        condition+="_xstd"
    source=load_source(args.model,condition,args.repeat,cfg)
    qfit,qvalid=_get_nodes(source,cfg)
    fit_q,fit_w,fit_q_meta,fit_x,fit_xw,fit_code,fit_x_meta=qfit
    valid_q,valid_w,valid_q_meta,valid_x,valid_xw,valid_code,valid_x_meta=qvalid
    meta=dict(suite=HERE.name,model=args.model,condition=condition,
              repeat=args.repeat,tag=args.tag,
              dim=source["B"].shape[1],
              code_sha256=code_hash(),source_hashes=source["hashes"],
              all_subject_count=source["all_subject_count"],
              subjects_used_only_for="Gaussian mu and covariance; post-freeze Oracle",
              frozen_decoder=True,theta_used_to_optimize_R=False,
              gaussian=dict(fit=fit_q_meta,valid=valid_q_meta,
                            original_Z_mean=source["mu"].tolist(),
                            original_Z_covariance=source["covariance"].tolist(),
                            native_Z_eigenvalues=source["eigenvalues"].tolist()),
              X=dict(fit=fit_x_meta,valid=valid_x_meta),
              config=cfg,
              caveat="Gaussian approximation to empirical posterior means; CR3/CR5 p=0/1 only matter for UX; default UU-only.")
    finger=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()[:16]
    out=(source["root"]/"rotations"/HERE.name/condition/f"repeat{args.repeat:02d}"/
         f"{args.tag}_{cfg['derivatives']}_{cfg['x_mode']}_{finger}")
    if args.dry_run:
        print("Gaussian source paths and hashes verified:",source["checkpoint"])
        print("Output:",out)
        print(f"{args.model} N_subjects={source['all_subject_count']} (only mu/cov), "
              f"FIT={len(fit_q)}q x {len(fit_x)}X, "
              f"VALID={len(valid_q)}q x {len(valid_x)}X")
        return out
    if out.exists():
        if (out/"fit_complete.json").exists():
            print("Existing frozen Gaussian rotation preserved:",out)
            if not args.no_evaluate and not (out/"oracle_comparison.json").is_file():
                fit_m=load_moments(out/"moments_fit.npz")
                valid_m=load_moments(out/"moments_validation.npz")
                with np.load(out/"rotation.npz",allow_pickle=False) as saved:
                    R=np.asarray(saved["R"],np.float64)
                postfreeze(source,R,fit_m,valid_m,cfg,out)
            return out
        raise FileExistsError("Incomplete run folder exists: "+str(out))
    if cfg["z_method"] in ("hermite","grid") and np.array_equal(fit_q,valid_q):
        print("WARNING: FIT and VALID have identical quadrature nodes; "
              "increase validation order to assess integration convergence.")
    model,device,mu,sd=restore(source,cfg["device"])
    source["x_mu"],source["x_sd"]=mu,sd
    print(f"{args.model} | Gaussian-integrated Hessian, subject-free AD, "
          f"FIT {len(fit_q)} q x {len(fit_x)} X, "
          f"VALID {len(valid_q)} q x {len(valid_x)} X",flush=True)
    def integral(label,pack):
        q,w,_,x,wx,code,_=pack
        def progress(done,total):
            if done==total or done%max(int(cfg["z_batch"]),16)==0:
                print(f" {label} Gaussian nodes {done}/{total}",flush=True)
        return accumulate(model,device,source,q,w,x,wx,code,
                          grouping=cfg["grouping"],
                          derivatives=cfg["derivatives"],
                          z_batch=int(cfg["z_batch"]),
                          x_batch=int(cfg["x_batch"]),progress=progress)
    fit_m=integral("FIT",qfit)
    valid_m=integral("VALID",qvalid)
    def loss(moment):
        return GaussianHessianLoss(moment,mode=cfg["derivatives"],
              epsilon=float(cfg["loss_epsilon"]),
              uu_weight=float(cfg["uu_weight"]),ux_weight=float(cfg["ux_weight"]))
    fit_loss=loss(fit_m);valid_loss=loss(valid_m)
    R,report=optimize(fit_loss,valid_loss,cfg)
    invariance=audit(model,device,source,R,fit_x,mu,sd,fit_q)
    plane=plane_landscapes(R,fit_loss,valid_loss)
    out.mkdir(parents=True,exist_ok=False)
    save_moments(out/"moments_fit.npz",fit_m)
    save_moments(out/"moments_validation.npz",valid_m)
    for name,q,w,x,wx,code in (
        ("fit",fit_q,fit_w,fit_x,fit_xw,fit_code),
        ("validation",valid_q,valid_w,valid_x,valid_xw,valid_code)):
        np.savez_compressed(out/f"quadrature_{name}.npz",
                            q=q,q_weights=w,X_physical=x,X_weights=wx,
                            X_group_codes=code)
    np.savez_compressed(out/"rotation.npz",
                        R=R,affine_native_Z_mean=source["mu"],
                        B=source["B"],W=source["W"],
                        empirical_native_Z_covariance=source["covariance"],
                        q_all_subjects=source["q"],
                        U_all_subjects=source["q"]@R.T,
                        all_subject_ids=np.arange(source["all_subject_count"]))
    json_out(out/"protocol.json",meta)
    json_out(out/"loss_diagnostics.json",report)
    json_out(out/"invariance_audit.json",invariance)
    _write_planes(out,plane,len(R))
    json_out(out/"fit_complete.json",dict(
        status="Gaussian-integrated role-free R frozen",
        theta_used_for_blind_rotation=False,
        gaussian_integral_nodes_no_empirical_Z=True,
        z_population_mean_cov_only=True,
        x_measure=cfg["x_mode"],
        R=R.tolist(),FIT=fit_loss.value(R),VALID=valid_loss.value(R),
        selection_rule="minimize FIT Gaussian integral; no Oracle used"))
    print("Frozen Gaussian integrated R:",out,flush=True)
    print("Hessian simplicity FIT",fit_loss.value(R),"VALID",valid_loss.value(R),flush=True)
    if not args.no_evaluate:
        oracle=postfreeze(source,R,fit_m,valid_m,cfg,out)
        print("ALL-subject (in-sample) mean r² R*",
              oracle["comparison"]["optimized_R_star"]["ALL"]["mean_r2"],
              "Oracle",oracle["comparison"]["oracle_correlation"]["ALL"]["mean_r2"],
              flush=True)
    return out


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",choices=("CR3","CR4","CR5"),required=True)
    ap.add_argument("--condition",default=None)
    ap.add_argument("--repeat",type=int,default=1)
    ap.add_argument("--seed",type=int,default=20811)
    ap.add_argument("--device",default="auto")
    ap.add_argument("--tag",default="main")
    ap.add_argument("--x-mode",choices=("bank","generative"),default="bank")
    ap.add_argument("--x-spec-file",default=None)
    ap.add_argument("--derivatives",choices=("uu_only","ux_only","joint"),default="uu_only")
    ap.add_argument("--grouping",choices=("pooled","balanced"),default="pooled")
    ap.add_argument("--set",action="append",default=[],metavar="KEY=VALUE")
    ap.add_argument("--dry-run",action="store_true")
    ap.add_argument("--no-evaluate",action="store_true")
    args=ap.parse_args(argv)
    return run(args)


if __name__=="__main__":
    main()
