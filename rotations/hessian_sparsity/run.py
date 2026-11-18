"""Standalone role-free Hessian-sparsity rotation of frozen CR3/CR4/CR5.

Usage:
    python rotations/hessian_sparsity/run.py --model CR3 --mode joint
    python rotations/hessian_sparsity/run.py --model CR4 --mode uu_only
    python rotations/hessian_sparsity/run.py --model CR5 --mode joint --grouping balanced
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"np"/"src")]

import numpy as np
import torch
from rotations.hessian_sparsity.data import load_source
from rotations.hessian_sparsity.derivatives import restore,extract,logit
from rotations.hessian_sparsity.loss import summarize_fields,HessianLoss
from rotations.hessian_sparsity.optimizer import optimize
from rotations.hessian_sparsity.planes import save_planes

HERE=Path(__file__).resolve().parent


def as_json(path,obj):
    def normalize(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(f"Cannot serialize {type(value)}")
    Path(path).write_text(json.dumps(obj,indent=2,default=normalize,allow_nan=False),
                          encoding="utf-8")


def source_hash():
    h=hashlib.sha256()
    for f in sorted(HERE.glob("*.py")):
        h.update(f.name.encode());h.update(f.read_bytes())
    return h.hexdigest()


def choose_config(args):
    cfg=json.loads((HERE/"config.json").read_text(encoding="utf-8"))
    cfg.update(seed=args.seed,mode=args.mode,grouping=args.grouping,
               device=args.device)
    for entry in args.set:
        key,sep,value=entry.partition("=")
        if not sep or key not in cfg:
            raise ValueError(f"Unknown --set {entry}; allowed: {sorted(cfg)}")
        try:cfg[key]=json.loads(value)
        except json.JSONDecodeError:
            if isinstance(cfg[key],str):cfg[key]=value
            else:raise ValueError(f"Cannot parse {entry} as JSON literal")
    if (cfg["mode"] not in ("joint","uu_only","ux_only") or
        cfg["grouping"] not in ("pooled","balanced")):
        raise ValueError("Unknown mode or grouping")
    if (min(int(cfg[n]) for n in ("fit_subjects","valid_subjects","fit_x",
                               "valid_x","batch_subjects","restarts","maxiter"))<1):
        raise ValueError("All sample budgets and optimizer iterations must be positive")
    return cfg


def audit_invariance(model,device,source,R,mu,sd):
    q=np.asarray(source["q"])
    reconstructed=(source["pca_mean"]+
                   ((q@R.T)@R)@source["B"].T +source["residual"])
    zerr=float(np.max(np.abs(reconstructed-source["z"])))
    subjects=source["fit_subject_ids"][:4]
    x=(source["bank_x"][:12]-mu)/sd
    xs=torch.as_tensor(x,dtype=torch.float32,device=device).unsqueeze(0).expand(len(subjects),-1,-1)
    with torch.no_grad():
        a=logit(model,torch.as_tensor(source["z"][subjects],dtype=torch.float32,device=device),xs)
        b=logit(model,torch.as_tensor(reconstructed[subjects],dtype=torch.float32,device=device),xs)
        perr=float(torch.max(torch.abs(torch.sigmoid(a)-torch.sigmoid(b))).cpu())
    if zerr>1e-8 or perr>5e-6:
        raise RuntimeError(f"Decoder prediction changed by rotation: zerr={zerr},perr={perr}")
    return {"max_Z_reconstruction_error":zerr,
            "max_predictive_probability_error":perr,
            "orthogonality_error":float(np.max(np.abs(R@R.T-np.eye(len(R))))),
            "determinant":float(np.linalg.det(R))}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=("CR3","CR4","CR5"),required=True)
    parser.add_argument("--condition",default=None)
    parser.add_argument("--repeat",type=int,default=1)
    parser.add_argument("--mode",choices=("joint","uu_only","ux_only"),default="joint")
    parser.add_argument("--grouping",choices=("pooled","balanced"),default="pooled")
    parser.add_argument("--seed",type=int,default=3405)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--tag",default="main")
    parser.add_argument("--set",action="append",default=[],metavar="KEY=VALUE")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--no-evaluate",action="store_true")
    args=parser.parse_args(argv)
    if args.repeat<1 or args.seed<0 or not args.tag.strip():
        parser.error("repeat >=1, nonnegative seed and nonempty tag required")
    dim=4 if args.model=="CR3" else 3
    if args.condition is None:
        args.condition=("beta0p0005_lambda0_ctx0p25to0p8_L4_xstd"
                        if args.model=="CR3" else
                        "beta0p0005_lambda0_ctx0p25to0p8_L3")
    if args.model=="CR3" and not args.condition.endswith("_xstd"):
        args.condition+="_xstd"
    cfg=choose_config(args)
    source=load_source(args.model,args.condition,args.repeat,dim,cfg)
    meta=dict(
        suite="hessian_sparsity",model=args.model,condition=args.condition,
        repeat=args.repeat,tag=args.tag,dimension=dim,config=cfg,
        hashes=source["hashes"],code_sha256=source_hash(),
        fit_subject_ids=source["fit_subject_ids"].tolist(),
        valid_subject_ids=source["valid_subject_ids"].tolist(),
        fit_x_ids=source["fit_x_ids"].tolist(),
        valid_x_ids=source["valid_x_ids"].tolist(),
        original_x_labels=source["group_names"],
        fit_x_condition_counts=np.bincount(
            source["group_codes"][source["fit_x_ids"]],
            minlength=len(source["group_names"])).tolist(),
        valid_x_condition_counts=np.bincount(
            source["group_codes"][source["valid_x_ids"]],
            minlength=len(source["group_names"])).tolist(),
        fitting_reads_true_theta=False,
        objective="role-free elementwise RMS Hessian sparsity",
        warning="For CR3/CR5 p=0/1, NN H_UX exists but CPT analytic X derivative need not.")
    fingerprint=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()[:16]
    out=source["root"]/"rotations"/"hessian_sparsity"/args.condition/f"repeat{args.repeat:02d}"/f"{args.tag}_{args.mode}_{args.grouping}_{fingerprint}"
    if args.dry_run:
        print("SOURCE VALIDATION: PASS")
        print("Output:",out)
        print("MODEL",args.model,"DIM",dim,"FIT",len(source["fit_subject_ids"]),
              "VALID",len(source["valid_subject_ids"]))
        return out
    if out.exists():
        if (out/"fit_complete.json").exists():
            print("Preserved complete Hessian rotation:",out)
            return out
        raise FileExistsError(f"Incomplete output folder exists: {out}")
    print(f"{args.model}: frozen decoder UU/UX autodiff, mode={cfg['mode']}, group={cfg['grouping']}",flush=True)
    model,device,mu,sd=restore(source,cfg["device"])
    include_ux=cfg["mode"]!="uu_only"
    fit_fields=extract(model,device,source,source["fit_subject_ids"],source["fit_x_ids"],
                       mu,sd,include_ux=include_ux,batch_size=int(cfg["batch_subjects"]))
    val_fields=extract(model,device,source,source["valid_subject_ids"],source["valid_x_ids"],
                       mu,sd,include_ux=include_ux,batch_size=int(cfg["batch_subjects"]))
    fit_mom=summarize_fields(fit_fields,cfg["grouping"],include_ux)
    val_mom=summarize_fields(val_fields,cfg["grouping"],include_ux)
    loss_kwargs=dict(mode=cfg["mode"],uu_weight=float(cfg["uu_weight"]),
                     ux_weight=float(cfg["ux_weight"]),epsilon=float(cfg["epsilon"]))
    fit_loss=HessianLoss(fit_mom,**loss_kwargs)
    valid_loss=HessianLoss(val_mom,**loss_kwargs)
    R,results=optimize(fit_loss,valid_loss,cfg)
    inv=audit_invariance(model,device,source,R,mu,sd)
    U=source["q"]@R.T
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/"rotation.npz",
                        R=R,A=R@source["W"],U=U,
                        W=source["W"],B=source["B"],q=source["q"],
                        native_Z_mean=source["pca_mean"],
                        native_Z_residual=source["residual"],
                        train_ids=source["train_ids"],
                        validation_ids=source["validation_ids"],
                        test_ids=source["test_ids"],
                        fit_subject_ids=source["fit_subject_ids"],
                        valid_subject_ids=source["valid_subject_ids"],
                        fit_x_ids=source["fit_x_ids"],valid_x_ids=source["valid_x_ids"])
    for prefix,fields in (("fit",fit_fields),("validation",val_fields)):
        np.savez_compressed(out/f"{prefix}_fields.npz",
                            eta=fields["eta"],H_UU=fields["UU"],
                            H_UX=fields["UX"] if fields["UX"] is not None else np.empty((0,)),
                            subject_ids=fields["subject_ids"],
                            trial_ids=fields["x_ids"],
                            condition_codes=fields["group_codes"])
    plane_report=save_planes(out,R,fit_loss,valid_loss)
    as_json(out/"plane_diagnostics.json",plane_report)
    as_json(out/"protocol.json",meta)
    as_json(out/"loss_diagnostics.json",results)
    as_json(out/"prediction_invariance.json",inv)
    as_json(out/"fit_complete.json",dict(
        rotation_frozen=True,mode=cfg["mode"],grouping=cfg["grouping"],
        R=R.tolist(),oracle_theta_used_for_optimization=False,
        selected_restart_by="FIT Hessian simplicity ONLY",
        objective_fit=float(results["fit_loss"]["total"]),
        objective_VALID=float(results["validation_loss"]["total"]),
        status="candidate; TEST parameter recovery not used for selection",
        plane_symmetry_checked=True))
    print("Frozen Hessian-sparsity R:",out,flush=True)
    print("FIT loss",results["fit_loss"]["total"],"VALID",results["validation_loss"]["total"],
          "VALID vs PCA improvement",results["valid_improvement"],flush=True)
    if not args.no_evaluate:
        from rotations.hessian_sparsity.evaluation import evaluate
        scores=evaluate(source,U,out)
        print("TEST mean r^2",scores["hessian_sparsity"]["mean_matched_test_r2"],
              "PCA",scores["identity_pca"]["mean_matched_test_r2"],flush=True)
    return out


if __name__=="__main__":
    main()
