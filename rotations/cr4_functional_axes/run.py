"""Independent CR3/CR4/CR5 output-gain + low-curvature rotation.

No imports from another rotation suite; no true parameter is used to fit R.
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

from rotations.cr4_functional_axes.data import load_source,sha256
from rotations.cr4_functional_axes.teacher import restore,logodds
from rotations.cr4_functional_axes.fields import extract
from rotations.cr4_functional_axes.objective import fit_functional_axes
from rotations.cr4_functional_axes.ux_audit import ux_field,ux_diagnostics

HERE=Path(__file__).resolve().parent


def read_config(args):
    config=json.loads((HERE/"config.json").read_text(encoding="utf-8"))
    config["seed"]=args.seed
    config["teacher_device"]=args.device
    for s in args.set:
        k,sep,v=s.partition("=")
        if not sep or k not in config:
            raise ValueError(f"Unknown config field: {k}. Keys: {sorted(config)}")
        try:config[k]=json.loads(v)
        except json.JSONDecodeError:
            if isinstance(config[k],str):config[k]=v
            else:raise ValueError(f"Expected JSON literal for {k}: {v}")
    return config


def code_sha():
    h=hashlib.sha256()
    for p in sorted(HERE.glob("*.py")):
        h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()


def dump_json(p, obj):
    def normal(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False,default=normal),encoding="utf-8")


def save_landscape(out,landscape):
    with (out/"curvature_landscape.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["angle_deg","FIT_curvature","VALID_curvature"])
        w.writerows(zip(landscape["angle_deg"],landscape["fit_curvature"],
                        landscape["validation_curvature"]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.plot(landscape["angle_deg"],landscape["fit_curvature"],label="FIT")
    ax.plot(landscape["angle_deg"],landscape["validation_curvature"],label="VALID")
    ax.set(xlabel="Angle within output-gain orthogonal plane (degrees)",
           ylabel="Normalized directional squared Hessian",
           title="Low-curvature search; not a theta-based scan")
    ax.legend();fig.tight_layout()
    fig.savefig(out/"curvature_landscape.png",dpi=160);plt.close(fig)


def roundtrip_audit(model,device,source,R,x_mu,x_sd):
    q=source["q"]
    B=source["B"]
    reconstructed=(source["pca_mean"]
                   + ((q@R.T)@R)@B.T + source["residual"])
    z_err=float(np.max(np.abs(reconstructed-source["z"])))
    if z_err>1e-8:
        raise RuntimeError("Rotated U did not reconstruct full native Z")
    ids=source["fit_subject_ids"][:min(4,len(source["fit_subject_ids"]))]
    bank=(source["bank"][:12]-x_mu)/x_sd
    xt=torch.as_tensor(bank,dtype=torch.float32,device=device).unsqueeze(0).expand(len(ids),-1,-1)
    with torch.no_grad():
        a=logodds(model,torch.as_tensor(source["z"][ids],dtype=torch.float32,device=device),xt)
        b=logodds(model,torch.as_tensor(reconstructed[ids],dtype=torch.float32,device=device),xt)
        p_err=float(torch.max(torch.abs(torch.sigmoid(a)-torch.sigmoid(b))).cpu())
    return dict(z_max_abs=z_err,probability_max_abs=p_err,
                residual_preserved=True,rotation_orthogonal=True)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",choices=("CR3","CR4","CR5"),default="CR4")
    ap.add_argument("--condition",default="beta0p0005_lambda0_ctx0p25to0p8_L3")
    ap.add_argument("--repeat",type=int,default=1)
    ap.add_argument("--rotation-dim",type=int,default=3)
    ap.add_argument("--seed",type=int,default=31)
    ap.add_argument("--device",default="auto")
    ap.add_argument("--tag",default="main")
    ap.add_argument("--set",action="append",default=[],metavar="KEY=VALUE")
    ap.add_argument("--dry-run",action="store_true")
    ap.add_argument("--no-evaluate",action="store_true")
    args=ap.parse_args(argv)
    if args.repeat<1 or args.seed<0:
        ap.error("repeat>=1 and seed>=0 are required")
    if args.rotation_dim not in ((3,4) if args.model=="CR3" else (3,)):
        ap.error("CR3 supports --rotation-dim 3 (partial) or 4 (complete); CR4/CR5 require 3")
    if args.model=="CR3" and not args.condition.endswith("_xstd"):
        args.condition += "_xstd"
        print(f"CR3 automatically uses new X-standardized S3/S5 condition: {args.condition}",flush=True)
    if args.model=="CR3" and args.rotation_dim==3:
        print("WARNING: CR3 has FOUR true parameters; L3 can only give a partial 3-of-4 axis comparison. Use L4 + --rotation-dim 4 for all four.",flush=True)
    config=read_config(args)
    if min(int(config[k]) for k in ("fit_subjects","valid_subjects","fit_x",
                                    "valid_x","batch_subjects"))<1:
        raise ValueError("Sampling and batch sizes must be positive")
    source=load_source(args,config)
    meta=dict(suite=HERE.name,model=args.model,condition=args.condition,repeat=args.repeat,
              rotation_dim=args.rotation_dim,tag=args.tag,source_hashes=source["hashes"],
              true_parameter_count=(4 if args.model=="CR3" else 3),
              full_parameter_axis_recovery_possible=(args.model!="CR3" or args.rotation_dim==4),
              configuration=config,code_sha256=code_sha(),
              fit_subject_ids=source["fit_subject_ids"].tolist(),
              validation_subject_ids=source["valid_subject_ids"].tolist(),
              fit_x_ids=source["fit_x_ids"].tolist(),
              validation_x_ids=source["valid_x_ids"].tolist(),
              gate_policy="Both FIT/VALID gain+curvature thresholds; never use theta to select")
    fingerprint=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()[:16]
    out=(source["root"]/"rotations"/HERE.name/args.condition/
         f"repeat{args.repeat:02d}"/f"{args.tag}_{fingerprint}")
    if args.dry_run:
        print(f"Planned independent {args.model} rotation:",out)
        print("FIT subjects",len(source["fit_subject_ids"]),
              "VALID subjects",len(source["valid_subject_ids"]),
              "FIT X",len(source["fit_x_ids"]),"VALID X",len(source["valid_x_ids"]))
        return out
    if out.exists():
        if (out/"fit_complete.json").is_file():
            print("Preserved completed rotation:",out)
            return out
        raise FileExistsError(f"Incomplete rotation folder; inspect before retry: {out}")
    print(f"{args.model} functional-axes: extracting frozen-decoder derivative fields",flush=True)
    model,device,x_mu,x_sd=restore(source,config["teacher_device"])
    fit=extract(model,device,source,source["fit_subject_ids"],source["fit_x_ids"],
                x_mu,x_sd,int(config["batch_subjects"]))
    valid=extract(model,device,source,source["valid_subject_ids"],source["valid_x_ids"],
                  x_mu,x_sd,int(config["batch_subjects"]))
    if args.rotation_dim==4:
        from rotations.cr4_functional_axes.objective_cr3_4d import fit_four_axes
        R,diagnostics,landscape=fit_four_axes(fit,valid,config)
    else:
        R,diagnostics,landscape=fit_functional_axes(fit,valid,config)
    if not np.allclose(R@R.T,np.eye(args.rotation_dim),atol=1e-8) or not np.isclose(np.linalg.det(R),1.,atol=1e-8):
        raise RuntimeError("Rotation not in SO(k)")
    U=source["q"]@R.T
    audit=roundtrip_audit(model,device,source,R,x_mu,x_sd)
    if audit["probability_max_abs"]>5e-6:
        raise RuntimeError("Rotation changed frozen decoder predictions")
    # Distinct UX diagnostic: never feed it into the optimizer or gate.
    ns=min(int(config.get("ux_subjects",24)),len(source["fit_subject_ids"]))
    nx=min(int(config.get("ux_x",48)),len(source["fit_x_ids"]))
    mixed=ux_field(model,device,source,source["fit_subject_ids"][:ns],
                   source["fit_x_ids"][:nx],x_mu,x_sd,int(config["batch_subjects"]))
    mixed_stats=ux_diagnostics(mixed,R)
    out.mkdir(parents=True)
    np.savez_compressed(out/"rotation.npz",R=R,A=R@source["W"],
                        A_inverse=np.linalg.pinv(R@source["W"]),
                        U=U,q=source["q"],W=source["W"],B=source["B"],
                        z_mean=source["pca_mean"],z_residual=source["residual"],
                        train_ids=source["tr"],validation_ids=source["va"],test_ids=source["te"],
                        fit_subject_ids=source["fit_subject_ids"],
                        fit_x_ids=source["fit_x_ids"])
    np.savez_compressed(out/"fit_fields.npz",eta=fit["eta"],G=fit["G"],H=fit["H"],
                        subject_ids=fit["subject_ids"],x_ids=fit["x_ids"])
    np.savez_compressed(out/"validation_fields.npz",eta=valid["eta"],G=valid["G"],H=valid["H"],
                        subject_ids=valid["subject_ids"],x_ids=valid["x_ids"])
    np.savez_compressed(out/"ux_audit.npz",ux_original_q=mixed,ux_rotated_u=np.einsum(
        "ia,sxaj->sxij",R,mixed,optimize=True))
    dump_json(out/"protocol.json",meta)
    dump_json(out/"loss_diagnostics.json",diagnostics)
    dump_json(out/"ux_diagnostics.json",mixed_stats)
    dump_json(out/"invariance_audit.json",audit)
    save_landscape(out,landscape)
    dump_json(out/"fit_complete.json",dict(
        R_frozen=True,R=R.tolist(),
        result="functional candidate; see gate status",
        gate_pass=diagnostics["both_gates_pass"],
        theta_unavailable_during_fit=True,derivative_source="frozen_decoder_autodiff",
        outputs=["rotation.npz","protocol.json","loss_diagnostics.json",
                 "invariance_audit.json","ux_diagnostics.json"]))
    print(f"Frozen {args.model} functional rotation:",out,flush=True)
    print("FIT/VALID gates:",diagnostics["both_gates_pass"],flush=True)
    if not args.no_evaluate:
        from rotations.cr4_functional_axes.evaluation import parameter_diagnostics
        scores=parameter_diagnostics(source,U,R,out)
        print("TEST r2 candidate:",scores["candidate"]["mean_test_r2"],flush=True)
        print("TEST r2 identity:",scores["identity_pca"]["mean_test_r2"],flush=True)
    return out


if __name__=="__main__":
    main()
