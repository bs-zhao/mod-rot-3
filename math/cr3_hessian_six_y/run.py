"""Six T-X-Y response channels -> identical data-only 10x10 Hessian estimators.

No neural-network Hessian / autodiff / neural gradient is calculated.
Frozen NN ONLY synthesizes Y for the same original T and fixed X.
Uses EXACT earlier math/cr3_hessian_expectation TensorPolynomial method.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT / "math" / "cr3_hessian_expectation"
sys.path.insert(0, str(PREVIOUS))
sys.path.insert(0, str(ROOT / "np" / "src"))

from data_only import TensorPolynomial
from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig, DiscreteNeuralProcess
from modlearn_rot.training.continuous import load_checkpoint

_spec = importlib.util.spec_from_file_location(
    "_previous_cr3_hessian_protocol", PREVIOUS / "run.py")
previous = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(previous)

CONDITION = "beta5em05_lambda0_ctx0p25to0p8_L4"
DEFAULT_DATA = ROOT / "outputs/choice/CR3/s1_generate/s1_generate__dataset.npz"
CHANNELS = (
    "original_choice", "original_logit", "original_probability",
    "nn_choiceprediction", "nn_logit", "nn_probability",
)
MATCHING_TRUTH = {
    "original_choice":"binary",
    "original_logit":"logit",
    "original_probability":"probability",
    "nn_choiceprediction":"binary",
    "nn_logit":"logit",
    "nn_probability":"probability",
}
VARIABLES = tuple(previous.NAMES)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def as_numpy_ids(obj):
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
    return np.asarray(obj, dtype=np.int64).reshape(-1)


def load_sources(dataset, checkpoint, latent):
    """Return the immutable original dataset, all-trial S5 Z and checkpoint."""
    if not Path(dataset).is_file():
        raise FileNotFoundError(f"CR3 S1 dataset not found: {dataset}")
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"S3 checkpoint not found: {checkpoint}")
    if not Path(latent).is_file():
        raise FileNotFoundError(
            f"All-trial S5 Z not found: {latent}\n"
            "Run: python np/s5_extract_z.py --model CR3 --beta 0.00005 "
            "--lambda-sparse 0 --latent-dim 4 --context-min 0.25 "
            "--context-max 0.8 --repeat 1"
        )
    T, X, Y, p = previous.load_cr3(dataset)
    checkpoint_info = load_checkpoint(checkpoint)
    file_hash = sha256(dataset)
    ck_hash = sha256(checkpoint)
    if checkpoint_info.get("extra_state", {}).get("dataset_sha256") != file_hash:
        raise RuntimeError("Checkpoint dataset SHA256 differs from current S1 file")
    cfg = checkpoint_info["model_config"]
    if (int(cfg["latent_dim"]), int(cfg["x_dim"]), int(cfg["num_choices"])) != (4,6,2):
        raise RuntimeError(f"Expected CR3 L4 binary decoder; got {cfg}")
    with np.load(latent, allow_pickle=False) as data:
        required = {"z", "checkpoint_sha256", "dataset_sha256", "context_mode",
                    "train_ids", "validation_ids", "test_ids"}
        if not required.issubset(data.files):
            raise RuntimeError(f"S5 NPZ missing fields: {sorted(required-set(data.files))}")
        if str(data["checkpoint_sha256"]) != ck_hash:
            raise RuntimeError("S5 Z was inferred from a DIFFERENT checkpoint")
        if str(data["dataset_sha256"]) != file_hash:
            raise RuntimeError("S5 Z belongs to a DIFFERENT dataset")
        if str(data["context_mode"]) != "all_observed_trials":
            raise RuntimeError("S5 Z is NOT inferred from all observed 500 trials")
        if "context_trials_per_subject" in data.files:
            if not np.all(np.asarray(data["context_trials_per_subject"]) == 500):
                raise RuntimeError("Not every subject has 500 context trials")
        Z = np.asarray(data["z"], dtype=np.float64)
        s5_splits={name:np.asarray(data[f"{name}_ids"],np.int64)
                   for name in ("train", "validation", "test")}
    if Z.shape != T.shape or not np.isfinite(Z).all():
        raise ValueError(f"Expected finite Z same shape as T (N,4); got {Z.shape}")
    ck_splits={name:as_numpy_ids(checkpoint_info["subject_splits"][name])
               for name in ("train", "validation", "test")}
    for name in ck_splits:
        if not np.array_equal(s5_splits[name], ck_splits[name]):
            raise RuntimeError(f"S5 split {name} disagrees with checkpoint")
    all_ids=np.concatenate(list(ck_splits.values()))
    if not np.array_equal(np.sort(all_ids),np.arange(len(T))):
        raise RuntimeError("Checkpoint splits are not disjoint/exhaustive")
    return T,X,Y,p,Z,checkpoint_info,ck_splits,dict(
        dataset_sha256=file_hash,checkpoint_sha256=ck_hash,latent_sha256=sha256(latent))


def fit_affine_map(Z,T,train,include_intercept=True,max_condition=1e10):
    """Train-only OLS T = Z @ B + b, with full-rank invertible B.

    In column-vector convention A=B.T, so T_col=A @ Z_col+b_col.
    """
    Z,T=np.asarray(Z,np.float64),np.asarray(T,np.float64)
    train=np.asarray(train,np.int64)
    if Z.ndim!=2 or Z.shape!=T.shape or Z.shape[1]!=4:
        raise ValueError("Z and T must have matching shape (N,4)")
    if len(train)<6:
        raise ValueError("Insufficient training subjects")
    if include_intercept:
        reg=np.column_stack((Z[train],np.ones(len(train))))
    else:
        reg=Z[train]
    coef,*_ = np.linalg.lstsq(reg,T[train],rcond=None)
    B=coef[:4,:]
    b=np.asarray(coef[4,:] if include_intercept else np.zeros(4),np.float64)
    sing=np.linalg.svd(B,compute_uv=False)
    if sing[-1] <= 0 or sing[0]/sing[-1] > max_condition:
        raise RuntimeError(
            "OLS Z->T map is singular or numerically unstable; "
            f"singular values={sing.tolist()}, condition={sing[0]/max(sing[-1],1e-300):.3g}. "
            "Cannot reliably invert. Inspect S5 latent Z; do NOT silently use pseudoinverse."
        )
    B_inverse=np.linalg.inv(B)
    mapped=(T-b)@B_inverse
    t_fit=Z@B+b
    if not np.allclose(mapped@B+b,T,rtol=5e-9,atol=5e-9):
        raise RuntimeError("Inverse affine mapping is inconsistent")
    return dict(B=B,A=B.T,b=b,B_inverse=B_inverse,mapped_Z=mapped,
                predicted_T_from_original_Z=t_fit,
                singular_values=sing,condition_number=float(sing[0]/sing[-1]))


def regression_diagnostics(mapping,Z,T,splits):
    reconstructed=mapping["predicted_T_from_original_Z"]
    inverseZ=mapping["mapped_Z"]
    out={
        "definition":"A @ z + b approximates theta (column convention); "
                     "Trow=Zrow @ A.T + b. Inverse mapped Zrow=(Trow-b) @ inv(A.T)",
        "A":mapping["A"],"b":mapping["b"],
        "singular_values":mapping["singular_values"],
        "condition_number":mapping["condition_number"],
        "note":"The affine mapping is fit ONLY on original S3 TRAIN subjects; "
               "true T is used to create counterfactual decoder Z for EVERY subject.",
        "groups":{}
    }
    for name,ids in (("all",np.arange(len(T))),*splits.items()):
        yt=T[ids];yp=reconstructed[ids];zp=inverseZ[ids];z=Z[ids]
        r2=[];corr=[]
        for k in range(4):
            sst=np.sum((yt[:,k]-yt[:,k].mean())**2)
            ssr=np.sum((yp[:,k]-yt[:,k])**2)
            r2.append(float(1-ssr/sst) if sst>1e-18 else None)
            if np.std(yt[:,k])>1e-12 and np.std(yp[:,k])>1e-12:
                corr.append(float(np.corrcoef(yt[:,k],yp[:,k])[0,1]))
            else:corr.append(None)
        out["groups"][name]=dict(n=int(len(ids)),R2_per_theta=r2,
           pearson_per_theta=corr,
           T_prediction_mae_per_dim=np.mean(np.abs(yp-yt),axis=0),
           T_prediction_rmse_per_dim=np.sqrt(np.mean((yp-yt)**2,axis=0)),
           Z_inverse_vs_encoder_mean_rmse_per_dim=
               np.sqrt(np.mean((zp-z)**2,axis=0)),
           Z_inverse_vs_encoder_mean_mae_per_dim=
               np.mean(np.abs(zp-z),axis=0))
    return out


@torch.inference_mode()
def decoder_responses(checkpoint_info,X,Z_map,batch_size=32,device="auto"):
    """Neural network ONLY produces three numeric Y arrays; no derivatives."""
    where=("cuda" if torch.cuda.is_available() else "cpu") if device=="auto" else device
    device=torch.device(where)
    model=DiscreteNeuralProcess(DiscreteNPConfig(**checkpoint_info["model_config"]))
    model.load_state_dict(checkpoint_info["model_state_dict"],strict=True)
    model.to(device).eval()
    x=torch.tensor(X,dtype=torch.float32,device=device).unsqueeze(0)
    n=len(Z_map);m=len(X)
    logits=np.empty((n,m),dtype=np.float64)
    probabilities=np.empty((n,m),dtype=np.float64)
    hard_choices=np.empty((n,m),dtype=np.float64)
    for start in range(0,n,batch_size):
        end=min(n,start+batch_size)
        z=torch.tensor(Z_map[start:end],dtype=torch.float32,device=device)
        xb=x.expand(end-start,-1,-1)
        # Decoder only, no encoder, no torch.autograd / Hessian / gradient.
        two_logits=model.decode(xb,z)
        odds=two_logits[...,1]-two_logits[...,0]
        prob=torch.softmax(two_logits,dim=-1)[...,1]
        hard=(prob >= .5).to(torch.float32)
        if not torch.isfinite(odds).all():
            raise RuntimeError(f"Nonfinite neural logit in subjects {start}:{end}")
        logits[start:end]=odds.cpu().numpy()
        probabilities[start:end]=prob.cpu().numpy()
        hard_choices[start:end]=hard.cpu().numpy()
        if start==0 or end==n or (start//batch_size)%25==0:
            print(f"Decoder Y: {end}/{n} subjects",flush=True)
    max_consistency=float(np.max(np.abs(expit(logits)-probabilities)))
    if max_consistency>2e-6:
        raise RuntimeError(f"NN logit/probability inconsistency: {max_consistency}")
    return dict(nn_choiceprediction=hard_choices,nn_logit=logits,
                nn_probability=probabilities),max_consistency


def pair_metrics(left,right):
    a=np.asarray(left,np.float64).reshape(-1)
    b=np.asarray(right,np.float64).reshape(-1)
    d=b-a
    x=a-a.mean();y=b-b.mean()
    r=float(x@y/np.sqrt((x@x)*(y@y))) if x@x>1e-20 and y@y>1e-20 else None
    return dict(n=len(a),pearson_r=r,mae=float(np.mean(abs(d))),
                rmse=float(np.sqrt(np.mean(d**2))),
                bias=float(np.mean(d)),left_mean=float(a.mean()),
                right_mean=float(b.mean()))


def write_matrix(path,mat):
    previous.write_matrix(path,np.asarray(mat,float))


def save_outputs(out,summary,arrays):
    with (out/"summary.json").open("w",encoding="utf8") as stream:
        json.dump(previous.to_builtin(summary),stream,indent=2,allow_nan=False)
    np.savez_compressed(out/"all_matrices.npz",**arrays)


def plot_comparison(out,channel,ref,estimate,block):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sub={"TT":(slice(None,4),slice(None,4)),
         "TX":(slice(None,4),slice(4,None)),
         "XX":(slice(4,None),slice(4,None))}[block]
    a=ref[sub];b=estimate[sub];c=b-a
    finite=np.concatenate([np.abs(x[np.isfinite(x)]).ravel() for x in (a,b,c)])
    limit=max(1e-12,float(finite.max(initial=1e-12)))
    fig,ax=plt.subplots(1,3,figsize=(14,4.3),constrained_layout=True)
    for p,values,title in zip(ax,(a,b,c),("true CR3","estimated via TXY","difference")):
        im=p.imshow(np.ma.masked_invalid(values),vmin=-limit,vmax=limit,cmap="coolwarm")
        p.set_title(f"{channel}: {title}")
        p.set_yticks(np.arange(values.shape[0]),VARIABLES[sub[0]],fontsize=8)
        p.set_xticks(np.arange(values.shape[1]),VARIABLES[sub[1]],
                     rotation=48,ha="right",fontsize=8)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i,j]):
                    p.text(j,i,f"{values[i,j]:.2g}",fontsize=7,ha="center",va="center")
    fig.colorbar(im,ax=ax,shrink=.7)
    fig.savefig(out/f"{channel}__{block}__comparison.png",dpi=145)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition",default=CONDITION)
    p.add_argument("--repeat",type=int,default=1)
    p.add_argument("--data",type=Path,default=DEFAULT_DATA)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--latent",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--degree-t",type=int,default=4)
    p.add_argument("--degree-x",type=int,default=3)
    p.add_argument("--ridge-t",type=float,default=.01)
    p.add_argument("--ridge-x",type=float,default=.01)
    p.add_argument("--quad-nodes",type=int,default=10)
    p.add_argument("--seed",type=int,default=20261106)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--max-condition",type=float,default=1e10)
    p.add_argument("--no-intercept",action="store_true")
    p.add_argument("--sample-only",action="store_true",
                   help="Skip prior quadrature, compare only finite-subject averages")
    p.add_argument("--no-plots",action="store_true")
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    args=p.parse_args()
    if args.repeat<1 or args.batch_size<1 or args.degree_t<2 or args.degree_x<2 or args.quad_nodes<2:
        p.error("Invalid command-line options")
    stem=ROOT/"outputs/choice/CR3"
    if args.checkpoint is None:
        args.checkpoint=stem/"s3_train"/args.condition/f"repeat{args.repeat:02d}"/"s3_train__checkpoint.pt"
    if args.latent is None:
        args.latent=stem/"s4_evaluate"/args.condition/f"repeat{args.repeat:02d}"/"s4_evaluate__latent.npz"
    if args.output is None:
        args.output=ROOT/"outputs/math/cr3_hessian_six_y"/args.condition/f"repeat{args.repeat:02d}"
    args.output.mkdir(parents=True,exist_ok=True)
    T,X,Yraw,p_true,Z,ck,splits,hashes=load_sources(
        args.data,args.checkpoint,args.latent)
    print(f"CR3 original T/X/Y: {len(T)} subjects x {len(X)} fixed trials; "
          f"Z shape={Z.shape}",flush=True)
    mapping=fit_affine_map(Z,T,splits["train"],
        include_intercept=not args.no_intercept,max_condition=args.max_condition)
    map_info=regression_diagnostics(mapping,Z,T,splits)
    print("Z -> T OLS: condition number:",mapping["condition_number"],flush=True)
    print("Z -> T R2 TEST per theta:",
          map_info["groups"]["test"]["R2_per_theta"],flush=True)
    nn_channels,nn_consistency=decoder_responses(
        ck,X,mapping["mapped_Z"],batch_size=args.batch_size,device=args.device)
    from oracle import oracle_outputs
    original_logit,recomputed_p=oracle_outputs(T,X)
    discrepancy=float(np.max(np.abs(recomputed_p-p_true)))
    if not np.allclose(recomputed_p,p_true,rtol=1e-12,atol=1e-12):
        raise RuntimeError(
            f"Original generator true probability does not match S1: {discrepancy}")
    responses=dict(
        original_choice=Yraw,
        original_logit=original_logit,
        original_probability=p_true,
        **nn_channels
    )
    checks={
        "nn_logit_vs_original_logit":pair_metrics(original_logit,nn_channels["nn_logit"]),
        "nn_probability_vs_original_probability":pair_metrics(p_true,nn_channels["nn_probability"]),
        "nn_hard_class_vs_original_binary_choice":pair_metrics(Yraw,nn_channels["nn_choiceprediction"]),
        "nn_hard_class_vs_original_true_probability":pair_metrics(p_true,nn_channels["nn_choiceprediction"]),
        "max_abs_original_probability_reconstruction":discrepancy,
        "max_abs_nn_sigmoid_logit_vs_probability":nn_consistency,
    }
    np.savez_compressed(args.output/"six_response_channels.npz",
        T=T,X=X,Z_original_all_trial_posterior_mean=Z,
        Z_inverse_mapped_from_true_T=mapping["mapped_Z"],
        A_z_to_t=mapping["A"],b_z_to_t=mapping["b"],**responses)
    print("Fitting the IDENTICAL previous 10D T-X-Y TensorPolynomial on six channels...",flush=True)
    fit=TensorPolynomial(T,X,args.degree_t,args.degree_x,
                        args.ridge_t,args.ridge_x,args.seed)
    cof={}
    holdout={}
    for ch in CHANNELS:
        print(" T-X-Y fit:",ch,flush=True)
        cof[ch]=fit.fit(responses[ch])
        holdout[ch]=fit.holdout_metrics(responses[ch],cof[ch])

    # The same original CR3 oracle for every response, including NN-derived Y.
    # For NN-derived channels, this measures surrogate discrepancy TO ORIGINAL
    # CR3, not error versus the NN's own Hessian.
    evals=[("sample",T,np.full(len(T),1/len(T)))]
    if not args.sample_only:
        priorT,priorW,mass_errors=previous.prior_quadrature(args.quad_nodes)
        evals.append(("prior",priorT,priorW))
    else:
        mass_errors=None
    report={
      "definition":"Six Y sources, same original four true T + same six fixed X; "
                   "same DATA-ONLY 10x10 TensorPolynomial Hessian estimation.",
      "no_nn_hessian":True,
      "nn_only_synthesizes_y":True,
      "sources":{"dataset":str(args.data),"checkpoint":str(args.checkpoint),
                 "all_trial_latent":str(args.latent),**hashes},
      "mapping":map_info,
      "response_sources":{
          "original_choice":"S1 actual Bernoulli choices",
          "original_logit":"Original CR3 generator logit reconstructed and verified vs S1 saved p",
          "original_probability":"S1 saved true p",
          "nn_choiceprediction":"hard argmax (p_nn >= 0.5) decoder class prediction",
          "nn_logit":"decoder two-class logit difference l1-l0 at Z=A^-1(T-b), X",
          "nn_probability":"decoder softmax choice-1 probability at Z=A^-1(T-b), X",
      },
      "interpretation":{
          "reference":"Exact ORIGINAL CR3 logit / probability analytic Hessian, not NN derivatives",
          "original_choice":"Bernoulli conditional-mean Hessian equals true p Hessian",
          "nn_choiceprediction":"Hard 0/1 classifier is not twice differentiable "
              "at its decision boundary (zero elsewhere). The CR3 p Hessian "
              "is only a common external COMPARATOR, NOT its own true Hessian.",
          "nn_logit_and_probability":"Discrepancy to original CR3 conflates "
              "decoder surrogate mismatch and TXY estimation error.",
          "x_derivatives":"Zero amount derivatives of CR3 may be undefined. "
              "The previous estimator reports conditional means over valid X.",
          "no_data_leakage_mapping":"Z->T affine mapping is trained only on S3 train subjects; "
                                    "sieve has its own subject-level train/test split.",
          "encoder":"All-trial S5 posterior means are used ONLY to fit the Z->T map; "
                    "the frozen decoder gets Z inversely mapped from TRUE T, not S5 Z.",
      },
      "dataset_shape":{"T":list(T.shape),"X":list(X.shape),
                       "Y":list(Yraw.shape)},
      "regression":{
          "method":"exact previous TensorPolynomial T-X polynomial sieve",
          "degree_t":args.degree_t,"degree_x":args.degree_x,
          "ridge_t":args.ridge_t,"ridge_x":args.ridge_x,
          "seed":args.seed,"train_fraction":0.8,
          "train_subject_count":len(fit.train_idx),
          "test_subject_count":len(fit.test_idx),
      },
      "nn_y_vs_original_y":checks,
      "holdout":holdout,
      "expectation":{},
    }
    arrays={"A_z_to_t":mapping["A"],"b_z_to_t":mapping["b"],
            "Z_inverse_mapped_from_true_T":mapping["mapped_Z"]}
    for name,t_eval,w_eval in evals:
        print("Calculating original analytic CR3 10D Hessian:",name,flush=True)
        ref=previous.oracle_moments(t_eval,X,w_eval,batch_size=args.batch_size)
        report["expectation"][name]={}
        for ch in CHANNELS:
            analytic=ref[MATCHING_TRUTH[ch]]
            print(f" Data-only Hessian {name}/{ch}",flush=True)
            estimated=fit.expected_hessian(cof[ch],t_eval,w_eval)
            entry=previous.comparison(analytic,estimated)
            entry["note"]=("T-X-Y estimate compared to ORIGINAL CR3 Hessian. "
                           "For NN outputs, deviation includes changed generating "
                           "function; it is not a pure Hessian-estimation error.")
            report["expectation"][name][ch]=entry
            for metric in ("conditional_mean","strict_mean","conditional_rms",
                           "conditional_std","fraction_defined"):
                arrays[f"{name}__{ch}__estimated__{metric}"]=estimated[metric]
            # Save analytic only once per class; binary and p are same truth.
            label=MATCHING_TRUTH[ch]
            if f"{name}__oracle__{label}__conditional_mean" not in arrays:
                for metric in ("conditional_mean","strict_mean","conditional_rms",
                               "conditional_std","fraction_defined"):
                    arrays[f"{name}__oracle__{label}__{metric}"]=analytic[metric]
            for block,r in entry.items():
                if isinstance(r,dict):
                    print(f"  {block}: MAE={r['mae']:.6g} "
                          f"RMSE={r['rmse']:.6g}",flush=True)
            write_matrix(args.output/f"{name}__{ch}__estimated_mean.csv",
                         estimated["conditional_mean"])
            if name=="sample" and not args.no_plots:
                for block in ("TT","TX","XX"):
                    plot_comparison(
                        args.output,ch,analytic["conditional_mean"],
                        estimated["conditional_mean"],block)
        # Direct comparison of estimates fitted from original vs NN Y
        pairs={"choice":("original_choice","nn_choiceprediction"),
               "logit":("original_logit","nn_logit"),
               "probability":("original_probability","nn_probability")}
        report["expectation"][name]["nn_minus_original_fitted"]={}
        for k,(left,right) in pairs.items():
            estimate_l=arrays[f"{name}__{left}__estimated__conditional_mean"]
            estimate_r=arrays[f"{name}__{right}__estimated__conditional_mean"]
            difference=estimate_r-estimate_l
            arrays[f"{name}__difference_nn_minus_original__{k}"]=difference
            report["expectation"][name]["nn_minus_original_fitted"][k]={
                "theta_theta_mae":float(np.mean(np.abs(difference[:4,:4]))),
                "theta_x_mae":float(np.mean(np.abs(difference[:4,4:]))),
                "x_x_mae":float(np.mean(np.abs(difference[4:,4:]))),
            }
    save_outputs(args.output,report,arrays)
    print(f"Done. Saved six-channel TXY Hessian comparison: {args.output}",flush=True)


if __name__=="__main__":
    main()
