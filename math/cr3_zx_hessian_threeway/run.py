"""Three-way CR3 Hessian experiment in ORIGINAL encoder Z and fixed X.

Run from repository root:
    python math/cr3_zx_hessian_threeway/run.py

Three parallel Hessian paths, all in W=(Z,X) coordinates:
  (1) fit same data-only matrix/TensorPolynomial estimator on oracle Y(AZ+b,X)
  (2) fit same data-only estimator on frozen decoder Y(Z,X)
  (3) analytic CR3 Hessian evaluated at T_hat=AZ+b, then pulled back to Z,X.

NO network derivative/autodiff is computed; the NN only generates numeric Y.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"np"/"src"))
from data_only import TensorPolynomial
from oracle import oracle_outputs,oracle_hessians
from reference_protocol import HessianMoments, load_cr3, comparison, to_builtin, write_matrix, NAMES
from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig,DiscreteNeuralProcess
from modlearn_rot.training.continuous import load_checkpoint

CONDITION="beta5em05_lambda0_ctx0p25to0p8_L4"
STEM=ROOT/"outputs"/"choice"/"CR3"
NAMES_ZX=("z1","z2","z3","z4")+tuple(NAMES[4:])
CHANNELS=("oracle_choice","oracle_logit","oracle_probability",
          "nn_choice","nn_logit","nn_probability")
PAIRING={"choice":("oracle_choice","nn_choice"),
         "logit":("oracle_logit","nn_logit"),
         "probability":("oracle_probability","nn_probability")}


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def numpy_ids(v):
    if isinstance(v,torch.Tensor):
        return v.detach().cpu().numpy().astype(np.int64)
    return np.asarray(v,dtype=np.int64)


def load_sources(data_path,checkpoint_path,latent_path):
    T,X,y_obs,p_obs=load_cr3(data_path)
    ck=load_checkpoint(checkpoint_path)
    data_hash=sha256(data_path)
    ck_hash=sha256(checkpoint_path)
    if ck.get("extra_state",{}).get("dataset_sha256")!=data_hash:
        raise RuntimeError("S3 checkpoint was trained on a different S1 dataset")
    config=ck["model_config"]
    if tuple(int(config[v]) for v in ("latent_dim","x_dim","num_choices"))!=(4,6,2):
        raise ValueError("Expected CR3 L4 / X6 / two-class checkpoint")
    with np.load(latent_path,allow_pickle=False) as arr:
        keys={"z","checkpoint_sha256","dataset_sha256","context_mode",
              "train_ids","validation_ids","test_ids"}
        if not keys.issubset(arr.files):
            raise ValueError("S5 all-trial latent file missing fields: "+str(keys-set(arr.files)))
        if str(arr["checkpoint_sha256"])!=ck_hash or str(arr["dataset_sha256"])!=data_hash:
            raise ValueError("S5 latent checkpoint/dataset SHA256 mismatch")
        if str(arr["context_mode"])!="all_observed_trials":
            raise ValueError("S5 latent Z must use ALL observed trials")
        if "context_trials_per_subject" in arr.files:
            if not np.all(np.asarray(arr["context_trials_per_subject"])==500):
                raise ValueError("S5 latent was not inferred from all 500 trials")
        Z=np.asarray(arr["z"],dtype=np.float64)
        splits={key:np.asarray(arr[key+"_ids"],dtype=np.int64)
                for key in ("train","validation","test")}
    if Z.shape!=T.shape or not np.isfinite(Z).all():
        raise ValueError(f"Z must be finite (subjects,4), got {Z.shape}")
    for name,ids in splits.items():
        from_ck=numpy_ids(ck["subject_splits"][name]).ravel()
        if not np.array_equal(ids,from_ck):
            raise ValueError("Subject split mismatch between S5 and checkpoint: "+name)
    if not np.array_equal(np.sort(np.concatenate(list(splits.values()))),np.arange(len(T))):
        raise ValueError("Split IDs are not disjoint or exhaustive")
    provenance=dict(dataset=str(data_path),checkpoint=str(checkpoint_path),
                    all_trial_latent=str(latent_path),
                    data_sha256=data_hash,checkpoint_sha256=ck_hash,
                    latent_sha256=sha256(latent_path))
    return T,X,Z,ck,splits,provenance


def fit_affine(Z,T,train_ids,*,intercept=True,max_condition=1e10):
    """OLS: column convention T_hat = A Z + b, evaluated for ALL subjects."""
    Z=np.asarray(Z,dtype=np.float64)
    T=np.asarray(T,dtype=np.float64)
    if T.shape!=Z.shape or Z.ndim!=2 or Z.shape[1]!=4:
        raise ValueError("T,Z must both have shape (N,4)")
    ids=np.asarray(train_ids,dtype=np.int64)
    if len(ids)<6:raise ValueError("Not enough subjects for affine regression")
    features=np.concatenate((Z[ids],np.ones((len(ids),1))),axis=1) if intercept else Z[ids]
    coefficients,*_=np.linalg.lstsq(features,T[ids],rcond=None)
    B=coefficients[:4,:]
    b=coefficients[4,:] if intercept else np.zeros(4)
    singular=np.linalg.svd(B,compute_uv=False)
    condition=float(singular[0]/max(singular[-1],1e-300))
    if singular[-1]<=0 or condition>max_condition:
        raise ValueError(f"Z->T affine A is singular or ill-conditioned: {singular}, kappa={condition}")
    A=B.T
    T_hat=Z@A.T+b
    return A,b,T_hat,singular,condition


def mapping_metrics(Z,T,T_hat,ids):
    rows={}
    for name,sel in ids.items():
        sel=np.asarray(sel,dtype=int)
        residual=T_hat[sel]-T[sel]
        std=np.std(T[sel],axis=0)
        sst=np.mean((T[sel]-T[sel].mean(0))**2,axis=0)
        mse=np.mean(residual**2,axis=0)
        out=np.divide(mse,sst,out=np.full_like(mse,np.nan),where=sst>1e-18)
        rows[name]=dict(n_subjects=int(len(sel)),
                        per_theta_R2=np.where(np.isfinite(out),1.-out,np.nan),
                        per_theta_RMSE=np.sqrt(mse),per_theta_MAE=np.mean(abs(residual),axis=0),
                        min_T_hat=T_hat[sel].min(axis=0),
                        max_T_hat=T_hat[sel].max(axis=0))
    return rows


def pullback_hessian(H_t_x,A):
    """J^T H_(T,X) J, where (T,X)=(A Z+b,X).

    Block-by-block to avoid NaNs from undefined X derivatives contaminating
    finite T-T values by NumPy's 0*NaN propagation.
    """
    H=np.asarray(H_t_x,dtype=np.float64)
    A=np.asarray(A,dtype=np.float64)
    if H.shape[-2:]!=(10,10) or A.shape!=(4,4):
        raise ValueError("Expected Hessian (...,10,10) and A=(4,4)")
    out=np.empty_like(H)
    tt=H[...,:4,:4]
    tx=H[...,:4,4:]
    out[...,:4,:4]=np.einsum("ia,...ij,jb->...ab",A,tt,A,optimize=True)
    zx=np.einsum("ia,...ij->...aj",A,tx,optimize=True)
    out[...,:4,4:]=zx
    out[...,4:,:4]=np.swapaxes(zx,-1,-2)
    out[...,4:,4:]=H[...,4:,4:]
    return out


@torch.inference_mode()
def decoder_y(ck,Z,X,batch_size=32,device="auto"):
    """Only frozen NN forward predictions; no autograd/Hessian."""
    where=("cuda" if torch.cuda.is_available() else "cpu") if device=="auto" else device
    torch_device=torch.device(where)
    model=DiscreteNeuralProcess(DiscreteNPConfig(**ck["model_config"]))
    model.load_state_dict(ck["model_state_dict"],strict=True)
    model.to(torch_device).eval()
    bank=torch.tensor(X,dtype=torch.float32,device=torch_device)[None,...]
    n,m=len(Z),len(X)
    pred_eta=np.empty((n,m),np.float64)
    pred_p=np.empty((n,m),np.float64)
    for i in range(0,n,batch_size):
        end=min(i+batch_size,n)
        latent=torch.as_tensor(np.asarray(Z[i:end],np.float32),device=torch_device)
        raw=model.decode(bank.expand(end-i,-1,-1),latent)
        odds=raw[...,1]-raw[...,0]
        probs=torch.softmax(raw,dim=-1)[...,1]
        pred_eta[i:end]=odds.cpu().numpy()
        pred_p[i:end]=probs.cpu().numpy()
        if i==0 or end==n or (i//batch_size)%30==0:
            print(f"Decoder forward at REAL Z: {end}/{n}",flush=True)
    diff=float(np.max(np.abs(expit(pred_eta)-pred_p)))
    if diff>2e-6:raise ValueError(f"NN p inconsistent with sigmoid(log-odds): {diff}")
    return pred_eta,pred_p,diff


def make_six_Y(eta_oracle,p_oracle,eta_nn,p_nn,seed,
               oracle_choice_mode="sampled",nn_choice_mode="hard"):
    if not(all(v.shape==p_oracle.shape for v in (eta_oracle,eta_nn,p_nn))):
        raise ValueError("Response shapes must match")
    rng=np.random.default_rng(seed)
    if oracle_choice_mode=="sampled":
        uniform=rng.random(p_oracle.shape)
        choice_oracle=(uniform<p_oracle).astype(np.float64)
    elif oracle_choice_mode=="hard":
        choice_oracle=(p_oracle>=.5).astype(np.float64)
        uniform=None
    else:raise ValueError("Invalid oracle choice mode")
    if nn_choice_mode=="sampled":
        if uniform is None:uniform=rng.random(p_nn.shape)
        choice_nn=(uniform<p_nn).astype(np.float64)
    elif nn_choice_mode=="hard":
        choice_nn=(p_nn>=.5).astype(np.float64)
    else:raise ValueError("Invalid NN choice mode")
    return dict(oracle_choice=choice_oracle,oracle_logit=eta_oracle,
                oracle_probability=p_oracle,nn_choice=choice_nn,
                nn_logit=eta_nn,nn_probability=p_nn)


def summary_pair(a,b):
    a=np.asarray(a,dtype=float).ravel()
    b=np.asarray(b,dtype=float).ravel()
    d=b-a
    aa=a-a.mean();bb=b-b.mean()
    den=np.linalg.norm(aa)*np.linalg.norm(bb)
    return dict(r=float(aa@bb/den) if den>1e-15 else None,
                rmse=float(np.sqrt(np.mean(d*d))),
                mae=float(np.mean(abs(d))),bias=float(d.mean()),
                mean_oracle=float(a.mean()),mean_nn=float(b.mean()))


def analytic_zx_moments(T_hat,X,A,batch_size=16):
    """Analytic H for CR3 at AZ+b; transformed to Z,X; NO decoder derivatives."""
    counters={key:HessianMoments() for key in ("logit","probability")}
    n=len(T_hat)
    for i in range(0,n,batch_size):
        end=min(i+batch_size,n)
        h_eta,h_p=oracle_hessians(T_hat[i:end],X)
        h_eta=pullback_hessian(h_eta,A)
        h_p=pullback_hessian(h_p,A)
        weights=np.full(end-i,1./n)
        counters["logit"].add(h_eta,weights)
        counters["probability"].add(h_p,weights)
        if i==0 or end==n or (i//batch_size)%30==0:
            print(f"CR3 analytic oracle in ZX coordinates: {end}/{n}",flush=True)
    return {k:v.result() for k,v in counters.items()}


def score_matrix(ref,est,kind):
    """Comparison of estimated mean vs corresponding oracle in ZX coordinates."""
    if kind=="ZZ":s=(slice(0,4),slice(0,4))
    elif kind=="ZX":s=(slice(0,4),slice(4,10))
    elif kind=="XX":s=(slice(4,10),slice(4,10))
    elif kind=="FULL":s=(slice(0,10),slice(0,10))
    else:raise KeyError(kind)
    a=np.asarray(ref["conditional_mean"][s],float)
    b=np.asarray(est["conditional_mean"][s],float)
    valid=np.asarray(ref["fraction_defined"][s])>0
    valid &=np.isfinite(a)&np.isfinite(b)
    if not valid.any():return dict(n=0,mae=None,rmse=None,relative_F=None,cosine=None)
    x=a[valid];y=b[valid];d=y-x
    na=float(np.linalg.norm(x));nb=float(np.linalg.norm(y))
    return dict(n=int(valid.sum()),mae=float(np.mean(abs(d))),
                rmse=float(np.sqrt(np.mean(d*d))),
                relative_F=float(np.linalg.norm(d)/max(na,1e-15)),
                cosine=float(x@y/max(na*nb,1e-15)))


def save_mat(path,mat):
    import csv
    with Path(path).open("w",encoding="utf-8",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["",*NAMES_ZX])
        for key,row in zip(NAMES_ZX,mat):
            writer.writerow([key,*[f"{float(x):.12g}" if np.isfinite(x) else ""
                                     for x in row]])


def plot_matrices(path,oracle,original,network,label,kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    indices={"ZZ":(slice(0,4),slice(0,4)),
             "ZX":(slice(0,4),slice(4,10)),
             "XX":(slice(4,10),slice(4,10)),
             "FULL":(slice(0,10),slice(0,10))}[kind]
    ref=oracle["conditional_mean"][indices]
    old=original["conditional_mean"][indices]
    nn=network["conditional_mean"][indices]
    vals=(ref,old,nn,old-ref,nn-ref)
    vmax=max(1e-10,max(float(np.max(np.abs(v[np.isfinite(v)])))
                     for v in vals if np.isfinite(v).any()))
    fig,axes=plt.subplots(1,5,figsize=(21,4.9),constrained_layout=True)
    labels=("Analytic truth", "Sieve from oracle Y", "Sieve from NN Y",
            "Oracle Y error", "NN Y error")
    for ax,mat,title in zip(axes,vals,labels):
        im=ax.imshow(np.ma.masked_invalid(mat),vmin=-vmax,vmax=vmax,cmap="coolwarm")
        ax.set_title(title,fontsize=10)
        ax.set_xticks(range(mat.shape[1]),NAMES_ZX[indices[1]],rotation=55,ha="right",fontsize=7)
        ax.set_yticks(range(mat.shape[0]),NAMES_ZX[indices[0]],fontsize=7)
    fig.colorbar(im,ax=list(axes),shrink=.7)
    fig.suptitle(f"{label}, H[{kind}] in Z/X coordinates")
    fig.savefig(path,dpi=155)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition",default=CONDITION)
    p.add_argument("--repeat",type=int,default=1)
    p.add_argument("--dataset",type=Path,default=STEM/"s1_generate"/"s1_generate__dataset.npz")
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--latent",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--mapping-fit",choices=("all","train"),default="all",
                   help="Subject IDs to fit affine A,b; default ALL as requested")
    p.add_argument("--no-intercept",action="store_true")
    p.add_argument("--max-condition",type=float,default=1e10)
    p.add_argument("--degree-z",type=int,default=4)
    p.add_argument("--degree-x",type=int,default=3)
    p.add_argument("--ridge-z",type=float,default=.01)
    p.add_argument("--ridge-x",type=float,default=.01)
    p.add_argument("--seed",type=int,default=20261106)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    p.add_argument("--oracle-choice-mode",choices=("sampled","hard"),default="sampled")
    p.add_argument("--nn-choice-mode",choices=("sampled","hard"),default="hard")
    p.add_argument("--no-plots",action="store_true")
    args=p.parse_args()
    if args.repeat<1 or args.batch_size<1 or args.degree_z<2 or args.degree_x<2:
        p.error("Invalid repeat, batch size, or polynomial degree")
    if args.checkpoint is None:
        args.checkpoint=STEM/"s3_train"/args.condition/f"repeat{args.repeat:02d}"/"s3_train__checkpoint.pt"
    if args.latent is None:
        args.latent=STEM/"s4_evaluate"/args.condition/f"repeat{args.repeat:02d}"/"s4_evaluate__latent.npz"
    if args.output is None:
        args.output=ROOT/"outputs"/"math"/"cr3_zx_hessian_threeway"/args.condition/f"repeat{args.repeat:02d}"
    T,X,Z,ck,splits,source=load_sources(args.dataset,args.checkpoint,args.latent)
    args.output.mkdir(parents=True,exist_ok=True)
    fit_ids=(np.arange(len(T)) if args.mapping_fit=="all" else splits["train"])
    A,b,T_hat,singular,kappa=fit_affine(Z,T,fit_ids,intercept=not args.no_intercept,
                                        max_condition=args.max_condition)
    print(f"OLS {args.mapping_fit}: condition(A)={kappa:.4g}; "
          f"T_hat range={np.min(T_hat,axis=0)}..{np.max(T_hat,axis=0)}",flush=True)
    if np.any(T_hat[:,0]<=0):
        raise RuntimeError("Fitted alpha <= 0 would make 0-amount powers ill-defined. "
                           "No silent clipping. Inspect map before running this experiment.")
    # Original S1 choices/true p are not applicable to counterfactual T_hat.
    eta_oracle,p_oracle=oracle_outputs(T_hat,X)
    if not np.all(np.isfinite(eta_oracle)):
        raise ValueError("Nonfinite CR3 oracle logit at predicted T_hat")
    eta_nn,p_nn,prob_check=decoder_y(ck,Z,X,args.batch_size,args.device)
    Y=make_six_Y(eta_oracle,p_oracle,eta_nn,p_nn,args.seed,
                 args.oracle_choice_mode,args.nn_choice_mode)
    channels_paired={
        key:summary_pair(Y[a],Y[b]) for key,(a,b) in PAIRING.items()
    }
    np.savez_compressed(args.output/"all_Y_and_mapping.npz",
        Z_all_trial=Z,T_true=T,T_hat_AZ_plus_b=T_hat,X_fixed=X,
        A=A,b=b,**Y,
        train_ids=splits["train"],validation_ids=splits["validation"],
        test_ids=splits["test"],checkpoint_sha256=np.array(source["checkpoint_sha256"]),
        dataset_sha256=np.array(source["data_sha256"]))
    print("Fitting six Y channels via IDENTICAL data-only TensorPolynomial(Z,X,Y)...",flush=True)
    fit=TensorPolynomial(Z,X,
        degree_t=args.degree_z,degree_x=args.degree_x,
        ridge_t=args.ridge_z,ridge_x=args.ridge_x,seed=args.seed)
    oracle=analytic_zx_moments(T_hat,X,A,batch_size=args.batch_size)
    response_type={"oracle_choice":"probability","oracle_logit":"logit","oracle_probability":"probability",
                   "nn_choice":"probability","nn_logit":"logit","nn_probability":"probability"}
    summary={
        "experiment":"AZ=T, same original Z/X, 2 sources x 3 Y, analytic oracle",
        "coordinate_system":"(z1,z2,z3,z4,gainA,lossA,pA,gainB,lossB,pB)",
        "map_formula":"T_hat_col=A @ Z_col + b, with A=OLS on selected subjects",
        "mapping_fit":args.mapping_fit,"intercept":not args.no_intercept,
        "A":A,"b":b,"A_singular_values":singular,"A_condition_number":kappa,
        "mapping_metrics":mapping_metrics(
            Z,T,T_hat,{"all":np.arange(len(T)),**splits}),
        "mapped_T_out_of_original_support_counts":{
            "alpha":int(np.sum((T_hat[:,0]<0)|(T_hat[:,0]>1))),
            "lambda":int(np.sum((T_hat[:,1]<.3)|(T_hat[:,1]>5))),
            "gamma":int(np.sum((T_hat[:,2]<0)|(T_hat[:,2]>1))),
            "tau":int(np.sum((T_hat[:,3]<.5)|(T_hat[:,3]>4)))},
        "provenance":source,
        "n_subjects":len(Z),"n_trials":len(X),
        "response_channels":{
            "oracle_logit":"analytic CR3 eta(T_hat=A Z+b,X)",
            "oracle_probability":"analytic CR3 sigmoid(eta)",
            "oracle_choice":f"analytic CR3 binary choice ({args.oracle_choice_mode})",
            "nn_logit":"frozen decoder at ACTUAL encoder Z and raw X: class1-class0",
            "nn_probability":"frozen decoder at ACTUAL encoder Z and raw X: softmax",
            "nn_choice":f"frozen NN choice from p ({args.nn_choice_mode})"},
        "nn_probability_consistency_max_abs":prob_check,
        "source_response_comparisons":channels_paired,
        "analytic_reference":"exact CR3 Hessian at T_hat=A Z+b in T,X coordinates, "
                             "then J.T @ H(T,X) @ J with J=block_diag(A,I6)",
        "no_nn_hessian":True,"derivative_estimator":"exact previous TensorPolynomial matrix/sieve regression on ORIGINAL Z,X",
        "regression":dict(degree_z=args.degree_z,degree_x=args.degree_x,
                          ridge_z=args.ridge_z,ridge_x=args.ridge_x,
                          n_train=len(fit.train_idx),n_test=len(fit.test_idx),
                          seed=args.seed),
        "important_caveats":[
            "No neural-network Hessian or autodiff is used: the NN produces Y only.",
            "Changing Y to NN output changes the underlying function; NN-vs-CR3 "
            "disagreement is NOT a pure estimation error.",
            "Oracle Bernoulli choice Hessian means the Hessian of its conditional mean p.",
            "NN hard argmax choice has no smooth probability Hessian: its comparison "
            "to CR3 smooth p is an EXTERNAL reference, not its own target.",
            "Estimator operates on Z,X not T,X. The oracle is transformed via chain rule.",
            "Discrete fixed X points do not nonparametrically identify continuous X Hessians.",
            "At zero gain/loss, classical amount derivatives may be undefined. "
            "Consult strict_mean/fraction_defined. ",
            "OLS map with all subjects uses all true T, so its evaluation is descriptive "
            "and not a held-out map recovery test; use --mapping-fit train for that."],
        "results":{},
    }
    arrays={"A":A,"b":b,"Z":Z,"T_hat":T_hat,
            "sieve_train_ids":fit.train_idx,"sieve_test_ids":fit.test_idx,
            "map_fit_ids":np.asarray(fit_ids,dtype=np.int64)}
    for label,ref in oracle.items():
        for metric in ("conditional_mean","strict_mean","conditional_rms",
                       "conditional_std","fraction_defined"):
            arrays[f"analytic_{label}__{metric}"]=ref[metric]
        save_mat(args.output/f"analytic_{label}__mean.csv",
                 ref["conditional_mean"])
    fitted={}
    for name in CHANNELS:
        print("Infer 10D Hessian from ZX and Y:",name,flush=True)
        coef=fit.fit(Y[name])
        arrays[f"coefficients_{name}"]=coef
        estimated=fit.expected_hessian(coef,Z,np.full(len(Z),1./len(Z)))
        ref=oracle[response_type[name]]
        fitted[name]=estimated
        result=fit.holdout_metrics(Y[name],coef)
        summary["results"][name]=dict(
            holdout_Y=result,
            mean_hessian_vs_original_CR3={
                part:score_matrix(ref,estimated,part)
                for part in ("ZZ","ZX","XX","FULL")},
            hessian_rms_moment_vs_original_CR3={
                part:score_matrix(
                    {"conditional_mean":ref["conditional_rms"],
                     "fraction_defined":ref["fraction_defined"]},
                    {"conditional_mean":estimated["conditional_rms"]},part)
                for part in ("ZZ","ZX","XX","FULL")},
            hessian_sd_moment_vs_original_CR3={
                part:score_matrix(
                    {"conditional_mean":ref["conditional_std"],
                     "fraction_defined":ref["fraction_defined"]},
                    {"conditional_mean":estimated["conditional_std"]},part)
                for part in ("ZZ","ZX","XX","FULL"]})
        for metric in ("conditional_mean","strict_mean","conditional_rms",
                       "conditional_std","fraction_defined","model_unmasked_mean"):
            arrays[f"fitted_{name}__{metric}"]=estimated[metric]
        save_mat(args.output/f"fitted_{name}__mean.csv",
                 estimated["conditional_mean"])
        for part in ("ZZ","ZX","XX"):
            errs=summary["results"][name]["mean_hessian_vs_original_CR3"][part]
            print(f"  {part} MAE={errs['mae']:.5g} rel_F={errs['relative_F']:.4g}",flush=True)
    summary["nn_minus_oracle_estimates"]={}
    for label,(a,b_name) in PAIRING.items():
        delta=fitted[b_name]["conditional_mean"]-fitted[a]["conditional_mean"]
        arrays[f"nn_minus_oracle_sieve__{label}"]=delta
        summary["nn_minus_oracle_estimates"][label]={
            part:score_matrix(
                {"conditional_mean":fitted[a]["conditional_mean"],
                 "fraction_defined":fitted[a]["fraction_defined"]},
                {"conditional_mean":fitted[b_name]["conditional_mean"]},part)
            for part in ("ZZ","ZX","XX","FULL")}
        if not args.no_plots:
            for part in ("ZZ","ZX","XX","FULL"):
                plot_matrices(args.output/f"threeway_{label}_{part}.png",
                              oracle["logit" if label=="logit" else "probability"],
                              fitted[a],fitted[b_name],label,part)
    (args.output/"summary.json").write_text(
        json.dumps(to_builtin(summary),indent=2,ensure_ascii=False,allow_nan=False),
        encoding="utf-8")
    np.savez_compressed(args.output/"all_matrices.npz",**arrays)
    print("Saved complete three-way ZX Hessian experiment:",args.output,flush=True)


if __name__=="__main__":
    main()
