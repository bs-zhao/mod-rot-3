"""CR3: decoder-class-logit differences versus true simulator logits, and p.

One independent, READ-ONLY analysis. No NP training and no upstream changes.

From repository root:
  python analysis/cr3_np_logit_vs_true/run.py

Each subject: infer q(z|ALL 500 observed x,y) using the checkpoint's own
normalization. Decode ALL 500 x using the deterministic posterior mean z.
The binary decoder produces TWO class logits. The comparable *log-odds* is
class_logit[1] - class_logit[0], and p=softmax(class_logits)[...,1].

This is all-context reconstruction. Because each target choice is among
the 500 encoder observations, it is NOT a held-out-trial prediction test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NP_SRC = ROOT / "np" / "src"
NP_REFERENCE = ROOT / "np" / "reference"
sys.path[:0] = [str(HERE), str(NP_SRC), str(NP_REFERENCE)]

from metrics import describe, summarize, axis_stats, binned_true
from plotting import (comparison_scatter, per_axis_hist, calibration_plots,
                      split_summary_plots)
from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig, DiscreteNeuralProcess
from modlearn_rot.training.continuous import load_checkpoint
from s0_cr2_design_recovery import logits as generator_logits

CONDITION = "beta5em05_lambda0_ctx0p25to0p8_L4"
DATA_FILE = ROOT / "outputs/choice/CR3/s1_generate/s1_generate__dataset.npz"
PARAMETER_NAMES = ("alpha", "lambda_loss", "gamma", "tau")
X_NAMES = ("gain_A", "loss_A", "p_gain_A", "gain_B", "loss_B", "p_gain_B")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(piece)
    return result.hexdigest()


def as_int_array(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy().astype(np.int64)
    return np.asarray(arr, dtype=np.int64)


def get_splits(checkpoint, n):
    raw = checkpoint.get("subject_splits")
    if not isinstance(raw, dict):
        raise RuntimeError("Checkpoint missing subject_splits")
    splits = {key: as_int_array(raw[key]).reshape(-1)
              for key in ("train", "validation", "test")}
    all_ids = np.concatenate(list(splits.values()))
    if not np.array_equal(np.sort(all_ids), np.arange(n)):
        raise RuntimeError("Checkpoint splits do not partition the dataset")
    labels = np.empty(n, dtype="<U10")
    for key, ids in splits.items():
        labels[ids] = key
    return splits, labels


def read_data(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing S1 CR3 dataset: {path}")
    with np.load(path, allow_pickle=False) as file:
        required = ("trial_counts", "parameters", "x", "y",
                    "predictive_true", "subject_index", "trial_index")
        absent = sorted(set(required).difference(file.files))
        if absent:
            raise ValueError(f"Missing S1 fields: {absent}")
        T = np.asarray(file["parameters"], np.float64)
        if T.ndim != 2 or T.shape[1] != 4:
            raise ValueError(f"Expected CR3 4P true parameters; got {T.shape}")
        n = len(T)
        counts = np.asarray(file["trial_counts"], np.int64)
        if len(counts) != n or not np.all(counts == 500):
            raise ValueError("This experiment requires the CR3 fixed 500-trial bank")
        xall = np.asarray(file["x"]).reshape(n, 500, 6)
        X = np.array(xall[0], dtype=np.float64, copy=True)
        if not np.all(xall == X[None, :, :]):
            raise ValueError("X is not identical across subjects")
        if not np.array_equal(file["subject_index"], np.repeat(np.arange(n), 500)):
            raise ValueError("Subject index is not in S1 canonical order")
        if not np.array_equal(file["trial_index"], np.tile(np.arange(500), n)):
            raise ValueError("Trial index is not in S1 canonical order")
        choice = np.asarray(file["y"], np.int64).reshape(n, 500)
        true_p = np.asarray(file["predictive_true"], np.float64).reshape(n, 500)
        if not np.isin(choice, [0,1]).all() or not (
            np.isfinite(true_p).all() and
            np.all((true_p >= 0) & (true_p <= 1))
        ):
            raise ValueError("Invalid choice/probability values")
    # The committed S1 generator does not save its pre-sigmoid logits in NPZ.
    # Recover the EXACT pre-sigmoid logit using the SAME code as S1, then
    # check expit(true_logit) against S1's saved predictive_true.
    true_logit = np.asarray(generator_logits(
        X[None,:,:], T[:,None,:]), np.float64)
    if true_logit.shape != (n,500):
        raise RuntimeError("Original generator logit shape mismatch")
    max_p_discrepancy = float(np.max(np.abs(expit(true_logit) - true_p)))
    if not np.allclose(expit(true_logit), true_p, atol=1e-12, rtol=1e-12):
        raise RuntimeError(
            f"Reconstructed true logit does not match S1 true probability: "
            f"max_abs_error={max_p_discrepancy:.12g}. "
            "Do not compare logits until dataset generator is verified."
        )
    return T, X, choice, true_p, true_logit, max_p_discrepancy


@torch.inference_mode()
def infer_all_and_decode(model, ck, X, choice, device, batch_size):
    """Exact S5 encoder preprocessing with ALL trials; deterministic mu decode."""
    n, m = choice.shape
    dx = int(X.shape[1])
    zdim = model.config.latent_dim
    prep = ck["preprocessing"]
    enc_mean = prep["encoder_mean"].to(device, dtype=torch.float32).reshape(-1)
    enc_std = prep["encoder_std"].to(device, dtype=torch.float32).reshape(-1)
    if enc_mean.numel() != dx+1 or enc_std.numel() != dx+1:
        raise RuntimeError("Checkpoint encoder preprocessing dimension mismatch")
    if (enc_std <= 0).any():
        raise RuntimeError("Invalid checkpoint encoder normalization")
    x_bank = torch.tensor(X, dtype=torch.float32, device=device)
    z_means = np.empty((n,zdim), np.float32)
    z_logvars = np.empty_like(z_means)
    logit_pred = np.empty((n,m), np.float32)
    p_pred = np.empty((n,m), np.float32)

    for start in range(0,n,batch_size):
        stop = min(start+batch_size,n)
        b = stop-start
        xs = x_bank.unsqueeze(0).expand(b,-1,-1)
        ys = torch.from_numpy(choice[start:stop]).to(device=device, dtype=torch.float32)
        obs = torch.cat((xs,ys.unsqueeze(-1)),dim=-1)
        standardized = (obs-enc_mean)/enc_std
        context_mask = torch.ones((b,m), dtype=torch.bool, device=device)
        mu, lv = model.encode_context(
            standardized[...,:dx], standardized[...,dx:], context_mask)
        class_logits = model.decode(xs,mu,context_mask)
        if class_logits.shape != (b,m,2):
            raise RuntimeError(f"Expected two decoder class logits, got {class_logits.shape}")
        # Softmax has a common-offset invariance; a single column of the
        # decoder output is NOT the Bernoulli logit.
        log_odds = class_logits[...,1]-class_logits[...,0]
        prob = torch.softmax(class_logits,dim=-1)[...,1]
        if not (torch.isfinite(log_odds).all() and torch.isfinite(prob).all()):
            raise RuntimeError(f"Non-finite decoder outputs: subjects {start}:{stop}")
        z_means[start:stop]=mu.cpu().numpy()
        z_logvars[start:stop]=lv.cpu().numpy()
        logit_pred[start:stop]=log_odds.cpu().numpy()
        p_pred[start:stop]=prob.cpu().numpy()
        if start == 0 or stop == n or ((start//batch_size)%20 == 0):
            print(f"Inferred all 500 choices and decoded: {stop}/{n} subjects",
                  flush=True)
    prob_difference = float(np.max(np.abs(
        expit(logit_pred.astype(np.float64))-p_pred.astype(np.float64))))
    if prob_difference > 2e-6:
        raise RuntimeError(f"Decoder p/log-odds inconsistency: {prob_difference}")
    return z_means,z_logvars,logit_pred,p_pred,prob_difference


def save_csv(path, rows):
    rows=list(rows)
    if not rows:return
    with Path(path).open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def per_axis_rows(a,b,axis,group_labels,theta=None,z=None,x=None):
    result=axis_stats(a,b,axis=axis)
    rows=[]
    n=len(result["rmse"])
    for i in range(n):
        row={"id":i}
        if group_labels is not None:row["split"]=str(group_labels[i])
        if theta is not None:
            row.update({f"true_{name}":float(theta[i,k])
                        for k,name in enumerate(PARAMETER_NAMES)})
        if z is not None:
            row.update({f"z_mean_{j+1}":float(z[i,j]) for j in range(z.shape[1])})
        if x is not None:
            row.update({name:float(x[i,j]) for j,name in enumerate(X_NAMES)})
        for name, values in result.items():
            val=float(values[i])
            row[name]=val if np.isfinite(val) else ""
        rows.append(row)
    return rows,result


def reliability_by_pred_p(true_p,pred_p,choice,bins=25):
    a=np.asarray(true_p,np.float64).ravel()
    p=np.asarray(pred_p,np.float64).ravel()
    y=np.asarray(choice,np.float64).ravel()
    edges=np.unique(np.quantile(p,np.linspace(0,1,bins+1)))
    if len(edges)<3:
        raise ValueError("Predicted probabilities too concentrated to bin")
    index=np.searchsorted(edges[1:-1],p,side="right")
    rows=[]
    for k in range(len(edges)-1):
        mask=index==k
        if not mask.any():continue
        rows.append(dict(bin_index=k,n=int(mask.sum()),
                         pred_min=float(edges[k]),pred_max=float(edges[k+1]),
                         pred_mean=float(np.mean(p[mask])),
                         true_mean=float(np.mean(a[mask])),
                         choice_mean=float(np.mean(y[mask])),
                         mae_to_true_p=float(np.mean(np.abs(a[mask]-p[mask])))))
    return rows


def predictive_metrics(true_logit,logit_pred,true_p,p_pred,choice):
    y=choice.astype(np.float64)
    m = {
        "mean_probability_abs_error":float(np.mean(np.abs(p_pred-true_p))),
        "mean_probability_squared_error":float(np.mean((p_pred-true_p)**2)),
        "brier_observed_choice":float(np.mean((p_pred-y)**2)),
        "brier_oracle_true_probability":float(np.mean((true_p-y)**2)),
        "accuracy_observed_choice":float(np.mean((p_pred>=.5)==choice)),
        "accuracy_oracle_true_probability":float(np.mean((true_p>=.5)==choice)),
        "cross_entropy_observed_choice":float(np.mean(np.logaddexp(0.,logit_pred)-y*logit_pred)),
        "cross_entropy_oracle_true_logits":float(np.mean(np.logaddexp(0.,true_logit)-y*true_logit)),
    }
    return m


def make_plots(folder,truth,estimated,subj_stats,trial_stats,logit_bins,
               prob_bins,all_metrics,cap,seed):
    for channel in ("logit", "probability"):
        comparison_scatter(
            truth[channel],estimated[channel],channel,
            folder/f"{channel}__all_points_scatter.png",cap=cap,seed=seed)
        per_axis_hist(
            subj_stats[channel],trial_stats[channel],channel,
            folder/f"{channel}__subject_trial_diagnostics.png")
        comparison_scatter(
            truth[channel].mean(axis=1),estimated[channel].mean(axis=1),
            channel,folder/f"{channel}__subject_means.png",cap=cap,seed=seed)
        comparison_scatter(
            truth[channel].mean(axis=0),estimated[channel].mean(axis=0),
            channel,folder/f"{channel}__trial_means.png",cap=cap,seed=seed)
    calibration_plots(logit_bins,prob_bins,folder/"logit_and_probability__calibration.png")
    split_summary_plots(all_metrics, folder/"logit_and_probability__splits.png")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", default=CONDITION)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=("auto","cpu","cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-plot-points", type=int, default=150000)
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--no-plots", action="store_true")
    args=parser.parse_args()
    if args.repeat<1 or args.batch_size<1 or args.max_plot_points<100 or args.bins<2:
        parser.error("Invalid repeat, batch size, maximum plotted points, or bin count")
    if args.checkpoint is None:
        args.checkpoint=(ROOT/"outputs/choice/CR3/s3_train"/args.condition/
                         f"repeat{args.repeat:02d}"/"s3_train__checkpoint.pt")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.output is None:
        args.output=(ROOT/"outputs/analysis/cr3_np_logit_vs_true"/args.condition/
                     f"repeat{args.repeat:02d}")
    args.output.mkdir(parents=True, exist_ok=True)
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available()
                        else "cpu" if args.device=="auto" else args.device)
    T,X,choice,true_p,true_logit,true_p_check=read_data(args.data)
    ck=load_checkpoint(args.checkpoint)
    dataset_hash=sha256(args.data)
    checkpoint_hash=sha256(args.checkpoint)
    source_hash=ck.get("extra_state",{}).get("dataset_sha256")
    if source_hash!=dataset_hash:
        raise RuntimeError(
            "Checkpoint was trained on a different S1 dataset: "
            f"checkpoint expects {source_hash}; actual {dataset_hash}.")
    cfg=ck["model_config"]
    if int(cfg["latent_dim"])!=4 or int(cfg["x_dim"])!=6 or int(cfg["num_choices"])!=2:
        raise RuntimeError(f"Unexpected checkpoint NP model config: {cfg}")
    if not np.isclose(float(ck["training_config"]["beta"]),5e-5,rtol=1e-8,atol=1e-13):
        raise RuntimeError("Expected beta=5e-5: supplied checkpoint is a different model")
    splits,subject_split_labels=get_splits(ck,len(T))
    model=DiscreteNeuralProcess(DiscreteNPConfig(**cfg))
    model.load_state_dict(ck["model_state_dict"],strict=True)
    model.to(device).eval()
    print(f"Checkpoint: {args.checkpoint}",flush=True)
    print(f"CR3: {len(T)} subjects × 500 trials, inference={device}, "
          f"latent_dim={cfg['latent_dim']}",flush=True)
    z,z_logvar,pred_logit,pred_p,pred_p_check=infer_all_and_decode(
        model,ck,X,choice,device,args.batch_size)

    truth={"logit":true_logit,"probability":true_p}
    estimated={"logit":pred_logit,"probability":pred_p}
    metrics_out={}
    subject_axis={}
    trial_axis={}
    for channel in ("logit","probability"):
        metrics_out[channel]={"all":summarize(truth[channel],estimated[channel])}
        for split,ids in splits.items():
            metrics_out[channel][split]=summarize(truth[channel],estimated[channel],ids)
        subj_rows,subject_axis[channel]=per_axis_rows(
            truth[channel],estimated[channel],axis=1,
            group_labels=subject_split_labels,theta=T,z=z)
        trial_rows,trial_axis[channel]=per_axis_rows(
            truth[channel],estimated[channel],axis=0,
            group_labels=None,x=X)
        save_csv(args.output/f"{channel}__per_subject.csv",subj_rows)
        save_csv(args.output/f"{channel}__per_trial.csv",trial_rows)

    logit_bins=binned_true(true_logit,pred_logit,bins=args.bins)
    p_bins=reliability_by_pred_p(true_p,pred_p,choice,bins=args.bins)
    save_csv(args.output/"logit__binned_by_true.csv",logit_bins)
    save_csv(args.output/"probability__binned_by_prediction.csv",p_bins)
    behavior={split:predictive_metrics(
        true_logit[ids],pred_logit[ids],
        true_p[ids],pred_p[ids],choice[ids])
        for split,ids in {"all":np.arange(len(T)),**splits}.items()}

    # Both complete per-subject/per-trial tables (NOT a scatter subsample).
    np.savez_compressed(args.output/"all_subject_trial_predictions.npz",
        true_logit=true_logit, pred_logit=pred_logit,
        true_p=true_p, pred_p=pred_p, binary_choice=choice.astype(np.uint8),
        z_all_trial_posterior_mean=z,z_all_trial_posterior_logvar=z_logvar,
        true_theta=T, fixed_X=X,
        train_ids=splits["train"],validation_ids=splits["validation"],test_ids=splits["test"],
        checkpoint_sha256=np.array(checkpoint_hash),
        dataset_sha256=np.array(dataset_hash),
        inference_mode=np.array("all_500_observed_choices_posterior_mean"),
    )

    summary={
        "experiment":"CR3 full-trial posterior mean Z -> decoder logit and p vs true generator",
        "condition":args.condition,"repeat":args.repeat,
        "n_subjects":int(len(T)),"n_trials_per_subject":int(X.shape[0]),
        "checkpoint":str(args.checkpoint),"checkpoint_sha256":checkpoint_hash,
        "dataset":str(args.data),"dataset_sha256":dataset_hash,
        "model_config":cfg,"device":str(device),
        "split_sizes":{key:int(len(ids)) for key,ids in splits.items()},
        "full_context":{
            "encoder_observation":"(X, observed binary Y) for all 500 trials",
            "encoder_input_preprocessing":"saved checkpoint encoder_mean/std",
            "latent": "deterministic posterior mean, NOT a sampled z",
            "decoder_input":"original unnormalized raw X and the per-subject posterior mean z",
            "decoder_log_odds":"two_class_logits[...,1]-two_class_logits[...,0]",
            "decoder_p":"softmax(two_class_logits,dim=-1)[...,1]",
            "important_limitation":"Every target choice is also in its subject's encoder context. "
                "This is full-context reconstruction, not held-out-trial prediction. "
                "Train/validation/test still refer to the original subject splits.",
        },
        "ground_truth":{
            "true_probability":"S1 predictive_true saved array",
            "true_logit":"recomputed using EXACT original S1 CR3 generator "
                "np/reference/s0_cr2_design_recovery.py:logits; S1 did not save logit array",
            "max_abs_generator_probability_difference":true_p_check,
            "max_abs_decoder_probability_vs_sigmoid_log_odds":pred_p_check,
        },
        "logit_vs_truth":metrics_out["logit"],
        "probability_vs_truth":metrics_out["probability"],
        "binary_choice_performance":behavior,
    }
    (args.output/"summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),
        encoding="utf-8")
    if not args.no_plots:
        make_plots(args.output,truth,estimated,subject_axis,trial_axis,
                   logit_bins,p_bins,metrics_out,args.max_plot_points,args.seed)

    print("RESULTS",flush=True)
    for channel in ("logit","probability"):
        for group in ("all","train","validation","test"):
            block=metrics_out[channel][group]
            print(f"{channel:11} {group:10} pooled r={block['pooled']['pearson_r']}, "
                  f"RMSE={block['pooled']['rmse']:.6g}, "
                  f"within-trial r={block['within_trial_centered']['pearson_r']}, "
                  f"median trial r={block['trial_r_median']}",flush=True)
    print(f"Saved complete (subjects × 500 trials) results: {args.output}",flush=True)


if __name__=="__main__":
    main()
