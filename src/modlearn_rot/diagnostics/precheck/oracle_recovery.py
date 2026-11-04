"""Parameter oracle recovery across simulated subjects.

Default gate remains all per-parameter uncalibrated R2 > 0.90.
The preregistered research-based CR1 v3 shared-gamma variant instead
accepts Pearson r >= 0.80 for EACH of its four physical parameters.

Both criteria are reported for every model so r is never confused with R2.
"""
import numpy as np
from time import perf_counter
from ...generators.dataset import iter_subject_slices


def _r2(y,p):
    ssr=float(np.sum((y-p)**2)); sst=float(np.sum((y-y.mean())**2))
    return 1.0-ssr/sst if sst>0 else float("nan")


def pearson_parameter_correlations(truth, estimates):
    """Return column-wise actual Pearson r, not sqrt(R2).

    Undefined correlations remain NaN and FAIL every recovery gate.
    """
    truth=np.asarray(truth,dtype=np.float64)
    estimates=np.asarray(estimates,dtype=np.float64)
    if truth.ndim!=2 or estimates.shape!=truth.shape:
        raise ValueError("Truth and oracle estimates must be matching 2D arrays")
    a=truth-truth.mean(axis=0)
    b=estimates-estimates.mean(axis=0)
    numerator=np.sum(a*b,axis=0)
    denominator=np.sqrt(np.sum(a*a,axis=0)*np.sum(b*b,axis=0))
    with np.errstate(divide="ignore",invalid="ignore"):
        correlations=numerator/denominator
    correlations=np.where(denominator>0,correlations,np.nan)
    return correlations


def recovery_acceptance(model, r2, pearson_r):
    """Choose a MODEL-SPECIFIC, explicitly documented recovery gate."""
    r2=np.asarray(r2,dtype=float)
    pearson_r=np.asarray(pearson_r,dtype=float)
    if getattr(model,"model_id",None)=="CR1":
        threshold=0.80
        criterion="per_parameter_pearson_r"
        valid=np.isfinite(pearson_r)&(pearson_r>=threshold)
    else:
        threshold=0.90
        criterion="per_parameter_uncalibrated_r2"
        valid=np.isfinite(r2)&(r2>threshold)
    return {
        "criterion":criterion,
        "threshold":threshold,
        "each_parameter_passed":valid,
        "passed":bool(np.all(valid)),
    }


def run_oracle_recovery(model,dataset):
    est=np.empty_like(dataset.parameters,dtype=float)
    started=perf_counter()
    total=len(dataset.trial_counts)
    if int(model.n_subjects)!=total:
        raise ValueError("Oracle dataset subject count differs from model")
    print(f"Oracle recovery: {total} subjects, "
          f"{int(np.min(dataset.trial_counts))}-"
          f"{int(np.max(dataset.trial_counts))} trials each",flush=True)
    for sid,left,right in iter_subject_slices(dataset):
        est[sid]=model.oracle_fit(
            dataset.x[left:right],dataset.y[left:right]
        )
        if (sid+1)%100==0 or sid+1==total:
            elapsed=perf_counter()-started
            rate=(sid+1)/max(elapsed,1e-10)
            remaining=(total-sid-1)/max(rate,1e-10)
            print(
                f"Oracle recovery: {sid+1}/{total} | "
                f"elapsed={elapsed/60:.1f} min | "
                f"ETA={remaining/60:.1f} min",flush=True
            )
    truth=np.asarray(dataset.parameters,dtype=float)
    r2=np.asarray([_r2(truth[:,j],est[:,j]) for j in range(model.n_factors)])
    pearson_r=pearson_parameter_correlations(truth,est)
    rmse=np.sqrt(np.mean((est-truth)**2,axis=0))
    bias=np.mean(est-truth,axis=0)
    gate=recovery_acceptance(model,r2,pearson_r)
    return {
        "estimates":est,
        "r2":r2,
        "pearson_r":pearson_r,
        "rmse":rmse,
        "bias":bias,
        "mean_r2":float(r2.mean()),
        "min_r2":float(r2.min()),
        "mean_pearson_r":float(pearson_r.mean()),
        "min_pearson_r":float(pearson_r.min()),
        "acceptance_criterion":gate["criterion"],
        "acceptance_threshold":gate["threshold"],
        "each_parameter_passed":gate["each_parameter_passed"],
        "passed":gate["passed"],
    }
