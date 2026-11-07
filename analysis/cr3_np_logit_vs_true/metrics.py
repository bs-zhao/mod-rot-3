"""Statistics for exact per-subject/per-trial NP-logit comparisons.

Pure NumPy. Uses regression convention predicted = intercept + slope*true.
Correlations are NaN for a degenerate constant series (never forced to zero).
"""
from __future__ import annotations
import numpy as np


def _safe_float(value):
    x = float(value)
    return x if np.isfinite(x) else None


def describe(true, pred):
    a = np.asarray(true, np.float64).reshape(-1)
    b = np.asarray(pred, np.float64).reshape(-1)
    if a.shape != b.shape or len(a) == 0 or not (
        np.isfinite(a).all() and np.isfinite(b).all()
    ):
        raise ValueError("Nonempty, matching finite true and predicted arrays required")
    error = b-a
    x = a-a.mean()
    y = b-b.mean()
    xx = float(np.mean(x*x))
    yy = float(np.mean(y*y))
    xy = float(np.mean(x*y))
    r = xy / np.sqrt(xx*yy) if xx > 1e-24 and yy > 1e-24 else np.nan
    slope = xy / xx if xx > 1e-24 else np.nan
    intercept = float(b.mean()-slope*a.mean()) if np.isfinite(slope) else np.nan
    r2 = 1.-float(np.mean(error**2))/xx if xx > 1e-24 else np.nan
    return {
        "n": int(len(a)),
        "pearson_r": _safe_float(r),
        "r2_prediction": _safe_float(r2),
        "slope_pred_on_true": _safe_float(slope),
        "intercept_pred_on_true": _safe_float(intercept),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error*error))),
        "bias_pred_minus_true": float(np.mean(error)),
        "true_mean": float(a.mean()),
        "pred_mean": float(b.mean()),
        "true_sd": float(np.sqrt(xx)),
        "pred_sd": float(np.sqrt(yy)),
        "true_min": float(a.min()), "true_max": float(a.max()),
        "pred_min": float(b.min()), "pred_max": float(b.max()),
    }


def axis_stats(true, pred, axis):
    """N x T -> one result per subject (axis=1) or trial (axis=0)."""
    a = np.asarray(true, np.float64)
    b = np.asarray(pred, np.float64)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError("Expected two matching 2-D arrays")
    error = b-a
    ac = a-a.mean(axis=axis, keepdims=True)
    bc = b-b.mean(axis=axis, keepdims=True)
    xx = np.mean(ac**2, axis=axis)
    yy = np.mean(bc**2, axis=axis)
    xy = np.mean(ac*bc, axis=axis)
    ok = (xx > 1e-24) & (yy > 1e-24)
    r = np.divide(xy, np.sqrt(xx*yy), out=np.full_like(xx, np.nan), where=ok)
    slope = np.divide(xy, xx, out=np.full_like(xx, np.nan), where=xx>1e-24)
    intercept = b.mean(axis=axis)-slope*a.mean(axis=axis)
    mse = np.mean(error*error, axis=axis)
    r2 = 1 - np.divide(mse, xx, out=np.full_like(xx,np.nan),where=xx>1e-24)
    return {
        "pearson_r": r,
        "r2_prediction": r2,
        "slope_pred_on_true": slope,
        "intercept_pred_on_true": intercept,
        "mae": np.mean(np.abs(error), axis=axis),
        "rmse": np.sqrt(mse),
        "bias_pred_minus_true": np.mean(error, axis=axis),
        "true_mean": np.mean(a, axis=axis),
        "pred_mean": np.mean(b, axis=axis),
        "true_sd": np.sqrt(xx), "pred_sd": np.sqrt(yy),
    }


def summarize(true, pred, ids=None):
    """Global, within-subject, within-trial, subject/trial-mean statistics.

    Overall Pearson may be dominated by 500 shared trial effects; within-
    trial correlation across subjects is the important individual-difference
    diagnostic. True/pred arrays are subjects by trials.
    """
    a = np.asarray(true, np.float64)
    b = np.asarray(pred, np.float64)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError("Expected arrays with shape (subjects,trials)")
    if ids is not None:
        ids = np.asarray(ids, dtype=np.int64)
        a, b = a[ids], b[ids]
    if len(a) < 2 or a.shape[1] < 2:
        raise ValueError("Need >=2 subjects and >=2 trials")
    subj = axis_stats(a,b,axis=1)
    trial = axis_stats(a,b,axis=0)
    return {
        "pooled": describe(a,b),
        "within_subject_centered": describe(
            a-a.mean(1,keepdims=True),b-b.mean(1,keepdims=True)),
        "within_trial_centered": describe(
            a-a.mean(0,keepdims=True),b-b.mean(0,keepdims=True)),
        "subject_mean_across_subjects": describe(a.mean(1),b.mean(1)),
        "trial_mean_across_trials": describe(a.mean(0),b.mean(0)),
        "subject_r_median": _safe_float(np.nanmedian(subj["pearson_r"])),
        "subject_r_mean": _safe_float(np.nanmean(subj["pearson_r"])),
        "trial_r_median": _safe_float(np.nanmedian(trial["pearson_r"])),
        "trial_r_mean": _safe_float(np.nanmean(trial["pearson_r"])),
        "subject_r_finite_count": int(np.isfinite(subj["pearson_r"]).sum()),
        "trial_r_finite_count": int(np.isfinite(trial["pearson_r"]).sum()),
    }


def binned_true(true, pred, bins=25):
    a=np.asarray(true, np.float64).ravel()
    b=np.asarray(pred, np.float64).ravel()
    if a.shape!=b.shape or len(a)<bins or bins<2:
        raise ValueError("Invalid bins or arrays")
    edges=np.unique(np.quantile(a,np.linspace(0,1,bins+1)))
    if len(edges)<3:
        raise ValueError("True logits have too few distinct quantile values")
    index=np.clip(np.searchsorted(edges[1:-1],a,side="right"),0,len(edges)-2)
    rows=[]
    for k in range(len(edges)-1):
        mask=index==k
        if not mask.any():continue
        rows.append(dict(bin_index=k,n=int(mask.sum()),true_min=float(edges[k]),
                         true_max=float(edges[k+1]),true_mean=float(a[mask].mean()),
                         pred_mean=float(b[mask].mean()),
                         pred_sd=float(b[mask].std()),
                         residual_mean=float((b[mask]-a[mask]).mean())))
    return rows
