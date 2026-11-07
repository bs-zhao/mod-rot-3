"""Scale-invariant, sign-invariant Hessian sparsity losses.

Motivation: [2,.011,.001] should be distinguishable from [2,.001,.001],
and [2,2,.01,.001] from [2,2,.001,.001], WITHOUT assuming a priori
that exactly one element per Z row is relevant.

All keys ending in '_loss' are to MINIMIZE. Diagnostic keys end in
'_diagnostic'.  Relative thresholds are measured against the maximum
absolute element of EACH vector, and must be checked on multiple scales.

A single hard support threshold is NOT a valid final sparsity objective:
it produces discontinuities and may classify every effect as active.
The continuous capped/log-sum/penalized-support losses are the primary
candidates; hard support sizes are diagnostic only.
"""
from __future__ import annotations
import numpy as np

# Fractions of the largest absolute effect within a vector.
SCALES=((0.0005,"0p0005"),(0.001,"0p001"),
        (0.005,"0p005"),(0.02,"0p02"))
SMALL_TOL=1e-13
PRIMARY=(
    "hoyer_loss",
    "lp025_loss",
    "lp05_loss",
    "tail1_k1_loss",
    "tail1_k2_loss",
    "tail2_k2_loss",
    "tail2sq_k2_loss",
    "softl0_t0p005_loss",
    "logsum_t0p005_loss",
    "capped_l2_t0p005_loss",
    "penalized_support_t0p005_loss",
)


def _clean(v):
    v=np.abs(np.asarray(v,dtype=np.float64).ravel())
    if v.ndim!=1 or not np.isfinite(v).all():
        raise ValueError("Sparsity input must be a finite vector")
    return v


def vector_losses(v,max_topk=4):
    """All scalar losses, 0 means an all-zero vector / perfect sparsity.

    Most are dimensionless and invariant under positive rescaling,
    coordinate permutations and sign changes. Counts are not normalized
    by vector length; only compare fixed block shapes across angles.
    """
    a=_clean(v)
    d=len(a)
    if not d:raise ValueError("Empty sparsity vector")
    m=float(a.max())
    out={}
    if m<=SMALL_TOL:
        # The zero vector is maximally sparse. Any smallest-magnitude
        # Hessian may yield low loss; report its Frobenius norm separately.
        s=np.zeros(d,dtype=float)
    else:
        s=np.sort(a/m)[::-1]
    l1=float(np.sum(s));l2=float(np.linalg.norm(s))
    if l2>1e-15:
        out["hoyer_loss"]=float((l1/l2-1)/(np.sqrt(d)-1)) if d>1 else 0.
    else:
        out["hoyer_loss"]=0.
    out["lp025_loss"]=float(np.sum(s**.25))
    out["lp05_loss"]=float(np.sum(np.sqrt(s)))
    out["lp075_loss"]=float(np.sum(s**.75))
    if l1>1e-15:
        p=s/l1
        q=p[p>1e-15]
        out["entropy_loss"]=float(-np.sum(q*np.log(q))/np.log(d)) if d>1 else 0.
        out["renyi05_loss"]=float(2*np.log(np.sqrt(q).sum())/np.log(d)) if d>1 else 0.
        out["renyi2_loss"]=float(-np.log(np.sum(q*q))/np.log(d)) if d>1 else 0.
        out["gini_impurity_loss"]=float((1-np.sum(p*p))/(1-1/d)) if d>1 else 0.
    else:
        for key in ("entropy_loss","renyi05_loss","renyi2_loss","gini_impurity_loss"):
            out[key]=0.
    out["active_nonzero_diagnostic"]=int(np.sum(a>SMALL_TOL))
    for k in range(1,min(d,max_topk+1)):
        tail=s[k:]
        out[f"tail1_k{k}_loss"]=float(tail.sum())
        out[f"tail2_k{k}_loss"]=float(np.linalg.norm(tail))
        out[f"tail2sq_k{k}_loss"]=float(np.dot(tail,tail))
        out[f"next_k{k}_loss"]=float(s[k])
    for threshold,label in SCALES:
        ratio=s/threshold
        # Large terms saturate; distinguish tiny nonzero residuals.
        out[f"softl0_t{label}_loss"]=float(np.sum(-np.expm1(-ratio)))
        out[f"capped_l1_t{label}_loss"]=float(np.minimum(ratio,1).sum())
        out[f"capped_l2_t{label}_loss"]=float(np.minimum(ratio*ratio,1).sum())
        out[f"logsum_t{label}_loss"]=float(
            np.log1p(ratio).sum()/np.log1p(1/threshold))
        out[f"fraction_t{label}_loss"]=float(
            np.sum(ratio/(1+ratio)))
        # Equivalent to min_{k=0..d} [lambda*k + sum_{j>k} s_j²].
        # No fixed k and no discontinuous hard-support selection.
        tail_sq=np.r_[np.cumsum(s[::-1]**2)[::-1],0.]
        penalties=np.arange(d+1)*threshold**2+tail_sq
        out[f"penalized_support_t{label}_loss"]=float(np.min(penalties))
        out[f"support_count_t{label}_diagnostic"]=int(np.sum(s>threshold))
        out[f"support_tail2_t{label}_diagnostic"]=float(
            np.sum(s[s<=threshold]**2))
    return out


def independent_entries(M,block):
    """Count symmetric off-diagonal entries once for flat block sparsity."""
    a=np.asarray(M,dtype=float)
    if a.ndim==1:return a
    if block in ("ZZ","XX","FULL"):
        return a[np.triu_indices(len(a))]
    return a.ravel()


def _aggregate_rows(a,prefix,*,major_only=False):
    a=np.asarray(a,dtype=float)
    if a.ndim!=2:raise ValueError("Expected 2D array for row/column sparsity")
    per=[vector_losses(row) for row in a]
    result={}
    keys=per[0].keys()
    for key in keys:
        if major_only and key not in PRIMARY:continue
        vals=np.asarray([v[key] for v in per],dtype=float)
        result[f"{prefix}_mean_{key}"]=float(np.mean(vals))
        # Worst-row scores: for a rotation basis, requiring EVERY Z
        # row to have sparse X effects can be stricter than mean alone.
        if key in PRIMARY:
            result[f"{prefix}_max_{key}"]=float(np.max(vals))
    return result


def rich_sparsity(M,block,*,x_std=None):
    """Return flat metric family + row/column metrics for ZX.

    x_std: empirical std of the SIX fixed X columns. Both raw and
    X-standardized effects can be reported separately. The latter are
    mixed partials wrt standardized X coordinates, not original units.
    """
    a=np.asarray(M,dtype=float)
    ans={f"flat_{key}":val
         for key,val in vector_losses(independent_entries(a,block)).items()}
    if block=="ZX":
        ans.update(_aggregate_rows(a,"zx_row"))
        # A column describes how many latent U axes influence one X.
        ans.update(_aggregate_rows(a.T,"zx_col",major_only=True))
        if x_std is not None:
            sd=np.asarray(x_std,dtype=float).ravel()
            if sd.shape!=(a.shape[1],) or np.any(sd<=0) or not np.isfinite(sd).all():
                raise ValueError("Expected six positive X standard deviations")
            scaled=a*sd[None,:]
            ans.update(_aggregate_rows(scaled,"zx_xstd_row"))
            ans.update(_aggregate_rows(scaled.T,"zx_xstd_col",major_only=True))
            ans.update({f"xstd_flat_{key}":val for key,val in
                        vector_losses(scaled.ravel()).items()})
    elif block=="XX" and x_std is not None:
        sd=np.asarray(x_std,dtype=float).ravel()
        if sd.shape!=(6,) or np.any(sd<=0) or not np.isfinite(sd).all():
            raise ValueError("Expected six positive X standard deviations")
        scaled=sd[:,None]*a*sd[None,:]
        ans.update({f"xstd_flat_{key}":val for key,val in
                    vector_losses(independent_entries(scaled,"XX")).items()})
    return ans


def metric_catalog():
    """Document optimization direction and interpretation; valid in any block."""
    catalog=[]
    sample=vector_losses([2.,.01,.001,.0,.0,.0])
    for key in sample:
        family=key.rsplit("_",1)[0]
        if key.endswith("_loss"):
            kind="min"
            if key.startswith("tail") or key.startswith("next"):
                desc="Sorted-effect tail beyond a fixed top-k; compare across k"
            elif key.startswith("penalized_support"):
                desc="Optimize support count and squared residual jointly; continuous nonconvex surrogate"
            elif key.startswith(("capped","softl0","logsum","fraction")):
                desc="Soft L0 / saturated penalty; relative threshold is within-vector max"
            elif key.startswith("lp"):
                desc="Sublinear quasi-norm of relative effect magnitudes, small effects weighted"
            else:desc="Scale-invariant effective support / inequality / concentration measure"
        else:
            kind="diagnostic"
            desc="Hard effective support and residual; discontinuous, not an objective by itself"
        catalog.append(dict(base_metric=key,selection=kind,family=family,
                            interpretation=desc))
    return catalog
