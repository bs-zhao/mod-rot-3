"""Standalone Hessian field -> matrix statistics -> scalar complexity atlas.

No imports from any other math experiment.  Metric names encode block,
aggregation, row/column/global scope, and loss-vs-diagnostic status.
Local-first scoring is computed separately from matrix-first scoring.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-12
AGGREGATORS = (
    "signed", "mean_abs", "rms", "mean_square", "variance", "std",
    "median_abs", "q90_abs", "q99_abs", "max_abs",
    "sign_persist", "positive", "negative", "trimmed_abs",
)
INVARIANT_DIAGNOSTICS = ("zz_local_frobenius_sq_diagnostic",
                         "zx_local_frobenius_sq_diagnostic")


def norm2(v):
    return float(np.linalg.norm(np.asarray(v, float)))


def tail(v, k, p=2, relative=False):
    """Best k-term approximation residual; k MUST be fixed a priori."""
    a = np.sort(np.abs(np.asarray(v, float).ravel()))[::-1]
    if k < 0 or k >= len(a):
        raise ValueError("tail k must satisfy 0 <= k < dimension")
    val = float(np.linalg.norm(a[k:], ord=p))
    if relative:
        val /= max(float(np.linalg.norm(a, ord=p)), EPS)
    return val


def gini(a):
    a = np.abs(np.asarray(a, float).ravel())
    s = float(a.sum())
    if s < EPS:
        return 0.0
    return float(np.abs(a[:, None]-a[None, :]).sum()/(2*len(a)*s))


def vector_scores(v, scale, ks=(1, 2, 3, 4), relative_threshold=.10):
    """~30 nonnegative complexity objectives, all MINIMIZED, given 1D effects."""
    a = np.abs(np.asarray(v, float).ravel())
    d = a.size
    if d < 2:
        return {}
    scale = max(float(scale), EPS)
    amax = max(float(a.max()), EPS)
    s = a/amax
    tau = max(float(relative_threshold)*scale, EPS)
    p = a/max(float(a.sum()), EPS)
    q = a*a/max(float(np.sum(a*a)), EPS)
    nz = p > 0
    entropy = -float(np.sum(p[nz]*np.log(p[nz])))
    l1 = float(np.sum(a))
    l2 = norm2(a)
    hoyer = (np.sqrt(d)-l1/max(l2, EPS))/(np.sqrt(d)-1)
    out = dict(
        l1_loss=l1, l2_loss=l2, l2sq_loss=l2*l2,
        l1_over_l2_loss=l1/max(l2, EPS),
        hoyer_loss=float(1-hoyer),
        shannon_loss=entropy,
        effective_support_loss=float(np.exp(entropy)),
        renyi2_loss=float(-np.log(max(float(np.sum(p*p)), EPS))),
        participation_ratio_loss=float(l1*l1/max(l2*l2, EPS)),
        gini_loss=float(1-gini(a)),
        # Separate absolute-unit and scale-relative sparsity thresholds.
        hard_support_loss=float(np.count_nonzero(a > tau)),
        softl0_rational_loss=float(np.sum(a*a/(a*a+tau*tau))),
        softl0_exponential_loss=float(np.sum(-np.expm1(-a/tau))),
        capped_l1_loss=float(np.minimum(a/tau, 1).sum()),
        capped_l2sq_loss=float(np.minimum((a/tau)**2, 1).sum()),
        logsum_loss=float(np.log1p(a/tau).sum()),
        lp025_loss=float(np.sum(a**.25)),
        lp050_loss=float(np.sum(np.sqrt(a))),
        lp075_loss=float(np.sum(a**.75)),
        relative_hard_support_loss=float(np.count_nonzero(s > relative_threshold)),
        relative_softl0_loss=float(np.sum(s*s/(s*s+relative_threshold**2))),
        relative_capped_l1_loss=float(np.minimum(s/relative_threshold, 1).sum()),
        relative_logsum_loss=float(np.log1p(s/relative_threshold).sum()),
        top1_fraction_loss=float(1.0-float(q.max())),
    )
    for k in ks:
        if k >= d:
            continue
        t1 = tail(a,k,1)
        t2 = tail(a,k,2)
        out[f"tail1_k{k}_loss"] = t1
        out[f"tail2_k{k}_loss"] = t2
        out[f"tail2sq_k{k}_loss"] = t2*t2
        out[f"relative_tail1_k{k}_loss"] = t1/max(l1, EPS)
        out[f"relative_tail2_k{k}_loss"] = t2/max(l2, EPS)
        out[f"relative_tail2sq_k{k}_loss"] = t2*t2/max(l2*l2, EPS)
        out[f"next_k{k}_loss"] = float(np.sort(a)[::-1][k])
    # Adaptive k trades residual complexity against paying for each retained term.
    for penalty in (.01, .05, .20):
        lam = penalty * scale**2
        candidates = [np.sum(np.sort(a*a)[::-1][k:])+lam*k
                      for k in range(0,d+1)]
        out[f"adaptive_k_{str(penalty).replace('.','p')}_loss"] = float(min(candidates))
    return out


def aggregate(h):
    """Matrix-wise statistics, without ever treating E[H]^2 as E[H^2]."""
    h = np.asarray(h, float)
    if h.ndim != 3 or not np.isfinite(h).all():
        raise ValueError("Local Hessian field must be finite (N,rows,cols)")
    av = np.abs(h)
    mu = np.mean(h,axis=0)
    sq = np.mean(h*h,axis=0)
    ss = np.sort(av,axis=0)
    n = h.shape[0]
    trim = int(.10*n)
    trimmed = ss[trim:n-trim].mean(axis=0) if 2*trim < n else av.mean(axis=0)
    return {
        "signed":mu,
        "mean_abs":av.mean(axis=0),
        "rms":np.sqrt(sq),
        "mean_square":sq,
        "variance":np.maximum(sq-mu*mu,0),
        "std":np.sqrt(np.maximum(sq-mu*mu,0)),
        "median_abs":np.median(av,axis=0),
        "q90_abs":np.quantile(av,.90,axis=0),
        "q99_abs":np.quantile(av,.99,axis=0),
        "max_abs":av.max(axis=0),
        "sign_persist":np.abs(np.sign(h).mean(axis=0)),
        "positive":np.maximum(h,0).mean(axis=0),
        "negative":np.maximum(-h,0).mean(axis=0),
        "trimmed_abs":trimmed,
    }


def _summarize_vectors(rows, prefix, scale, scores, threshold):
    # Mean across rows: each input parameter gets equal weight.
    values = [vector_scores(v, scale, relative_threshold=threshold) for v in rows]
    if not values or not values[0]:
        return
    for name in values[0]:
        scores[f"{prefix}_{name}"] = float(np.mean([v[name] for v in values]))


def _field_scores(zz, zx, scores, rz, rx, threshold):
    n, d, _ = zz.shape
    off = zz[:, ~np.eye(d,dtype=bool)]
    diag = np.diagonal(zz,axis1=1,axis2=2)
    scores.update({
        "zz_local_diag_abs_loss":float(np.mean(np.abs(diag))),
        "zz_local_diag_rms_loss":float(np.sqrt(np.mean(diag**2))),
        "zz_local_diag_variance_loss":float(np.mean(np.var(diag,axis=0))),
        "zz_local_offdiag_abs_loss":float(np.mean(np.abs(off))),
        "zz_local_offdiag_rms_loss":float(np.sqrt(np.mean(off**2))),
        "zz_local_offdiag_energy_loss":float(np.mean(np.sum(off*off,axis=1))),
        "zz_local_offdiag_fraction_loss":float(np.sum(off*off)/max(float(np.sum(zz*zz)), EPS)),
        "zx_local_abs_loss":float(np.mean(np.abs(zx))),
        "zx_local_rms_loss":float(np.sqrt(np.mean(zx*zx))),
        "zx_local_variance_loss":float(np.mean(np.var(zx,axis=0))),
        "zz_local_frobenius_sq_diagnostic":float(np.mean(np.sum(zz*zz,axis=(1,2)))),
        "zx_local_frobenius_sq_diagnostic":float(np.mean(np.sum(zx*zx,axis=(1,2)))),
        "zz_mean_cancellation_diagnostic":float(norm2(zz.mean(axis=0))/max(float(np.sqrt(np.mean(np.sum(zz*zz,axis=(1,2))))),EPS)),
        "zx_mean_cancellation_diagnostic":float(norm2(zx.mean(axis=0))/max(float(np.sqrt(np.mean(np.sum(zx*zx,axis=(1,2))))),EPS)),
    })
    # Local-first order deliberately kept separate from matrix-first.
    idx = np.linspace(0,n-1,min(n,24),dtype=int)
    for block,arr,scl in (("zx",zx,rx),("zz_off",np.stack(
            [zz[:,i, np.arange(d)!=i] for i in range(d)],axis=1),rz)):
        for k in (1,2,3):
            if k >= arr.shape[-1]:
                continue
            vals = [tail(v,k,2) for v in arr[idx].reshape(-1,arr.shape[-1])]
            scores[f"{block}_localfirst_tail2_k{k}_loss"] = float(np.mean(vals))
            scores[f"{block}_localfirst_tail2sq_k{k}_loss"] = float(np.mean(np.square(vals)))
        soft = [vector_scores(v,scl,relative_threshold=threshold)["softl0_rational_loss"]
                for v in arr[idx].reshape(-1,arr.shape[-1]) if arr.shape[-1]>1]
        if soft:
            scores[f"{block}_localfirst_softl0_loss"] = float(np.mean(soft))


def compute_metrics(zz, zx, threshold=.10, higher=None):
    """Return wide dictionary with hundreds of independent scored metrics.

    zz=(N,d,d), zx=(N,d,p), both already scaled to dimensionless W
    coordinates and rotated at the LOCAL Hessian level.
    """
    zz=np.asarray(zz,float);zx=np.asarray(zx,float)
    if zz.ndim!=3 or zx.ndim!=3 or zz.shape[:2]!=zx.shape[:2] or zz.shape[2]!=zz.shape[1]:
        raise ValueError("Invalid ZZ and ZX tensor shapes")
    if not np.isfinite(zz).all() or not np.isfinite(zx).all():
        raise ValueError("Nonfinite input Hessian; do not silently drop singular CR3 rows")
    d=zz.shape[1]
    rz=np.sqrt(np.mean(zz**2))
    rx=np.sqrt(np.mean(zx**2))
    scores={}
    _field_scores(zz,zx,scores,rz,rx,threshold)
    agg_z=aggregate(zz);agg_x=aggregate(zx)
    for key in AGGREGATORS:
        A,B=agg_z[key],agg_x[key]
        ab=np.abs(A); bb=np.abs(B)
        diagonal=np.diag(ab)
        off=np.array([ab[i,j] for i in range(d) for j in range(i+1,d)])
        pre=f"zz_{key}"
        scores[f"{pre}_diag_l1_loss"]=float(diagonal.sum())
        scores[f"{pre}_diag_l2_loss"]=norm2(diagonal)
        scores[f"{pre}_offdiag_l1_loss"]=float(off.sum())
        scores[f"{pre}_offdiag_l2_loss"]=norm2(off)
        scores[f"{pre}_offdiag_l2sq_loss"]=float(off@off)
        scores[f"{pre}_offdiag_over_total_loss"]=float(2*off@off/max(float(np.sum(A*A)),EPS))
        scores[f"{pre}_offdiag_over_diag_loss"]=float(2*off@off/max(float(diagonal@diagonal),EPS))
        _summarize_vectors(
            [A[i,np.arange(d)!=i] for i in range(d)],
            f"{pre}_row_off",rz,scores,threshold)
        pre=f"zx_{key}"
        scores[f"{pre}_total_l1_loss"]=float(bb.sum())
        scores[f"{pre}_total_l2_loss"]=norm2(bb)
        _summarize_vectors([B[i,:] for i in range(d)], f"{pre}_row",rx,scores,threshold)
        _summarize_vectors([B[:,j] for j in range(B.shape[1])],
                           f"{pre}_column",rx,scores,threshold)
    # Spectral / Gram: keep separate semantics from entrywise RMS.
    for block,field in (("zz",zz),("zx",zx)):
        gram=np.einsum("nab,ncb->ac",field,field,optimize=True)/len(field)
        ov=gram-np.diag(np.diag(gram))
        scores[f"{block}_gram_offdiag_l1_loss"]=float(np.sum(np.abs(ov)))
        scores[f"{block}_gram_offdiag_l2sq_loss"]=float(np.sum(ov*ov))
        ev=np.linalg.eigvalsh(gram)
        scores[f"{block}_gram_spectral_entropy_diagnostic"]=float(
            -np.sum((ev/ev.sum())*np.log(np.maximum(ev/ev.sum(),EPS))) if ev.sum()>EPS else 0.)
        scores[f"{block}_gram_effective_rank_diagnostic"]=float(
            np.exp(scores[f"{block}_gram_spectral_entropy_diagnostic"]))
        scores[f"{block}_gram_trace_diagnostic"]=float(np.trace(gram))
    ev=np.linalg.eigvalsh(zz)
    absop=np.einsum("nij,nj,nkj->ik",*(),optimize=True) if False else None
    # Matrix spectral absolute, via eigenvectors/eigenvalues at each W.
    eig, eigvec=np.linalg.eigh(zz)
    specabs=np.einsum("nij,nj,nkj->ik",eigvec,np.abs(eig),eigvec,optimize=True)/len(zz)
    ospec=specabs-np.diag(np.diag(specabs))
    scores["zz_spectral_absolute_offdiag_l1_loss"]=float(np.sum(np.abs(ospec)))
    scores["zz_spectral_absolute_offdiag_l2sq_loss"]=float(np.sum(ospec**2))
    # Covariance of unique symmetric Hessian elements: not a n x n matrix.
    iu=np.triu_indices(d)
    flat=zz[:,iu[0],iu[1]]
    cv=np.cov(flat,rowvar=False,bias=True)
    scores["zz_hessian_element_covariance_offdiag_l1_diagnostic"]=float(
        np.sum(np.abs(cv-np.diag(np.diag(cv)))))
    scores["zz_hessian_element_covariance_trace_diagnostic"]=float(np.trace(cv))
    if higher:
        for key,arr in higher.items():
            a=np.asarray(arr,float)
            if not np.isfinite(a).all():
                continue
            scores[f"{key}_rms_loss"]=float(np.sqrt(np.mean(a*a)))
            scores[f"{key}_mean_abs_loss"]=float(np.mean(np.abs(a)))
            scores[f"{key}_q90_abs_loss"]=float(np.quantile(np.abs(a),.90))
            scores[f"{key}_max_abs_loss"]=float(np.max(np.abs(a)))
            scores[f"{key}_signed_mean_diagnostic"]=float(np.mean(a))
    if not all(np.isfinite(v) for v in scores.values()):
        bad=[k for k,v in scores.items() if not np.isfinite(v)]
        raise ArithmeticError(f"Invalid metrics: {bad[:5]}")
    return scores


def metric_descriptor(name):
    if name.endswith("_diagnostic"):
        direction="diagnostic_only"
    else:
        direction="minimize_candidate"
    if "tail" in name or "next_k" in name:
        family="best_k_term_support_residual"
    elif any(t in name for t in ("softl0","capped","logsum","support","lp0")):
        family="sparse_support_nonlinear"
    elif any(t in name for t in ("hoyer","shannon","renyi","participation","gini")):
        family="concentration_entropy"
    elif "gram" in name or "spectral" in name or "covariance" in name:
        family="matrix_operator"
    elif "third" in name or "fourth" in name:
        family="higher_order_curvature"
    elif "localfirst" in name:
        family="local_first"
    elif "diag" in name or "self" in name:
        family="diagonal_vs_interaction"
    else:
        family="hessian_magnitude"
    return dict(metric=name,family=family,direction=direction,
                interpretation="Candidate minimum loss, not a proven theory-axis objective"
                if direction!="diagnostic_only" else "Description/invariance control; do not optimize blindly")
