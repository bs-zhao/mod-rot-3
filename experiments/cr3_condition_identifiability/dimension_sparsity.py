"""Universal, condition-stratified latent DIMENSION sparsity objectives.

No CR3 parameter-name conventions, special null directions, or theoretical
active-dimension counts enter the loss. The SAME L is applied to every trial
category, then aggregated using one common orthogonal R.

Uses *only* analytic derivative second moments from the existing A/B/C
experiment. Thus each candidate rotation costs small matrix products, not
recomputation of a decoder, AD, or boundary-crossing X derivatives.

Compare:
  stratified : sum_s pi_s L(q_s(R))
  pooled     : L(q_from_sum_s_pi_s_moments(R))
These differ because L is nonlinear. Both use the same generic L.
"""
from __future__ import annotations

import numpy as np

COMPONENTS = {
    "U": ("U",),
    "U_UU": ("U", "UU"),
    "U_UU_UX": ("U", "UU", "UX"),
}
AGGREGATIONS = ("stratified", "pooled")
METRICS = ("lp", "auto_topk")


def prepare(stats, ux_kind="tangent"):
    """Return per-condition 4x4 PSD row-energy matrices for each derivative family.

    For UU, M[a,b] = sum_j E[H[a,j]*H[b,j]]: consequently
    diag(R M R.T)[i] = sum_j E[(R H R.T)[i,j]**2].
    For UX, M = sum_j support[s,j] * E[UX[:,j] UX[:,j]^T | valid_j].
    Undefined zero-amount X derivatives are not imputed as zero: they were
    excluded from their *conditional* covariance and support is explicit.
    UX uses only condition-tangent directions unless another explicit
    source kind is selected.
    """
    counts = np.asarray(stats["counts"], dtype=int)
    G = np.asarray(stats["G_cov"], dtype=float)
    H4 = np.asarray(stats["H4"], dtype=float)
    if G.shape != (4, 4, 4) or H4.shape != (4, 4, 4, 4, 4):
        raise ValueError("Expect exactly 4 disjoint CR3 conditions and 4 U dimensions")
    if np.any(counts <= 0):
        raise ValueError("Each condition must have trials")
    if ux_kind not in stats["C_cov"] or ux_kind not in stats["C_support"]:
        raise KeyError(f"Missing UX support/second moments: {ux_kind}")
    C = np.asarray(stats["C_cov"][ux_kind], dtype=float)
    support = np.asarray(stats["C_support"][ux_kind], dtype=float)
    if C.shape != (4, 6, 4, 4) or support.shape != (4, 6):
        raise ValueError("UX moment shape mismatch")
    if not (np.isfinite(G).all() and np.isfinite(H4).all() and
            np.isfinite(C).all() and np.isfinite(support).all()):
        raise ValueError("Non-finite moment")
    if np.any((support < 0) | (support > 1)):
        raise ValueError("UX supports must lie in [0,1]")
    M = {
        "U": G,
        "UU": np.einsum("sakbk->sab", H4, optimize=True),
        "UX": np.einsum("sj,sjab->sab", support, C, optimize=True),
    }
    for name, tensor in M.items():
        if np.max(np.abs(tensor - tensor.transpose(0, 2, 1))) > 1e-8:
            raise ValueError(f"{name} moment matrix not symmetric")
        if np.min(np.linalg.eigvalsh(tensor)) < -1e-8 * max(1, np.max(abs(tensor))):
            raise ValueError(f"{name} moment matrix is not PSD")
        M[name] = (tensor + tensor.transpose(0, 2, 1))/2
    return {"components":M, "counts":counts, "ux_kind":ux_kind,
            "channel":str(stats["channel"]), "x_scale":str(stats["x_scale"]),
            "supports":support}


def mixing_weights(counts, balance=0.0):
    """Convex interpolation empirical -> four conditions balanced.

    balance=0: original empirical 60:60:40:340 (in bank order)
    balance=1: all four conditions have weight 1/4.
    No dependence on theta labels or candidate R.
    """
    if not 0 <= balance <= 1:
        raise ValueError("balance must be in [0,1]")
    counts = np.asarray(counts, dtype=float)
    if len(counts)!=4 or np.any(counts<=0):
        raise ValueError("four nonempty trial groups required")
    return (1-balance)*counts/counts.sum()+balance*np.ones(4)/4


def _validate_rotation(R):
    R=np.asarray(R,dtype=float)
    if R.shape!=(4,4) or not np.isfinite(R).all() or not np.allclose(R@R.T,np.eye(4),atol=1e-6):
        raise ValueError("Expected orthogonal 4x4 R")
    return R


def _normalized_axis_energy(M, R):
    """One PSD family -> nonnegative 4-vector, normalized independently."""
    v = np.einsum("ia,ab,ib->i", R, M, R, optimize=True)
    v = np.maximum(v, 0)
    total = float(v.sum())
    # All-zero family adds no information, and must not be treated as an
    # arbitrary uniform vector nor assigned a misleading 'zero axis'.
    if total <= 1e-15:
        return None
    return v/total


def _group_q(comps, names, R):
    vectors = [_normalized_axis_energy(comps[name],R) for name in names]
    active = [q for q in vectors if q is not None]
    if not active:
        return None, []
    q = np.mean(np.stack(active, axis=0), axis=0)
    q = np.maximum(q, 0)
    q = q/q.sum()
    return q, [name for name,v in zip(names,vectors) if v is not None]


def dimension_loss(q, metric="lp", power=.5, complexity_penalty=.3):
    """Minimal scalar L(q), identical for all trial types.

    lp: sum_i q_i^power, 0<power<1, sparsity preferred.
    auto_topk: min_{1<=k<=d} [sum_(i>k) q_(i) + penalty*k/d],
    where q is sorted descending. k is determined by the data, never
    hardcoded from CR3's known parameter structure.
    """
    if not 0<power<1:
        raise ValueError("power must lie in (0,1)")
    if not complexity_penalty>0:
        raise ValueError("complexity_penalty must be positive")
    if metric not in METRICS:
        raise ValueError(f"Unknown dimension metric {metric}")
    if q is None:
        return {"loss":0., "effective_dims":0., "selected_k":0,
                "drop_one_fraction":0., "q":None}
    q=np.asarray(q,dtype=float)
    if np.min(q)<-1e-10 or abs(np.sum(q)-1)>1e-7:
        raise ValueError("Energy proportions must be nonnegative and sum to 1")
    q=np.maximum(q,0)
    q=q/q.sum()
    sorted_q=np.sort(q)[::-1]
    entropy=-float(np.sum(q[q>0]*np.log(q[q>0])))
    if metric=="lp":
        score=float(np.sum(q**power))
        selected_k=None
    else:
        costs=np.array([float(sum(sorted_q[k:]))+complexity_penalty*k/len(q)
                        for k in range(1,len(q)+1)])
        selected_k=int(np.argmin(costs)+1)
        score=float(costs[selected_k-1])
    return {
        "loss":score,"selected_k":selected_k,"effective_dims":float(np.exp(entropy)),
        "drop_one_fraction":float(np.min(q)),"q":q.tolist(),
    }


def evaluate(M, R, recipe="U", metric="lp", balance=0., aggregation="stratified",
             power=.5, complexity_penalty=.3):
    """Return scalar loss and full category decomposition.

    The same R and *exactly the same* criterion and tuning parameters
    are used for every condition. For pooled form, moment matrices are
    combined before the nonlinear sparsity loss.
    """
    R=_validate_rotation(R)
    if recipe not in COMPONENTS:
        raise ValueError(recipe)
    if aggregation not in AGGREGATIONS:
        raise ValueError(aggregation)
    weights=mixing_weights(M["counts"],balance)
    names=COMPONENTS[recipe]
    by_condition=[]
    for s in range(4):
        comps={name:M["components"][name][s] for name in names}
        q,active=_group_q(comps,names,R)
        info=dimension_loss(q,metric,power,complexity_penalty)
        by_condition.append({"group_index":s,"weight":float(weights[s]),
                             "active_components":active,**info})
    stratified=float(sum(item["weight"]*item["loss"] for item in by_condition))
    pooled_moments={
        name:np.einsum("s,sab->ab",weights,M["components"][name],optimize=True)
        for name in names
    }
    q_pooled,active_pooled=_group_q(pooled_moments,names,R)
    pooled_info=dimension_loss(q_pooled,metric,power,complexity_penalty)
    pooled_info["active_components"]=active_pooled
    pooled=float(pooled_info["loss"])
    return {"score":stratified if aggregation=="stratified" else pooled,
            "stratified_score":stratified,"pooled_score":pooled,
            "by_condition":by_condition,"pooled":pooled_info,
            "mixture_weights":weights.tolist(),"recipe":recipe,
            "metric":metric,"aggregation":aggregation,
            "power":power,"complexity_penalty":complexity_penalty}
