"""Fixed, T-free structural measurements on an anonymous (U,F) pair.

Every metric returns a LOSS: smaller implies 'simpler'. Control metrics
are expressly NOT candidates for ranking because orthogonal transformations
leave them invariant (within numerical precision).

Linear response fingerprints:
    F_subject ~ intercept + U_subject @ B
Nonlinear response decomposability:
    F_subject ~ polynomial(U_subject), degree <= 3
No use of generator, model identity, R, rotation angle or decoder.
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


EPS = 1e-12
MATRIX_METRICS = (
    "B_column_l1_loss",
    "B_column_hoyer_loss",
    "B_column_entropy_loss",
    "B_column_top1_tail_loss",
    "B_global_l1_over_fro_loss",
    "B_varimax_negative_loss",
    "B_row_overlap_loss",
)
CONTROL_METRICS = ("control_B_fro_norm", "control_B_singular_fro")
FUNCTION_METRICS = (
    "function_add_quad_mse",
    "function_full_quad_mse",
    "function_quad_interaction_gap",
    "function_add_cubic_mse",
    "function_full_cubic_mse",
    "function_cubic_interaction_gap",
    "function_budget_4_mse",
    "function_budget_8_mse",
    "function_budget_12_mse",
    "function_budget_20_mse",
    "function_budget_auc_loss",
    "function_full_cubic_group_l1_loss",
)


def normalize_response(f, train_ids, *, min_rel_sd=1e-3):
    """Standardize F columns with ONLY the fixed TRAIN subject subset."""
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError("Expected a subject by response-feature matrix")
    mu = f[train_ids].mean(axis=0)
    sd = f[train_ids].std(axis=0, ddof=0)
    maximum = float(sd.max()) if len(sd) else 0.
    keep = (sd > max(1e-10, min_rel_sd * maximum))
    if not keep.any():
        return None, {"selected":0,"total":len(sd)}
    values = (f[:,keep] - mu[None,keep]) / sd[None,keep]
    if not np.isfinite(values).all():
        raise ArithmeticError("Non-finite standardized response profile")
    return values, {"selected":int(keep.sum()),"total":int(len(sd))}


def _ridge_centered(u,f,train,ridge):
    uc = u[train].mean(axis=0)
    fc = f[train].mean(axis=0)
    X = u[train]-uc
    Y = f[train]-fc
    b = np.linalg.solve(X.T@X + ridge*np.eye(u.shape[1]), X.T@Y)
    return b, uc, fc


def _fingerprint_scores(b):
    """Normalize each response-feature column before measuring axis occupancy."""
    d,p = b.shape
    energy = np.sum(b*b,axis=0)
    valid = energy > max(EPS, float(np.max(energy, initial=0)) * 1e-9)
    if not valid.any():
        return {k: float("nan") for k in MATRIX_METRICS+CONTROL_METRICS}
    q = b[:,valid] / np.sqrt(energy[valid])[None,:]
    a = np.abs(q)
    l1 = a.sum(axis=0)
    mass = q*q
    top = mass.max(axis=0)
    entropy = -(mass * np.log(np.maximum(mass, EPS))).sum(axis=0)
    # Column normalization makes stimulus-feature scales comparable.
    varimax = float(np.var(mass, axis=1).mean())
    # Pairwise overlap between rows describes how much stimulus effects overlap.
    norm_rows = np.linalg.norm(q,axis=1)
    rnorm = q / np.maximum(norm_rows[:,None], EPS)
    gram = np.abs(rnorm @ rnorm.T)
    off = gram[~np.eye(d,dtype=bool)]
    sv = np.linalg.svd(b,compute_uv=False)
    return {
        "B_column_l1_loss":float(l1.mean()),
        "B_column_hoyer_loss":float(((l1-1)/max(np.sqrt(d)-1,EPS)).mean()),
        "B_column_entropy_loss":float((entropy/max(np.log(d),EPS)).mean()),
        "B_column_top1_tail_loss":float((1-top).mean()),
        "B_global_l1_over_fro_loss":float(
            np.sum(np.abs(b)) / max(float(np.linalg.norm(b)),EPS) /
            np.sqrt(max(p,1))),
        "B_varimax_negative_loss": -varimax,
        "B_row_overlap_loss":float(off.mean()),
        "control_B_fro_norm":float(np.linalg.norm(b)),
        "control_B_singular_fro":float(np.linalg.norm(sv)),
    }


def linear_fingerprint(u, f, train_ids, *, ridge=1e-2, pre_normalized=False):
    """Fit the same isotropic Ridge per orientation; return raw B and metrics."""
    if ridge < 0:
        raise ValueError("Ridge must be nonnegative")
    f_std, meta = (np.asarray(f,float), {"selected":f.shape[1], "total":f.shape[1]})
    if pre_normalized else normalize_response(f,train_ids)
    if f_std is None:
        return None, {"usable":False,**meta}, None
    b, uc, fc = _ridge_centered(np.asarray(u,float), f_std, train_ids,ridge)
    metrics = _fingerprint_scores(b)
    return metrics, {"usable":True,**meta, "fingerprint_shape": list(b.shape)}, b


def _poly(u, train_ids):
    """Training-only axis standardization; signed permutations remain symmetric."""
    u = np.asarray(u,dtype=np.float64)
    mu = u[train_ids].mean(axis=0)
    sd = u[train_ids].std(axis=0,ddof=0)
    if np.any(sd < EPS):
        raise ValueError("Constant latent dimension")
    q = (u-mu)/sd
    poly = PolynomialFeatures(degree=3,include_bias=True)
    design = poly.fit_transform(q)
    powers = poly.powers_
    m = design[train_ids].mean(axis=0)
    s = design[train_ids].std(axis=0, ddof=0)
    s[0] = 1.
    m[0] = 0.
    s = np.maximum(s,1e-10)
    design = (design-m)/s
    if not np.allclose(design[:,0],1.):
        raise AssertionError("Polynomial intercept is not 1")
    return design, powers


def _fit_predict(x,y,train,test,feature_ids,ridge):
    """Ridge on standardized features; no penalty for intercept."""
    Xtr = x[np.ix_(train,feature_ids)]
    Xte = x[np.ix_(test,feature_ids)]
    ytr = y[train]
    gram = Xtr.T@Xtr
    reg = np.eye(len(feature_ids))*ridge
    intercept = np.flatnonzero(feature_ids == 0)
    if len(intercept):
        reg[intercept[0],intercept[0]] = 0.
    coef = np.linalg.solve(gram+reg,Xtr.T@ytr)
    pred = Xte@coef
    mse = float(np.mean((pred-y[test])**2))
    return mse,coef


def functional_scores(u, f, train_ids, test_ids, *, ridge=0.1,
                      max_profile_outputs=64, pre_normalized=False,
                      precomputed_poly=None):
    """Compare additive vs full polynomial response mappings on heldout subjects.

    The complete cubic polynomial spans the same space after rotation;
    any variation in its heldout MSE is a numerical/regularization diagnostic.
    C1: all profile coefficients. CR3-like shared banks: choose a stable,
    uniformly spaced subset of up to 64 observed stimulus positions.
    """
    if ridge <= 0 or max_profile_outputs < 1:
        raise ValueError("Invalid regularization or profile output budget")
    f_std, meta = (np.asarray(f,float), {"selected":f.shape[1], "total":f.shape[1]})
    if pre_normalized else normalize_response(f,train_ids)
    if f_std is None:
        return None, {"usable":False,**meta}
    if f_std.shape[1] > max_profile_outputs:
        selected = np.linspace(0,f_std.shape[1]-1,
                               max_profile_outputs,dtype=int)
        f_std = f_std[:,selected]
    x,powers = (_poly(u,train_ids) if precomputed_poly is None
                else precomputed_poly)
    if x.shape[0] != len(u) or powers.shape[0] != x.shape[1]:
        raise ValueError("Precomputed polynomial design does not match U")
    degree = powers.sum(axis=1)
    n_active = (powers>0).sum(axis=1)
    add_quad = np.flatnonzero((degree<=2)&(n_active<=1))
    full_quad = np.flatnonzero(degree<=2)
    add_cubic = np.flatnonzero(n_active<=1)
    full_cubic = np.arange(len(powers))
    mse_a2,_ = _fit_predict(x,f_std,train_ids,test_ids,add_quad,ridge)
    mse_f2,_ = _fit_predict(x,f_std,train_ids,test_ids,full_quad,ridge)
    mse_a3,_ = _fit_predict(x,f_std,train_ids,test_ids,add_cubic,ridge)
    mse_f3,coef = _fit_predict(x,f_std,train_ids,test_ids,full_cubic,ridge)

    # Fixed budget compression: choose features from TRAIN full-model
    # group coefficient norms only, refit each budget on TRAIN, evaluate
    # on TEST. No supervised T and no use of case angle.
    feature_order = np.argsort(-np.linalg.norm(coef[1:],axis=1),kind="stable")+1
    budgets = [4,8,12,20]
    budget_mse = {}
    for k in budgets:
        selected_features = np.r_[0,feature_order[:min(k,len(feature_order))]]
        loss,_ = _fit_predict(x,f_std,train_ids,test_ids,
                              selected_features, ridge)
        budget_mse[k] = loss
    norms = np.linalg.norm(coef[1:],axis=1)
    group_l1 = float(norms.sum()/max(float(np.linalg.norm(norms)),EPS))
    metrics = {
        "function_add_quad_mse":mse_a2,
        "function_full_quad_mse":mse_f2,
        "function_quad_interaction_gap":mse_a2-mse_f2,
        "function_add_cubic_mse":mse_a3,
        "function_full_cubic_mse":mse_f3,
        "function_cubic_interaction_gap":mse_a3-mse_f3,
        "function_budget_auc_loss":float(np.mean(list(budget_mse.values()))),
        "function_full_cubic_group_l1_loss":group_l1,
    }
    for k,v in budget_mse.items():
        metrics[f"function_budget_{k}_mse"] = v
    return metrics, {
        "usable":True, **meta, "used_profile_columns":int(f_std.shape[1]),
        "n_add_quad_features":int(len(add_quad)),
        "n_full_quad_features":int(len(full_quad)),
        "n_add_cubic_features":int(len(add_cubic)),
        "n_full_cubic_features":int(len(full_cubic)),
        "budgets_n_nonintercept":budgets,
        "feature_selection": "TRAIN full-cubic group coefficient L2; select top k",
        "profile_standardization": "TRAIN only",
    }
