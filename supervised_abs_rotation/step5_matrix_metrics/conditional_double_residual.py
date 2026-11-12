"""Crossfitted response residuals for conditional feature covariance diagnostics."""
import numpy as np
import torch

import conditional_ux as conditional


@torch.no_grad()
def regressions(design, target, train, query, linear_count, ridges):
    """GPU/CPU ridge predictions with the affine complement unpenalized."""
    train = torch.as_tensor(train, dtype=torch.long, device=target.device)
    query = torch.as_tensor(query, dtype=torch.long, device=target.device)
    affine, nonlinear = design[:, :linear_count], design[:, linear_count:]
    inverse = torch.linalg.pinv(affine[train], rtol=1e-10)
    correction = inverse @ nonlinear[train]
    residual = nonlinear[train]-affine[train] @ correction
    query_residual = nonlinear[query]-affine[query] @ correction
    affine_coef = inverse @ target[train]
    y = target[train]-affine[train] @ affine_coef
    gram = residual.T @ residual
    values, vectors = torch.linalg.eigh((gram+gram.T)/2)
    values = torch.clamp(values,min=0.)
    scale = max(float(torch.trace(gram))/max(len(gram),1),1e-12)
    rhs = vectors.T @ (residual.T @ y)
    base, left = affine[query] @ affine_coef, query_residual @ vectors
    return [base+left @ (rhs/(values[:,None]+ridge*scale)) for ridge in ridges]


@torch.no_grad()
def residual_targets(z, direction, target, lengthscale, degree=5, centers=128, seed=444, folds=5):
    design, audit = conditional.complement_design(z,direction,degree,centers,lengthscale,seed)
    design = torch.tensor(design,dtype=target.dtype,device=target.device)
    partition = np.array_split(np.random.default_rng(seed+1).permutation(len(z)),folds)
    residual = torch.empty_like(target); history = []
    ridges = (1e-8,1e-6,1e-4,.01,.1)
    for k, query in enumerate(partition):
        train = np.setdiff1d(np.arange(len(z)),query)
        order = np.random.default_rng(seed+2+k).permutation(train)
        cut = int(.75*len(order)); inner, valid = order[:cut],order[cut:]
        predictions = regressions(design,target,inner,valid,audit['linear_count'],ridges)
        normalizer = max(float((target[valid]-target[inner].mean(0)).square().sum()),1e-30)
        errors = [float((p-target[valid]).square().sum())/normalizer for p in predictions]
        chosen = int(np.argmin(errors))
        prediction = regressions(design,target,train,query,audit['linear_count'],(ridges[chosen],))[0]
        residual[query] = target[query]-prediction
        history.append(dict(fold=k,ridge=ridges[chosen],validation_relative_function_mse=errors,
                            heldout_subjects=len(query)))
    residual -= residual.mean(0)
    audit.update(degree=degree,radial_centers=centers,all_subjects=len(z),crossfit_folds=folds,
        history=history,response_tuning='equal-energy X/channel function targets, nested heldout error',
        residual_total_relative_energy=float(residual.square().sum()/target.square().sum().clamp_min(1e-30)))
    return residual,audit
