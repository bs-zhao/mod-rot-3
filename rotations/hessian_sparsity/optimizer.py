"""Unconstrained-angle parametrization of SO(d), seeded multistart search.

We optimize the FULL d(d-1)/2 Givens angles jointly; there is no
pre-selected role/axis or sequential greedy direction. Derivatives of
the precomputed Hessian-moment objective are exact torch-autograd
derivatives (NOT third derivatives of the decoder).
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.optimize import minimize


def pairs(dim):
    if dim not in (3,4):
        raise ValueError("SO rotation supports d=3 or 4")
    return [(i,j) for i in range(dim) for j in range(i+1,dim)]


def givens(angles,dim):
    if len(angles)!=len(pairs(dim)):
        raise ValueError("Incorrect Givens angle count")
    eye=torch.eye(dim,dtype=angles.dtype,device=angles.device)
    R=eye
    for index,(i,j) in enumerate(pairs(dim)):
        ei=eye[i];ej=eye[j]
        angle=angles[index]
        cs=torch.cos(angle)
        sn=torch.sin(angle)
        G=eye+(cs-1)*(torch.outer(ei,ei)+torch.outer(ej,ej)) + (
            sn*(torch.outer(ei,ej)-torch.outer(ej,ei)))
        R=G@R
    return R


def optimize(fit,valid,config):
    dim=fit.dim
    if dim!=valid.dim:
        raise ValueError("FIT and VALID dimensions mismatch")
    nstarts=int(config["restarts"])
    if nstarts<1:raise ValueError("Need positive multistart budget")
    rng=np.random.default_rng(int(config["seed"])+1712)
    starts=[np.zeros(len(pairs(dim)),dtype=float)]
    starts.extend(rng.uniform(-np.pi,np.pi,size=len(pairs(dim)))
                  for _ in range(nstarts-1))
    best=None
    records=[]
    for idx,x0 in enumerate(starts):
        def fun(angles):
            x=torch.tensor(angles,dtype=torch.float64,requires_grad=True)
            R=givens(x,dim)
            val,_=fit.torch_terms(R)
            grad=torch.autograd.grad(val,x)[0]
            return float(val.detach()),grad.detach().cpu().numpy().astype(np.float64)
        res=minimize(fun,x0,method="L-BFGS-B",jac=True,
                     bounds=[(-4*np.pi,4*np.pi)]*len(x0),
                     options={"maxiter":int(config["maxiter"]),
                              "ftol":1e-12,"gtol":2e-8})
        angles=np.asarray(res.x)
        R=givens(torch.as_tensor(angles,dtype=torch.float64),dim).detach().cpu().numpy()
        score=fit.value(R)
        valid_score=valid.value(R)
        records.append(dict(start=idx,initial_angles=x0.tolist(),
                            angles=angles.tolist(),fit=score,valid=valid_score,
                            scipy_success=bool(res.success),iterations=int(res.nit),
                            message=str(res.message)))
        # Fit-only selection; validation scores are diagnostic and never
        # influence objective choice, start selection or regularization.
        if best is None or score<best["fit"]:
            best=dict(R=R,fit=score,valid=valid_score,start=idx,
                      angles=angles.tolist())
    R=best["R"]
    if not np.allclose(R@R.T,np.eye(dim),atol=1e-9) or abs(np.linalg.det(R)-1.)>1e-9:
        raise RuntimeError("Optimized matrix is not in SO(d)")
    I=np.eye(dim)
    report=dict(
        parameterization="full_SO(d)_Givens_no_role_labels",
        dimension=dim,number_of_angles=len(pairs(dim)),
        best_fit_start=int(best["start"]),
        fit_loss=fit.scalar(R),validation_loss=valid.scalar(R),
        identity_fit_loss=fit.scalar(I),identity_validation_loss=valid.scalar(I),
        fit_improvement=float(fit.value(I)-fit.value(R)),
        valid_improvement=float(valid.value(I)-valid.value(R)),
        candidate_improves_VALID=bool(valid.value(R)<valid.value(I)),
        fit_valid_gap=float(valid.value(R)-fit.value(R)),
        all_starts=records,
        warning="VALID is diagnostic only. TRUE parameters are never used for selection.")
    return R,report
