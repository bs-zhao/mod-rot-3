"""Full simultaneous SO(3)/SO(4) Givens-angle optimization, no role labels."""
from __future__ import annotations
import numpy as np
import torch
from scipy.optimize import minimize


def pairs(d):
    if d not in (3,4):raise ValueError("SO(d) requires d=3 or d=4")
    return [(i,j) for i in range(d) for j in range(i+1,d)]


def givens(angles,d):
    if len(angles)!=len(pairs(d)):
        raise ValueError("Wrong angle vector size")
    E=torch.eye(d,dtype=angles.dtype,device=angles.device)
    R=E
    for k,(i,j) in enumerate(pairs(d)):
        ei,ej=E[i],E[j]
        a=angles[k];c=torch.cos(a);s=torch.sin(a)
        G=E+(c-1)*(torch.outer(ei,ei)+torch.outer(ej,ej))
        G=G+s*(torch.outer(ei,ej)-torch.outer(ej,ei))
        R=G@R
    return R


def optimize(fit,valid,config):
    d=fit.dim
    if valid.dim!=d:raise ValueError("FIT/VALID Gaussian dimension mismatch")
    seed=int(config["seed"])
    n=int(config["restarts"])
    if n<1:raise ValueError("Need at least one restart")
    rng=np.random.default_rng(seed+523)
    angles=len(pairs(d))
    starts=[np.zeros(angles)]+[
        rng.uniform(-np.pi,np.pi,size=angles) for _ in range(n-1)]
    results=[]
    best_R=None;best_loss=float("inf");chosen=-1
    for k,x0 in enumerate(starts):
        def fun(x):
            a=torch.tensor(x,dtype=torch.float64,requires_grad=True)
            R=givens(a,d)
            f,_=fit.torch_terms(R)
            gr=torch.autograd.grad(f,a)[0]
            return float(f.detach()),gr.detach().cpu().numpy()
        sol=minimize(fun,x0,jac=True,method="L-BFGS-B",
                     bounds=[(-4*np.pi,4*np.pi)]*angles,
                     options=dict(maxiter=int(config["maxiter"]),
                                  ftol=1e-12,gtol=1e-8))
        R=givens(torch.tensor(sol.x,dtype=torch.float64),d).detach().cpu().numpy()
        fit_value=fit.value(R)
        valid_value=valid.value(R)
        results.append(dict(start=k,initial=x0.tolist(),
                            solution=np.asarray(sol.x).tolist(),
                            FIT=fit_value,VALID=valid_value,
                            success=bool(sol.success),iterations=int(sol.nit),
                            message=str(sol.message)))
        if fit_value<best_loss:
            best_R=R;best_loss=fit_value;chosen=k
    if not np.allclose(best_R@best_R.T,np.eye(d),atol=1e-9) or not np.isclose(np.linalg.det(best_R),1.,atol=1e-9):
        raise RuntimeError("Optimizer produced a non-SO(d) matrix")
    identity=np.eye(d)
    diag=dict(dimension=d,angles=angles,method="full_SO(d)_Givens_multistart",
              selection="minimum FIT integral simplicity ONLY; no true theta",
              selected_start=chosen,
              optimized_FIT=fit.scalar(best_R),
              optimized_VALID=valid.scalar(best_R),
              identity_FIT=fit.scalar(identity),
              identity_VALID=valid.scalar(identity),
              FIT_gain=fit.value(identity)-fit.value(best_R),
              VALID_gain=valid.value(identity)-valid.value(best_R),
              restarts=results)
    return best_R,diag


def plane_landscapes(R,fit,valid,count=31):
    angles=np.linspace(0,90,int(count))
    records=[]
    d=len(R)
    for i,j in pairs(d):
        for angle in angles:
            theta=np.deg2rad(angle)
            c,s=np.cos(theta),np.sin(theta)
            G=np.eye(d);G[i,i]=c;G[j,j]=c;G[i,j]=s;G[j,i]=-s
            RR=G@R
            records.append(dict(i=i+1,j=j+1,angle_deg=float(angle),
                                FIT=float(fit.value(RR)),VALID=float(valid.value(RR))))
        values=[r["FIT"] for r in records if r["i"]==i+1 and r["j"]==j+1]
        if abs(values[0]-values[-1])>1e-7:
            raise RuntimeError("Hessian loss must be invariant to 90-degree axis permutation")
    return records
