"""SO(K) multistart optimizer for effect-fingerprint rank alone."""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from .metrics import FingerprintObjective


def as_rotation(angles,k):
    angles=np.asarray(angles,float)
    if angles.shape!=(k*(k-1)//2,):
        raise ValueError("Incorrect Givens-generator parameter vector length")
    A=np.zeros((k,k))
    i,j=np.triu_indices(k,1)
    A[i,j]=angles
    A[j,i]=-angles
    return expm(A)


def optimize(proxy,config,anchor=None):
    obj=FingerprintObjective(proxy,config,anchor=anchor)
    k=proxy.k
    n=k*(k-1)//2
    starts=int(config["starts"]);maxiter=int(config["maxiter"])
    if starts<1 or maxiter<1:
        raise ValueError("starts and maxiter must be >=1")
    algorithm=str(config.get("optimizer","Powell"))
    if algorithm not in ("Powell","BFGS","Nelder-Mead"):
        raise ValueError("Unsupported optimizer")
    rng=np.random.default_rng(int(config["seed"]))
    runs=[]
    for i in range(starts):
        x0=np.zeros(n) if i==0 else rng.normal(0,.45,n)
        fitted=minimize(lambda v:obj.value(as_rotation(v,k)),x0,
                        method=algorithm,options={"maxiter":maxiter})
        if np.isfinite(fitted.fun):
            runs.append((float(fitted.fun),as_rotation(fitted.x,k),fitted))
    if not runs:
        raise RuntimeError("All rotation starts failed")
    cost,R,solution=min(runs,key=lambda x:x[0])
    report=dict(method="effect_fingerprint",rank_metric=obj.metric,
                channel_mode=obj.channel_mode,
                identity=obj.components(np.eye(k)),best=obj.components(R),
                starts=starts,maxiter=maxiter,
                objective_evaluations=obj.evals,
                optimizer=algorithm,success=bool(solution.success),
                message=str(solution.message))
    return R,report,obj
