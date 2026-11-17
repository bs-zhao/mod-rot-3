"""Matrix-free projected derivatives; not a full tensor estimator."""
from __future__ import annotations
import numpy as np
import torch
from torch.func import grad, jvp

def directional_report(h,V,X,seed=17,device="cpu",precision="float64",n_points=32):
    rng=np.random.default_rng(seed)
    dtype=torch.float64 if precision=="float64" else torch.float32
    d=V.shape[1]; p=X.shape[-1];K=X.shape[1] if X.ndim==3 else len(X)
    def full(w): return h(w[:d],w[d:])
    grad_full=grad(full)
    first=[]; second=[]; third=[]
    for _ in range(n_points):
        i=int(rng.integers(0,len(V))); k=int(rng.integers(0,K))
        x=X[i,k] if X.ndim==3 else X[k]
        w=torch.as_tensor(np.r_[V[i],x],device=device,dtype=dtype)
        v=torch.as_tensor(rng.normal(size=d+p),device=device,dtype=dtype)
        v=v/torch.linalg.norm(v)
        with torch.enable_grad():
            g=grad_full(w)
            Hv=jvp(grad_full,(w,),(v,))[1]
            d3=jvp(lambda q: jvp(grad_full,(q,),(v,))[1],(w,),(v,))[1]
        first.append(float(g@v)); second.append(float(Hv@v)); third.append(float(d3@v))
    return {"n_points":n_points,"kind":"W directional contractions",
            "first_mean":float(np.mean(first)),"first_rms":float(np.sqrt(np.mean(np.square(first)))),
            "second_mean":float(np.mean(second)),"second_rms":float(np.sqrt(np.mean(np.square(second)))),
            "third_mean":float(np.mean(third)),"third_rms":float(np.sqrt(np.mean(np.square(third))))}
