"""Controlled one-, two-, and three-order Taylor reconstruction tests."""
from __future__ import annotations
import numpy as np
import torch
from torch.func import jacfwd


def taylor_tests(h, V, X, device="cpu", seed=20261116,
                 n_points=20, radii=(.01,.05,.15), precision="float64"):
    dtype=torch.float64 if precision=="float64" else torch.float32
    rng=np.random.default_rng(seed)
    S,d=V.shape
    K=X.shape[1] if X.ndim==3 else len(X)
    records=[]
    def full(w):
        return h(w[:d],w[d:])
    g=jacfwd(full)
    H=jacfwd(g)
    T=jacfwd(H)
    for direction in ("U","X","W"):
        for size in radii:
            errors={k:[] for k in ("first","second","third","gradient_variation","hessian_variation","factorial_contrast")}
            for _ in range(n_points):
                s=int(rng.integers(0,S)); k=int(rng.integers(0,K))
                x=X[s,k] if X.ndim==3 else X[k]
                w=torch.as_tensor(np.concatenate((V[s],x)),device=device,dtype=dtype)
                direction_vec=rng.normal(size=len(w))
                if direction=="U": direction_vec[d:]=0
                if direction=="X": direction_vec[:d]=0
                direction_vec=direction_vec/(np.linalg.norm(direction_vec)+1e-12)
                z=torch.as_tensor(direction_vec,device=device,dtype=dtype)*size
                with torch.enable_grad():
                    y0=full(w)
                    grad=g(w); hess=H(w); third=T(w)
                    ytarget=full(w+z)
                    grad_target=g(w+z)
                    hess_target=H(w+z)
                linear=y0+grad@z
                quad=linear+.5*torch.einsum("i,ij,j->",z,hess,z)
                cubic=quad+torch.einsum("i,j,k,ijk->",z,z,z,third)/6
                for name,pred in (("first",linear),("second",quad),("third",cubic)):
                    errors[name].append(float((pred-ytarget).detach().abs()))
                # F04/F05: finite-scale controlled change of gradient/Hessian
                # predicted by the next derivative order, respectively.
                errors["gradient_variation"].append(float(
                    torch.linalg.norm((grad_target-grad)-hess@z).detach()))
                errors["hessian_variation"].append(float(
                    torch.linalg.norm((hess_target-hess)-torch.einsum("ijk,k->ij",third,z)).detach()))
                # F06: an exact 2x2 factorial contrast between a U and X axis
                # versus its second-order Taylor interaction approximation.
                ui=int(rng.integers(0,d))
                xi=int(rng.integers(d,len(w)))
                zu=torch.zeros_like(w); zx=torch.zeros_like(w)
                zu[ui]=size; zx[xi]=size
                with torch.no_grad():
                    interaction=full(w+zu+zx)-full(w+zu)-full(w+zx)+y0
                predicted=hess[ui,xi]*size*size
                errors["factorial_contrast"].append(float((interaction-predicted).detach().abs()))
            records.append({"direction":direction,"radius":size,"n_points":n_points,
                            **{f"F0{j}.mae":float(np.mean(errors[name]))
                               for j,name in enumerate(("first","second","third",
                                    "gradient_variation","hessian_variation","factorial_contrast"),start=1)}})
    return records
