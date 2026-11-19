"""Stream quadrature-weighted frozen Decoder UU/UX derivative SECOND moments.

Only artificial Gaussian q nodes and X integration nodes enter decoder AD.
All 5000 participants are absent. Their Z are used once in data.fit_gaussian
to compute mu and Sigma. We store tiny d^4 and d*d*6 moments, not a giant
Gaussian-node by trial Hessian array.

Weights are probability weights, not unweighted grid averages. FIT and
VALID evaluations receive their OWN independent quadrature nodes.
"""
from __future__ import annotations
import numpy as np
import torch
from .teacher import logit


def derivative_chunk(model,z,x,B,include_ux):
    """z: (Q,native_dim), x: (Q,X,6), B: d_native x d_q.

    Symmetric second derivative w.r.t. q holding physical-X fixed.
    For UX, X derivative is with respect to S3-standardized decoder X.
    """
    dim=B.shape[1]
    directions=[B[:,i].expand_as(z) for i in range(dim)]
    with torch.enable_grad():
        def out(z_,x_):return logit(model,z_,x_)
        hs=[];js=[]
        for i in range(dim):
            def d_i(z_,x_,i=i):
                return torch.func.jvp(lambda zz:out(zz,x_),
                                      (z_,),(directions[i],))[1]
            hs.append(torch.stack([
                torch.func.jvp(lambda zz:d_i(zz,x),
                               (z,),(directions[j],))[1]
                for j in range(dim)],dim=-1))
            if include_ux:
                je=[]
                for j in range(6):
                    v=torch.zeros_like(x)
                    v[...,j]=1.
                    je.append(torch.func.jvp(
                        lambda xx:d_i(z,xx),(x,),(v,))[1])
                js.append(torch.stack(je,dim=-1))
        h=torch.stack(hs,dim=-2)
        h=(h+h.transpose(-1,-2))/2
        j=torch.stack(js,dim=-2) if include_ux else None
    return (h.detach().cpu().numpy().astype(np.float64),
            None if j is None else j.detach().cpu().numpy().astype(np.float64))


def accumulate(model,device,source,q,qweights,x,xweights,groups,
               *,grouping="pooled",derivatives="uu_only",
               z_batch=4,x_batch=32,progress=None):
    q=np.asarray(q,np.float64);wq=np.asarray(qweights,np.float64)
    x=np.asarray(x,np.float64);wx=np.asarray(xweights,np.float64)
    groups=np.asarray(groups,int)
    d=source["B"].shape[1]
    if q.ndim!=2 or q.shape[1]!=d or x.ndim!=2 or x.shape[1]!=6:
        raise ValueError("Gaussian q / physical X dimension mismatch")
    if wq.shape!=(len(q),) or wx.shape!=(len(x),) or groups.shape!=(len(x),):
        raise ValueError("Quadrature weights/groups dimension mismatch")
    if not np.isclose(wq.sum(),1,atol=1e-9) or not np.isclose(wx.sum(),1,atol=1e-9):
        raise ValueError("Quadrature weights must sum to one")
    if np.any(wq<=0) or np.any(wx<=0):
        raise ValueError("Quadrature weights must be positive")
    if derivatives not in ("uu_only","joint","ux_only"):
        raise ValueError("Unknown derivative mode")
    include_ux=derivatives!="uu_only"
    if grouping not in ("pooled","balanced"):
        raise ValueError("Unknown grouping")
    if int(z_batch)<1 or int(x_batch)<1:raise ValueError("Batch sizes must be >=1")
    if grouping=="balanced" and len(np.unique(groups))>1:
        names=[(f"group_{code}",int(code)) for code in np.unique(groups)]
    else:
        names=[("pooled",None)]
    result={}
    for name,g in names:
        mass=float(wx.sum() if g is None else wx[groups==g].sum())
        if mass<=0:raise ValueError("Group has zero X measure")
        result[name]=dict(
            name=name,code=g,probability_mass=mass,
            UU=np.zeros((d,d,d,d),dtype=np.float64),
            UX=np.zeros((d,d,6),dtype=np.float64) if include_ux else None,
            q_points=len(q),
            x_points=len(x) if g is None else int(np.sum(groups==g)))
    z_mu=torch.tensor(source["mu"],device=device,dtype=torch.float32)
    B=torch.tensor(source["B"],device=device,dtype=torch.float32)
    x_mu=torch.tensor(source["x_mu"],device=device,dtype=torch.float32)
    x_sd=torch.tensor(source["x_sd"],device=device,dtype=torch.float32)
    x_std=torch.as_tensor(x,dtype=torch.float32,device=device)
    x_std=(x_std-x_mu)/x_sd
    for a in range(0,len(q),int(z_batch)):
        stop=min(a+int(z_batch),len(q))
        qs=torch.as_tensor(q[a:stop],dtype=torch.float32,device=device)
        zz=z_mu+qs@B.T
        for b in range(0,len(x),int(x_batch)):
            stopx=min(b+int(x_batch),len(x))
            xx=x_std[b:stopx].unsqueeze(0).expand(len(zz),-1,-1).clone()
            H,J=derivative_chunk(model,zz,xx,B,include_ux)
            if not np.isfinite(H).all() or (J is not None and not np.isfinite(J).all()):
                raise RuntimeError(f"Nonfinite NN derivatives at q[{a}:{stop}], X[{b}:{stopx}]")
            for item in result.values():
                mask=(np.ones(stopx-b,dtype=bool) if item["code"] is None
                      else groups[b:stopx]==item["code"])
                if not np.any(mask):continue
                weight_x=wx[b:stopx][mask]/item["probability_mass"]
                item["UU"]+=np.einsum(
                    "p,t,ptij,ptkl->ijkl",wq[a:stop],weight_x,
                    H[:,mask],H[:,mask],optimize=True)
                if include_ux:
                    item["UX"]+=np.einsum(
                        "p,t,ptij,ptkj->ikj",wq[a:stop],weight_x,
                        J[:,mask],J[:,mask],optimize=True)
        if progress is not None:
            progress(stop,len(q))
    output=[]
    for item in result.values():
        uu=item["UU"]; ux=item["UX"]
        unorm=float(np.einsum("ijij->",uu))
        xnorm=(float(np.einsum("iij->",ux)) if ux is not None else None)
        if unorm<1e-16 or not np.isfinite(unorm):
            raise ValueError("Near-zero Gaussian UU Hessian energy; structural rotation unidentifiable")
        if xnorm is not None and (xnorm<1e-16 or not np.isfinite(xnorm)):
            raise ValueError("Near-zero Gaussian UX Hessian energy")
        output.append({**item,"UU_norm":unorm,"UX_norm":xnorm})
    return output


def save_moments(path,groups):
    """Compact NPZ: only tensor integrals, not q*x individual derivatives."""
    arrays={"group_names":np.array([g["name"] for g in groups]),
            "group_x_points":np.array([g["x_points"] for g in groups]),
            "group_mass":np.array([g["probability_mass"] for g in groups]),
            "uu_norms":np.array([g["UU_norm"] for g in groups]),
            "uu_moments":np.stack([g["UU"] for g in groups])}
    if groups[0]["UX"] is not None:
        arrays["ux_norms"]=np.array([g["UX_norm"] for g in groups])
        arrays["ux_moments"]=np.stack([g["UX"] for g in groups])
    np.savez_compressed(path,**arrays)


def load_moments(path):
    with np.load(path,allow_pickle=False) as a:
        groups=[]
        for i,name in enumerate(a["group_names"]):
            ux=np.asarray(a["ux_moments"][i]) if "ux_moments" in a.files else None
            groups.append(dict(
                name=str(name),code=None,
                x_points=int(a["group_x_points"][i]),
                probability_mass=float(a["group_mass"][i]),
                UU=np.asarray(a["uu_moments"][i]),
                UX=ux,UU_norm=float(a["uu_norms"][i]),
                UX_norm=float(a["ux_norms"][i]) if ux is not None else None))
    return groups
