"""Independent numerical and analytic self-consistency checks."""
from __future__ import annotations
import numpy as np
import torch
from torch.func import jacfwd,jacrev
from .derivatives import chain_rule_sigmoid


def _full(h,d):
    return lambda w:h(w[:d],w[d:])


def audit_derivatives(h,V,X,device="cpu",precision="float64",
                      points=3,step=1e-3):
    dtype=torch.float64 if precision=="float64" else torch.float32
    d=V.shape[1]; p=X.shape[-1]
    f=_full(h,d)
    g=jacfwd(f);H=jacfwd(g);T=jacfwd(H)
    H_alt=jacrev(g)
    stats={"symmetry_H":[],"symmetry_T":[],"H_ad_order":[],"H_fd_gradient":[],
           "T_fd_hessian":[],"H_UU_max_abs":[],"T_UUU_max_abs":[],"T_UUX_max_abs":[]}
    for i in range(min(points,len(V))):
        x=X[i%len(X)] if X.ndim==2 else X[i,i%X.shape[1]]
        w=torch.as_tensor(np.r_[V[i],x],dtype=dtype,device=device)
        with torch.enable_grad():
            hh=H(w);tt=T(w)
            h_other=H_alt(w)
            plus_h=[];plus_t=[]
            for j in range(len(w)):
                e=torch.zeros_like(w);e[j]=step
                plus_h.append((g(w+e)-g(w-e))/(2*step))
                plus_t.append((H(w+e)-H(w-e))/(2*step))
        stats["symmetry_H"].append(float((hh-hh.T).abs().max().detach()))
        stats["symmetry_T"].append(float(max(
             (tt-tt.permute(0,2,1)).abs().max().detach(),
             (tt-tt.permute(1,0,2)).abs().max().detach())))
        stats["H_ad_order"].append(float((hh-h_other).abs().max().detach()))
        h_from_fd=torch.stack(plus_h,dim=-1)
        t_from_fd=torch.stack(plus_t,dim=-1)
        stats["H_fd_gradient"].append(float((hh-h_from_fd).abs().max().detach()))
        stats["T_fd_hessian"].append(float((tt-t_from_fd).abs().max().detach()))
        stats["H_UU_max_abs"].append(float(hh[:d,:d].abs().max().detach()))
        stats["T_UUU_max_abs"].append(float(tt[:d,:d,:d].abs().max().detach()))
        stats["T_UUX_max_abs"].append(float(tt[:d,:d,d:].abs().max().detach()))
    return {"n_points":min(points,len(V)),"step":step,
            "max_errors":{name:max(v) if v else None for name,v in stats.items()}}


def audit_probability_chain(logit,prob,V,X,device="cpu",precision="float64",points=2):
    dtype=torch.float64 if precision=="float64" else torch.float32
    d=V.shape[1]
    e=_full(logit,d);p=_full(prob,d)
    gE=jacfwd(e);HE=jacfwd(gE);TE=jacfwd(HE)
    gP=jacfwd(p);HP=jacfwd(gP);TP=jacfwd(HP)
    errors=[]
    for i in range(min(points,len(V))):
        x=X[i%len(X)] if X.ndim==2 else X[i,i%X.shape[1]]
        w=torch.as_tensor(np.r_[V[i],x],dtype=dtype,device=device)
        with torch.enable_grad():
            z=e(w);ge=gE(w);he=HE(w);te=TE(w)
            gp=gP(w);hp=HP(w);tp=TP(w)
        exp=chain_rule_sigmoid(np.asarray(z.detach().cpu()),
                               ge.detach().cpu().numpy(),
                               he.detach().cpu().numpy(),
                               te.detach().cpu().numpy())
        errors.append({"gradient":float(np.max(np.abs(exp[0]-gp.detach().cpu().numpy()))),
                       "hessian":float(np.max(np.abs(exp[1]-hp.detach().cpu().numpy()))),
                       "third":float(np.max(np.abs(exp[2]-tp.detach().cpu().numpy())))})
    return {"points":len(errors),"max_chain_errors":
            {key:max(x[key] for x in errors) if errors else None for key in ("gradient","hessian","third")} }
