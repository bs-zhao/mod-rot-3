"""Uniform SECOND-DERIVATIVE structural simplicity loss for SO(d).

No parameter roles, CPT formula, theta, model names, output-gain tests,
or supervised axis labels are used. The exact same formula applies to all
three models, for both 3D and 4D.

A key subtlety: total squared Frobenius norms are invariant to R. To
actually select an orientation, minimize L1 of *element-wise RMS* Hessian
couplings (a smooth approximation to sparsity of the derivative tensor).
Each objective is normalized by the corresponding invariant Frobenius
energy and number of matrix elements; losses are dimensionless.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class MomentGroup:
    name: str
    trials: int
    uu: np.ndarray
    ux: np.ndarray | None
    uu_norm: float
    ux_norm: float | None


def summarize_fields(fields,grouping="pooled",include_ux=True):
    H=np.asarray(fields["UU"],dtype=np.float64)
    J=np.asarray(fields["UX"],dtype=np.float64) if fields["UX"] is not None else None
    code=np.asarray(fields["group_codes"],dtype=int)
    if H.ndim!=4 or H.shape[-1]!=H.shape[-2] or H.shape[1]!=len(code):
        raise ValueError("UU must have shape (subjects,trials,d,d)")
    if include_ux and (J is None or J.shape!=(*H.shape[:2],H.shape[-1],6)):
        raise ValueError("UX required for joint/ux_only mode; expected (subjects,trials,d,6)")
    if grouping not in ("pooled","balanced"):
        raise ValueError("grouping must be pooled or balanced")
    if grouping=="pooled":
        groups=[("pooled",np.ones(len(code),dtype=bool))]
    else:
        groups=[(f"condition_{i}",code==i) for i in sorted(np.unique(code))]
    result=[]
    for name,select in groups:
        if not np.any(select):
            raise ValueError(f"Empty trial condition {name}")
        h=H[:,select].reshape(-1,H.shape[-1],H.shape[-1])
        moments_uu=np.einsum("nab,ncd->abcd",h,h,optimize=True)/len(h)
        norm_uu=float(np.einsum("abab->",moments_uu))
        if norm_uu<1e-16:
            raise ValueError(f"{name} has effectively zero UU signal; no identifiable Hessian objective")
        if include_ux:
            j=J[:,select].reshape(-1,H.shape[-1],6)
            moments_ux=np.einsum("naj,nbj->abj",j,j,optimize=True)/len(j)
            norm_ux=float(np.einsum("aaj->",moments_ux))
            if norm_ux<1e-16:
                raise ValueError(f"{name} has effectively zero UX signal")
        else:
            moments_ux=None
            norm_ux=None
        result.append(MomentGroup(name,int(np.sum(select)),moments_uu,
                                  moments_ux,norm_uu,norm_ux))
    return result


class HessianLoss:
    def __init__(self,groups,mode="joint",uu_weight=1.,ux_weight=1.,
                 epsilon=1e-8):
        if not groups:
            raise ValueError("No Hessian moment groups")
        if mode not in ("joint","uu_only","ux_only"):
            raise ValueError("mode must be joint, uu_only or ux_only")
        if epsilon<=0 or uu_weight<=0 or ux_weight<=0:
            raise ValueError("epsilon and weights must be positive")
        self.groups=list(groups)
        self.mode=mode
        self.uw=float(uu_weight)
        self.xw=float(ux_weight)
        self.eps=float(epsilon)
        self.dim=self.groups[0].uu.shape[0]
        self.tensors=[]
        for group in self.groups:
            if group.uu.shape!=(self.dim,)*4:
                raise ValueError("Inconsistent UU moment shape")
            if mode!="uu_only" and group.ux is None:
                raise ValueError("UX moments missing for requested objective")
            self.tensors.append((
                torch.as_tensor(group.uu,dtype=torch.float64),
                torch.as_tensor(group.ux,dtype=torch.float64) if group.ux is not None else None,
                float(group.uu_norm),group.ux_norm))

    def torch_terms(self,R):
        dim=self.dim
        out=[]
        for C,D,norm_uu,norm_ux in self.tensors:
            # M_{ij} = E[(R H R^T)_{ij}^2]
            e_uu=torch.einsum("ia,jb,ic,jd,abcd->ij",R,R,R,R,C)
            e_uu=torch.clamp(e_uu,min=0.)
            loss_uu=torch.sqrt(e_uu+self.eps*norm_uu/(dim*dim)).sum() / (
                dim*np.sqrt(norm_uu))
            if self.mode!="uu_only":
                # E[(R J)_{ij}^2], J=H_UX; 6 original X columns remain fixed.
                e_ux=torch.einsum("ia,ib,abj->ij",R,R,D)
                e_ux=torch.clamp(e_ux,min=0.)
                loss_ux=torch.sqrt(e_ux+self.eps*norm_ux/(dim*6)).sum() / (
                    np.sqrt(dim*6)*np.sqrt(norm_ux))
            else:
                loss_ux=None
            if self.mode=="joint":
                val=(self.uw*loss_uu+self.xw*loss_ux)/(self.uw+self.xw)
            elif self.mode=="uu_only":
                val=loss_uu
            else:
                val=loss_ux
            out.append((val,loss_uu,loss_ux))
        value=torch.stack([i[0] for i in out]).mean()
        return value,out

    def scalar(self,R):
        rot=torch.as_tensor(R,dtype=torch.float64)
        value,terms=self.torch_terms(rot)
        def number(x):return None if x is None else float(x.detach().cpu())
        return dict(total=number(value),mode=self.mode,grouping="equal_weight_per_group",
                    by_group=[dict(name=g.name,trials=g.trials,
                                   loss=number(v),uu=number(u),ux=number(x))
                              for g,(v,u,x) in zip(self.groups,terms)])

    def value(self,R):
        return float(self.scalar(R)["total"])
