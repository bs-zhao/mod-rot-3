"""Universal role-free Gaussian-integrated Hessian RMS sparsity loss.

The only novelty in version 1 is the integration MEASURE (artificial q
Gaussian, fixed or generated X), not a claim to solve the weakness of
global RMS-L1. Both UU-only and optional UX/joint use SAME formula for all
models. Total Frobenius energy is invariant; coordinate elementwise RMS-L1
is not. No parameter labels enter this objective.
"""
from __future__ import annotations
import numpy as np
import torch


class GaussianHessianLoss:
    def __init__(self,groups,mode="uu_only",epsilon=1e-8,
                 uu_weight=1.,ux_weight=1.):
        if not groups:raise ValueError("Need moment groups")
        if mode not in ("uu_only","ux_only","joint"):
            raise ValueError("mode must be uu_only, ux_only, or joint")
        if epsilon<=0 or min(uu_weight,ux_weight)<=0:
            raise ValueError("Need positive regularization and weights")
        self.groups=groups
        self.mode=mode
        self.eps=float(epsilon)
        self.uw=float(uu_weight)
        self.xw=float(ux_weight)
        self.dim=groups[0]["UU"].shape[0]
        self.parts=[]
        for g in groups:
            if g["UU"].shape!=(self.dim,)*4:
                raise ValueError("Invalid UU moment shape")
            if mode!="uu_only" and g["UX"] is None:
                raise ValueError("Mode requires UX moments")
            C=torch.tensor(g["UU"],dtype=torch.float64)
            D=torch.tensor(g["UX"],dtype=torch.float64) if g["UX"] is not None else None
            self.parts.append((C,D,float(g["UU_norm"]),g["UX_norm"]))

    def torch_terms(self,R):
        d=self.dim
        by_group=[]
        for C,D,nu,nx in self.parts:
            # E[(R H R^T)_ij²] via fixed E[H_ab H_cd] tensor
            e_uu=torch.einsum("ia,jb,ic,jd,abcd->ij",
                               R,R,R,R,C).clamp(min=0)
            uu=torch.sqrt(e_uu+self.eps*nu/(d*d)).sum()/(d*np.sqrt(nu))
            if self.mode!="uu_only":
                e_ux=torch.einsum("ia,ib,abj->ij",R,R,D).clamp(min=0)
                ux=torch.sqrt(e_ux+self.eps*nx/(d*6)).sum()/(
                    np.sqrt(d*6)*np.sqrt(nx))
            else:ux=None
            if self.mode=="uu_only":score=uu
            elif self.mode=="ux_only":score=ux
            else:score=(self.uw*uu+self.xw*ux)/(self.uw+self.xw)
            by_group.append((score,uu,ux))
        return torch.stack([g[0] for g in by_group]).mean(),by_group

    def scalar(self,R):
        T=torch.tensor(R,dtype=torch.float64)
        v,details=self.torch_terms(T)
        def num(z):return None if z is None else float(z.detach())
        return dict(total=num(v),mode=self.mode,grouping="equal_across_groups",
                    per_group=[dict(name=g["name"],x_points=g["x_points"],
                                    original_x_mass=g["probability_mass"],
                                    total=num(p),uu=num(u),ux=num(x))
                               for g,(p,u,x) in zip(self.groups,details)])

    def value(self,R):
        return self.scalar(R)["total"]
