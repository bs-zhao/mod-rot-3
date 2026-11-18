"""CR4 UX audit, orthogonal rotations cannot repair stimulus-column energies.

This is a POST-FIT diagnostic, not an extra optimization loss. All X probes
are original fixed CR4 bank points; X differentiations are taken in the
exact standardized decoder coordinates saved by the CR4 training checkpoint.
"""
from __future__ import annotations
import numpy as np
import torch
from .teacher import logodds


def ux_field(model,device,source,subject_ids,x_ids,x_mu,x_sd,batch_subjects=6):
    z=np.asarray(source["z"][subject_ids],dtype=np.float64)
    x=(np.asarray(source["bank"][x_ids],float)-x_mu)/x_sd
    directions=torch.as_tensor(source["B"].T,dtype=torch.float32,device=device)
    xx=torch.as_tensor(x,dtype=torch.float32,device=device)
    rows=[]
    for a in range(0,len(z),batch_subjects):
        zb=torch.as_tensor(z[a:a+batch_subjects],dtype=torch.float32,device=device)
        xb=xx.unsqueeze(0).expand(len(zb),-1,-1)
        tensor=[]
        for i in range(len(directions)):
            tangent=directions[i].expand_as(zb)
            def dU(xv):
                return torch.func.jvp(
                    lambda zz:logodds(model,zz,xv),(zb,),(tangent,)
                )[1]
            cols=[]
            for j in range(xb.shape[-1]):
                basis=torch.zeros_like(xb)
                basis[...,j]=1.
                cols.append(torch.func.jvp(dU,(xb,),(basis,))[1])
            tensor.append(torch.stack(cols,dim=-1))
        rows.append(torch.stack(tensor,dim=-2).detach().cpu().numpy())
    values=np.concatenate(rows).astype(np.float64)
    if not np.isfinite(values).all():raise RuntimeError("Nonfinite CR4 mixed derivatives")
    return values


def ux_diagnostics(field,R):
    old=np.asarray(field,float)
    rotated=np.einsum("ia,sxaj->sxij",R,old,optimize=True)
    e=np.square(rotated).mean(axis=(0,1))
    e0=np.square(old).mean(axis=(0,1))
    xfrac=e.sum(0)/max(float(e.sum()),1e-15)
    xfrac0=e0.sum(0)/max(float(e0.sum()),1e-15)
    if not np.allclose(e.sum(0),e0.sum(0),rtol=2e-6,atol=1e-10):
        raise RuntimeError("Rotated UX stimulus-column energies are not invariant")
    return dict(
        ux_definition="d2 choice_logit / d q_i d X_standardized_j",
        rotated_UX_element_RMS=np.sqrt(e).tolist(),
        identity_UX_element_RMS=np.sqrt(e0).tolist(),
        X_energy_fraction_rotated=xfrac.tolist(),
        X_energy_fraction_identity=xfrac0.tolist(),
        X_energy_invariant_to_R=True,
        probability_columns_fraction=float(xfrac[2]+xfrac[5]),
        x_names=("gain_A","loss_A","p_gain_A","gain_B","loss_B","p_gain_B"),
        warning="UX is diagnostic only; column energies cannot be changed by an orthogonal U rotation")
