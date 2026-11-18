"""Frozen-decoder exact AD Hessians with X coordinates standardized at S3.

H_UU is computed at the original bank; H_UX uses derivatives wrt the
decoder's standardized X. The latter is finite for the smooth NN even
where CR3/5's original CPT formula has an X endpoint singularity. Never
reinterpret the NN's boundary Hessian as an analytic CPT truth derivative.
"""
from __future__ import annotations
import numpy as np
import torch
from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig,DiscreteNeuralProcess
from modlearn_rot.training.continuous import load_checkpoint
from .data import sha256


def restore(source,device="auto"):
    ckpt=load_checkpoint(source["checkpoint"])
    pre=ckpt.get("extra_state",{}).get("decoder_x_standardization")
    if pre is None:
        raise RuntimeError("Use a NEW X-standardized CR3 checkpoint (_xstd), or CR4/CR5 S3. Old raw-X CR3 is not compatible.")
    if pre.get("bank_sha256")!=source["hashes"]["bank"]:
        raise RuntimeError("S3 decoder X scaler belongs to a different stimulus bank")
    mu=np.asarray(pre.get("mean"),dtype=float)
    sd=np.asarray(pre.get("sd"),dtype=float)
    if mu.shape!=(6,) or sd.shape!=(6,) or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sd)) or np.any(sd<=0):
        raise RuntimeError("Missing/invalid persisted standardization statistics")
    cfg=DiscreteNPConfig(**ckpt["model_config"])
    if cfg.latent_dim!=source["z"].shape[1] or cfg.x_dim!=6 or cfg.num_choices!=2:
        raise RuntimeError("Decoder dimensionality differs from S5 latent")
    dev=torch.device("cuda" if device=="auto" and torch.cuda.is_available()
                     else "cpu" if device=="auto" else device)
    model=DiscreteNeuralProcess(cfg).to(dev)
    model.load_state_dict(ckpt["model_state_dict"],strict=True)
    model.eval().requires_grad_(False)
    return model,dev,mu,sd


def logit(model,z,x):
    y=model.decode(x,z)
    return y[...,1]-y[...,0]


def extract(model,device,source,subjects,trial_ids,mu,sd,
            include_ux=True,batch_size=4):
    """Returns one flattened pair of arrays, not repeated per model-specific role.

    Every returned derivative component uses TRAIN-PCA whitened q, and
    standardized X. Full original-Z residual is held fixed under q changes.
    """
    subjects=np.asarray(subjects,dtype=int)
    trial_ids=np.asarray(trial_ids,dtype=int)
    z=np.asarray(source["z"][subjects],dtype=np.float64)
    x=(source["bank_x"][trial_ids]-mu)/sd
    tangent=torch.as_tensor(source["B"].T,dtype=torch.float32,device=device)
    xbase=torch.as_tensor(x,dtype=torch.float32,device=device)
    k=tangent.shape[0]
    eta_values,uu_values,ux_values=[],[],[]
    for first in range(0,len(z),int(batch_size)):
        zz=torch.as_tensor(z[first:first+batch_size],dtype=torch.float32,device=device)
        # clone avoids the aliased expanded view with nested X-forward AD.
        xx=xbase.unsqueeze(0).expand(len(zz),-1,-1).clone()
        directions=[v.expand_as(zz) for v in tangent]
        def value(zarg,xarg):
            return logit(model,zarg,xarg)
        with torch.enable_grad():
            eta=value(zz,xx)
            hs=[]
            js=[]
            for i in range(k):
                def d_i(zarg,xarg,t= directions[i]):
                    return torch.func.jvp(
                        lambda z2:value(z2,xarg),(zarg,),(t,))[1]
                hs.append(torch.stack([
                    torch.func.jvp(lambda z2:d_i(z2,xx),
                                   (zz,),(directions[j],))[1]
                    for j in range(k)],dim=-1))
                if include_ux:
                    js.append(torch.stack([
                        torch.func.jvp(
                            lambda xarg:d_i(zz,xarg),
                            (xx,),(torch.nn.functional.one_hot(
                                torch.tensor(j,device=device),num_classes=6
                            ).to(dtype=xx.dtype).view(1,1,6).expand_as(xx),))[1]
                        for j in range(6)],dim=-1))
            H=torch.stack(hs,dim=-2)
            H=(H+H.transpose(-1,-2))*0.5
            eta_values.append(eta.detach().cpu().numpy())
            uu_values.append(H.detach().cpu().numpy())
            if include_ux:
                ux_values.append(torch.stack(js,dim=-2).detach().cpu().numpy())
    eta=np.concatenate(eta_values).astype(np.float64)
    h=np.concatenate(uu_values).astype(np.float64)
    j=np.concatenate(ux_values).astype(np.float64) if include_ux else None
    if not np.isfinite(eta).all() or not np.isfinite(h).all() or (j is not None and not np.isfinite(j).all()):
        raise RuntimeError("Non-finite frozen Decoder second-derivative field")
    return {"eta":eta,"UU":h,"UX":j,"subject_ids":subjects,
            "x_ids":trial_ids,"group_codes":source["group_codes"][trial_ids]}
