"""Exact frozen-decoder U gradient and U-U Hessian (no X differentiation).

Evaluates at the ORIGINAL CR4 random bank, not synthetic counterfactual X.
Physical X is standardized by the SAME persisted scaler used during S3;
all derivatives are along PCA-whitened latent directions, with the native-Z
residual preserved. Never differentiate the simulation's true parameter.
"""
from __future__ import annotations
import numpy as np
import torch
from .teacher import logodds


def extract(model,device,source,subjects,trial_ids,x_mu,x_sd,batch_subjects=8):
    Z = np.asarray(source["z"][subjects],dtype=np.float64)
    bank = np.asarray(source["bank"][trial_ids],dtype=np.float64)
    xstd = (bank - np.asarray(x_mu)) / np.asarray(x_sd)
    B = np.asarray(source["B"],dtype=np.float32)
    k = B.shape[1]
    if k != 3:
        raise ValueError("Suite is intended for exactly 3 CR4 factors")
    x0 = torch.as_tensor(xstd,dtype=torch.float32,device=device)
    directions = torch.as_tensor(B.T,dtype=torch.float32,device=device)
    values, gradients, hessians = [], [], []
    for start in range(0,len(Z),batch_subjects):
        zb = torch.as_tensor(Z[start:start+batch_subjects],dtype=torch.float32,device=device)
        x = x0.unsqueeze(0).expand(len(zb),-1,-1)
        tangents = [v.expand_as(zb) for v in directions]
        def value(z):
            return logodds(model,z,x)
        with torch.enable_grad():
            eta = value(zb)
            g = []
            h = []
            for i in range(k):
                def derivative_i(z):
                    return torch.func.jvp(value,(z,),(tangents[i],))[1]
                g.append(derivative_i(zb))
                h.append(torch.stack([
                    torch.func.jvp(derivative_i,(zb,),(tangents[j],))[1]
                    for j in range(k)],dim=-1))
            gradients.append(torch.stack(g,dim=-1).detach().cpu().numpy())
            hh = torch.stack(h,dim=-2)
            hessians.append(((hh+hh.transpose(-1,-2))/2).detach().cpu().numpy())
            values.append(eta.detach().cpu().numpy())
    eta = np.concatenate(values).astype(np.float64)
    g = np.concatenate(gradients).astype(np.float64)
    h = np.concatenate(hessians).astype(np.float64)
    if not np.all(np.isfinite(eta)) or not np.all(np.isfinite(g)) or not np.all(np.isfinite(h)):
        raise RuntimeError("Nonfinite frozen decoder fields")
    return dict(eta=eta,G=g,H=h,subject_ids=np.asarray(subjects),x_ids=np.asarray(trial_ids),
                x_raw=bank,
                max_hessian_asymmetry=float(np.max(np.abs(h-h.swapaxes(-1,-2)))),
                U_coordinate="TRAIN PCA whitened q", X_coordinate="CR4 checkpoint X standardized",
                response="choice logit1 - logit0")
