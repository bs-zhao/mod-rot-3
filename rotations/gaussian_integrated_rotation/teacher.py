"""Frozen Decoder; preserve physical-X to decoder-X preprocessing exactly."""
from __future__ import annotations
import numpy as np
import torch
from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig,DiscreteNeuralProcess
from modlearn_rot.training.continuous import load_checkpoint


def restore(source,device="auto"):
    ckpt=load_checkpoint(source["checkpoint"])
    meta=ckpt.get("extra_state",{}).get("decoder_x_standardization")
    if meta is None:
        raise RuntimeError("Decoder lacks persisted X z-score metadata; "
                           "CR3 requires new _xstd S3, not legacy raw-X checkpoint")
    if meta.get("bank_sha256")!=source["hashes"]["bank"]:
        raise RuntimeError("S3 decoder X scaler bank SHA differs from S0 stimulus bank")
    mu=np.asarray(meta.get("mean"),dtype=np.float64)
    sd=np.asarray(meta.get("sd"),dtype=np.float64)
    if mu.shape!=(6,) or sd.shape!=(6,) or not np.isfinite(mu).all() or not np.isfinite(sd).all() or np.any(sd<=0):
        raise RuntimeError("Bad decoder X mean/SD")
    cfg=DiscreteNPConfig(**ckpt["model_config"])
    if (cfg.latent_dim != source["B"].shape[0] or cfg.x_dim!=6
        or cfg.num_choices!=2):
        raise RuntimeError("Gaussian fit dimension mismatches frozen checkpoint")
    if device=="auto":
        device="cuda" if torch.cuda.is_available() else "cpu"
    dev=torch.device(device)
    model=DiscreteNeuralProcess(cfg).to(dev)
    model.load_state_dict(ckpt["model_state_dict"],strict=True)
    model.eval().requires_grad_(False)
    return model,dev,mu,sd


def logit(model,z,x):
    out=model.decode(x,z)
    if out.ndim!=3 or out.shape[-1]!=2:
        raise RuntimeError("Expected binary 2-logit decoder")
    return out[...,1]-out[...,0]
