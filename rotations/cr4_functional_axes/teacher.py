"""Frozen CR4 decoder; apply CR4's persisted decoder-X transform exactly."""
from __future__ import annotations
import numpy as np
import torch
from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig,DiscreteNeuralProcess
from modlearn_rot.training.continuous import load_checkpoint


def restore(source, device="auto"):
    ck = load_checkpoint(source["checkpoint"])
    extra = ck.get("extra_state", {})
    pre = extra.get("decoder_x_standardization")
    if not pre:
        raise RuntimeError("CR4 checkpoint missing decoder_x_standardization: retrain with CR4-adapted S3")
    from rotations.cr4_functional_axes.data import sha256
    if pre.get("bank_sha256") != sha256(source["root"]/"s0_design"/"CR4_3P_random500.npz"):
        raise RuntimeError("CR4 decoder-X scaler originated from a different stimulus bank")
    x_mu = np.asarray(pre["mean"], dtype=np.float64)
    x_sd = np.asarray(pre["sd"], dtype=np.float64)
    if x_mu.shape != (6,) or x_sd.shape != (6,) or np.any(x_sd <= 0):
        raise RuntimeError("Invalid CR4 decoder X scaler")
    cfg = DiscreteNPConfig(**ck["model_config"])
    if cfg.latent_dim != source["z"].shape[1] or cfg.x_dim != 6 or cfg.num_choices != 2:
        raise RuntimeError("Checkpoint decoder dimensions mismatch")
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                       "cpu" if device == "auto" else device)
    model = DiscreteNeuralProcess(cfg)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model = model.to(dev).eval()
    model.requires_grad_(False)
    return model, dev, x_mu, x_sd


def logodds(model, z, x):
    y = model.decode(x,z)
    return y[...,1] - y[...,0]
