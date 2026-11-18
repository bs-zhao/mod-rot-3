"""Frozen CR4/CR5 decoders; apply their saved decoder-X scaler."""
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
        raise RuntimeError("CR4/CR5 checkpoint missing decoder_x_standardization: retrain with S3")
    from rotations.cr4_functional_axes.data import sha256
    bank_name = ("CR4_3P_random500.npz" if source["model"]=="CR4"
                 else "CR5_3P_cr3bank500.npz")
    if pre.get("bank_sha256") != sha256(source["root"]/"s0_design"/bank_name):
        raise RuntimeError(f'{source["model"]} decoder-X scaler originated from a different bank')
    x_mu = np.asarray(pre["mean"], dtype=np.float64)
    x_sd = np.asarray(pre["sd"], dtype=np.float64)
    if x_mu.shape != (6,) or x_sd.shape != (6,) or np.any(x_sd <= 0):
        raise RuntimeError("Invalid 3P decoder X scaler")
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
