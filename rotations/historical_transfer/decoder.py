# Frozen decoder read utilities copied from mlp_teacher_rotation/teacher.py at 2d89e4e for suite independence.
"""Frozen NP decoder as teacher: soft targets and exact Z-derivative fingerprints.

This suite does not train, fine-tune or update the NP. True simulator theta is
not read here. Decoder X is RAW, as in DiscreteNPTrainer; only the encoder
receives normalized observations.
"""
from __future__ import annotations
import numpy as np
import torch
from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig, DiscreteNeuralProcess
from modlearn_rot.training.continuous import load_checkpoint


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _restore(source, config):
    ck = load_checkpoint(source["checkpoint_path"])
    model_config = DiscreteNPConfig(**ck["model_config"])
    if model_config.latent_dim != source["z"].shape[1]:
        raise RuntimeError("Decoder latent dimension differs from S5 Z")
    if model_config.x_dim != source["x_fit"].shape[-1] or model_config.num_choices != 2:
        raise RuntimeError("Decoder X dimension or binary response family mismatch")
    device = _device(str(config["teacher_device"]))
    model = DiscreteNeuralProcess(model_config)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, device


