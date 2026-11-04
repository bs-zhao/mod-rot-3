"""Shared Gaussian Set-Transformer encoder construction for NP families.

The module assigns the reference-compatible names encoder_backbone, fc_mu, and
fc_logvar directly onto the owning NP model.  This keeps Choice-RT state-dict
keys compatible with latent_choice_rt-v1 while letting continuous/choice use
the exact same pre-z architecture.
"""

from __future__ import annotations

import torch.nn as nn

from .context import pack_context
from .set_transformer import SetTransformer


def initialize_gaussian_set_encoder(
    owner: nn.Module,
    *,
    observation_dim: int,
    latent_dim: int,
    summary_dim: int = 64,
    num_heads: int = 4,
    embed_hidden: int = 32,
):
    owner.encoder_backbone = SetTransformer(
        input_dim=int(observation_dim),
        summary_dim=int(summary_dim),
        num_heads=int(num_heads),
        embed_hidden=int(embed_hidden),
    )
    owner.fc_mu = nn.Linear(int(summary_dim), int(latent_dim))
    owner.fc_logvar = nn.Linear(int(summary_dim), int(latent_dim))


def encode_gaussian_set(owner: nn.Module, observations, key_padding_mask=None):
    summary = owner.encoder_backbone(
        observations, key_padding_mask=key_padding_mask
    )
    return owner.fc_mu(summary), owner.fc_logvar(summary)


def encode_gaussian_set_from_keep_mask(owner: nn.Module, observations, keep_mask):
    packed, pad_mask = pack_context(observations, keep_mask)
    return encode_gaussian_set(owner, packed, key_padding_mask=pad_mask)
