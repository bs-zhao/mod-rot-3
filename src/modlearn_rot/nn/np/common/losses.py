"""Latent regularizers shared across Neural-Process response families."""

from __future__ import annotations

import torch


def gaussian_kl_loss(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    """KL to N(0,I): sum latent dimensions, then mean subjects."""
    per_subject = -0.5 * torch.sum(
        1.0 + log_variance - mean.pow(2) - log_variance.exp(),
        dim=1,
    )
    return per_subject.mean()


def latent_l1_loss(mean: torch.Tensor) -> torch.Tensor:
    """L1 on posterior means: sum latent dimensions, then mean subjects."""
    return torch.abs(mean).sum(dim=1).mean()
