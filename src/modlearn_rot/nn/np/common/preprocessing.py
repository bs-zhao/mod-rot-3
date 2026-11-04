"""Train-only channel standardization shared by NP encoders."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class ChannelStandardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform_numpy(self, x):
        return (np.asarray(x, dtype=np.float32) - self.mean) / self.std

    def transform_torch(self, x: torch.Tensor):
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return (x - mean) / std


def masked_mean_std(x, mask, eps=1e-6):
    """Reference-compatible train-only mean/std over valid trials."""
    x = np.asarray(x)
    mask = np.asarray(mask)
    w = (mask > 0.5).astype(np.float64)[..., None]
    denom = np.maximum(w.sum(axis=(0, 1), keepdims=True), 1.0)
    x64 = x.astype(np.float64)
    mean = (x64 * w).sum(axis=(0, 1), keepdims=True) / denom
    var = ((x64 - mean) ** 2 * w).sum(axis=(0, 1), keepdims=True) / denom
    return mean.astype(np.float32), (np.sqrt(var) + eps).astype(np.float32)


def fit_channel_standardizer(x, mask, eps=1e-6):
    mean, std = masked_mean_std(x, mask, eps=eps)
    return ChannelStandardizer(mean=mean, std=std)


def standardizer_from_checkpoint(preprocessing, *, mean_key="encoder_mean", std_key="encoder_std"):
    if preprocessing is None:
        raise KeyError("Checkpoint is missing encoder preprocessing statistics.")
    mean = preprocessing[mean_key]
    std = preprocessing[std_key]
    if hasattr(mean, "detach"):
        mean = mean.detach().cpu().numpy()
    if hasattr(std, "detach"):
        std = std.detach().cpu().numpy()
    return ChannelStandardizer(
        mean=np.asarray(mean, dtype=np.float32),
        std=np.asarray(std, dtype=np.float32),
    )
