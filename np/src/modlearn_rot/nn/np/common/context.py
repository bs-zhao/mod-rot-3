"""Shared context-set utilities for Neural Processes.

The functions match the latent_choice_rt-v1 context semantics.  A caller may
choose the policy (for example per-epoch or per-presentation); the set encoder
itself always receives a packed context plus a padding mask.
"""

from __future__ import annotations

import numpy as np
import torch


def use_full_context(p1, p2):
    return float(p1) >= 1.0 and float(p2) >= 1.0


def context_count_bounds(n_trials, p1, p2):
    p1, p2 = float(p1), float(p2)
    if p2 < p1:
        p1, p2 = p2, p1
    p1 = min(max(p1, 0.0), 1.0)
    p2 = min(max(p2, 0.0), 1.0)
    low = max(1, int(np.floor(p1 * n_trials)))
    high_incl = max(low, int(np.round(p2 * n_trials)))
    return low, min(high_incl, n_trials)


def sample_context_keep_mask_np(valid_mask, p1, p2, rng):
    n_subj, n_trials = valid_mask.shape
    keep = np.zeros((n_subj, n_trials), dtype=bool)
    valid = valid_mask > 0.5
    for i in range(n_subj):
        valid_idx = np.flatnonzero(valid[i])
        n_valid = valid_idx.size
        if n_valid == 0:
            continue
        low, high_incl = context_count_bounds(n_valid, p1, p2)
        n_pick = int(rng.integers(low, high_incl + 1))
        chosen = rng.choice(valid_idx, size=n_pick, replace=False)
        keep[i, chosen] = True
    return keep


def sample_context_prefix_mask_np(valid_mask, p1, p2, rng):
    n_subj, n_trials = valid_mask.shape
    keep = np.zeros((n_subj, n_trials), dtype=bool)
    valid = valid_mask > 0.5
    for i in range(n_subj):
        valid_idx = np.flatnonzero(valid[i])
        n_valid = valid_idx.size
        if n_valid == 0:
            continue
        low, high_incl = context_count_bounds(n_valid, p1, p2)
        n_pick = int(rng.integers(low, high_incl + 1))
        keep[i, valid_idx[:n_pick]] = True
    return keep


def sample_context_keep_mask_torch(valid_mask, p1, p2, generator=None):
    B, T = valid_mask.shape
    keep = torch.zeros(B, T, dtype=torch.bool, device=valid_mask.device)
    for i in range(B):
        valid_idx = torch.nonzero(
            valid_mask[i] > 0.5, as_tuple=False
        ).squeeze(-1)
        n_valid = int(valid_idx.numel())
        if n_valid == 0:
            continue
        low, high_incl = context_count_bounds(n_valid, p1, p2)
        if generator is None:
            n_pick = int(
                torch.randint(
                    low, high_incl + 1, (1,), device=valid_mask.device
                ).item()
            )
            perm = torch.randperm(
                n_valid, device=valid_mask.device
            )[:n_pick]
        else:
            n_pick = int(
                torch.randint(
                    low, high_incl + 1, (1,),
                    device=valid_mask.device,
                    generator=generator,
                ).item()
            )
            perm = torch.randperm(
                n_valid, device=valid_mask.device, generator=generator
            )[:n_pick]
        keep[i, valid_idx[perm]] = True
    return keep


def pack_context(x_enc, keep_mask):
    B, T, D = x_enc.shape
    x_ctx = torch.zeros_like(x_enc)
    pad_mask = torch.ones(B, T, dtype=torch.bool, device=x_enc.device)
    for i in range(B):
        idx = torch.nonzero(keep_mask[i], as_tuple=False).squeeze(-1)
        n = idx.numel()
        if n == 0:
            pad_mask[i, 0] = False
            continue
        x_ctx[i, :n] = x_enc[i, idx]
        pad_mask[i, :n] = False
    return x_ctx, pad_mask
