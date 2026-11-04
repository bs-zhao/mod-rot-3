"""Shared Set Transformer for all Neural-Process subject encoders.

This is numerically aligned with latent_choice_rt-v1/src/models/set_transformer.py.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


class SetAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(x + attn_out)
        return self.ln2(x + self.fc(x))


class PoolingByMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_seeds=1):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, num_seeds, embed_dim))
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask=None):
        bsz = x.shape[0]
        seeds = self.seed.repeat(bsz, 1, 1)
        attn_out, _ = self.mha(seeds, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(seeds + attn_out)
        return self.ln2(x + self.fc(x)).squeeze(1)


class SetTransformer(nn.Module):
    def __init__(self, input_dim=2, summary_dim=64, num_heads=4, embed_hidden=32):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_hidden),
            nn.ReLU(),
            nn.Linear(embed_hidden, embed_hidden),
        )
        self.sab1 = SetAttentionBlock(embed_hidden, num_heads)
        self.sab2 = SetAttentionBlock(embed_hidden, num_heads)
        self.pma = PoolingByMultiheadAttention(
            embed_hidden, num_heads, num_seeds=1
        )
        self.out_proj = nn.Linear(embed_hidden, summary_dim)

    def forward(self, x, key_padding_mask: Optional[torch.Tensor] = None):
        out = self.embed(x)
        out = self.sab1(out, key_padding_mask=key_padding_mask)
        out = self.sab2(out, key_padding_mask=key_padding_mask)
        out = self.pma(out, key_padding_mask=key_padding_mask)
        return self.out_proj(out)
