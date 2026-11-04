"""Compatibility re-export of the shared NP Set Transformer."""

from .np.common.set_transformer import (
    PoolingByMultiheadAttention,
    SetAttentionBlock,
    SetTransformer,
)

__all__=["SetTransformer","SetAttentionBlock","PoolingByMultiheadAttention"]
