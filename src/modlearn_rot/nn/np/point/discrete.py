"""Categorical point-response Neural Process.

The pre-z encoder is exactly the shared NP encoder used by continuous and
Choice-RT.  Response-family differences begin after z: this module decodes
categorical logits from [X,z].
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from ..common.encoder import (
    encode_gaussian_set_from_keep_mask,
    initialize_gaussian_set_encoder,
)


def _make_decoder(dimensions: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index,(input_dim,output_dim) in enumerate(
        zip(dimensions[:-1],dimensions[1:],strict=True)
    ):
        layers.append(nn.Linear(input_dim,output_dim))
        if index < len(dimensions)-2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


@dataclass(frozen=True, slots=True)
class DiscreteNPConfig:
    x_dim: int
    num_choices: int = 2
    latent_dim: int = 16
    summary_dim: int = 64
    decoder_hidden_dims: tuple[int,...] = (128,256,128)

    def __post_init__(self):
        if self.x_dim <= 0:
            raise ValueError("x_dim must be positive.")
        if self.num_choices < 2:
            raise ValueError("num_choices must be at least 2.")
        if self.latent_dim <= 0 or self.summary_dim <= 0:
            raise ValueError("latent_dim and summary_dim must be positive.")

    @property
    def observation_dim(self) -> int:
        # One scalar categorical response channel is used by the shared encoder.
        return int(self.x_dim + 1)

    def to_dict(self) -> dict:
        return asdict(self)


class DiscreteNeuralProcess(nn.Module):
    """Subject-level NP for categorical choice."""

    def __init__(self,config:DiscreteNPConfig):
        super().__init__()
        self.config=config
        initialize_gaussian_set_encoder(
            self,
            observation_dim=config.observation_dim,
            latent_dim=config.latent_dim,
            summary_dim=config.summary_dim,
            num_heads=4,
            embed_hidden=32,
        )
        self.decoder=_make_decoder(
            [
                config.x_dim+config.latent_dim,
                *config.decoder_hidden_dims,
                config.num_choices,
            ]
        )

    @staticmethod
    def reparameterize(mean:Tensor,log_variance:Tensor)->Tensor:
        sd=torch.exp(0.5*log_variance)
        return mean+torch.randn_like(sd)*sd

    def encode_context(
        self,
        x_enc:Tensor,
        choice_enc:Tensor,
        mask:Tensor,
    )->tuple[Tensor,Tensor]:
        if x_enc.ndim!=3 or choice_enc.ndim!=3:
            raise ValueError("x_enc and choice_enc must have shapes (B,T,D).")
        if choice_enc.shape[-1]!=1:
            raise ValueError("choice_enc must have one scalar channel.")
        if x_enc.shape[:2]!=choice_enc.shape[:2] or mask.shape!=x_enc.shape[:2]:
            raise ValueError("x_enc, choice_enc, and mask must agree on B and T.")
        if x_enc.shape[-1]!=self.config.x_dim:
            raise ValueError("Unexpected X dimension.")
        observations=torch.cat([x_enc,choice_enc],dim=-1)
        return encode_gaussian_set_from_keep_mask(
            self,observations,mask.bool()
        )

    def decode(
        self,
        target_x:Tensor,
        latent:Tensor,
        target_mask:Tensor|None=None,
    )->Tensor:
        if target_x.ndim!=3 or target_x.shape[-1]!=self.config.x_dim:
            raise ValueError("target_x must have shape (B,T,x_dim).")
        if latent.shape!=(target_x.shape[0],self.config.latent_dim):
            raise ValueError("latent must have shape (B,latent_dim).")
        repeated=latent.unsqueeze(1).expand(-1,target_x.shape[1],-1)
        safe_x=target_x
        if target_mask is not None:
            safe_x=target_x.masked_fill(
                ~target_mask.bool().unsqueeze(-1),0.0
            )
        logits=self.decoder(torch.cat([safe_x,repeated],dim=-1))
        if target_mask is not None:
            logits=logits.masked_fill(
                ~target_mask.bool().unsqueeze(-1),0.0
            )
        return logits

    def forward(
        self,
        context_x_enc:Tensor,
        context_choice_enc:Tensor,
        context_mask:Tensor,
        target_x:Tensor,
        target_mask:Tensor|None=None,
        *,
        sample_latent:bool=True,
    )->dict[str,Tensor]:
        mean,log_variance=self.encode_context(
            context_x_enc,context_choice_enc,context_mask
        )
        latent=(
            self.reparameterize(mean,log_variance)
            if sample_latent else mean
        )
        logits=self.decode(target_x,latent,target_mask)
        return {
            "logits":logits,
            "mean":mean,
            "log_variance":log_variance,
            "latent":latent,
        }


__all__=["DiscreteNPConfig","DiscreteNeuralProcess"]
