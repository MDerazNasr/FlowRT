from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import SinusoidalPositionalEmbedding, TimestepEmbedding, Timesteps
from torch import nn


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        return self.timestep_embedder(timesteps_proj)


class AdaLayerNorm(nn.Module):
    """
    LayerNorm + timestep-conditioned affine scale/shift.

    pattern: x = layernorm(x) * (1 + scale) + shift
    where scale, shift = linear(silu(temb)).chunk(2)

    This is the pattern fused_ln_linear_time.cu targets in FlowRT.
    On Intel iGPU, this maps to the AdaLayerNorm fusion contribution
    described in the OpenVINO GSoC proposal.
    """

    def __init__(self, embedding_dim, norm_eps=1e-5):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, norm_eps, elementwise_affine=False)

    def forward(self, x, temb):
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim,
                 dropout=0.0, cross_attention_dim=None, activation_fn="geglu",
                 attention_bias=False, norm_type="layer_norm", norm_eps=1e-5,
                 final_dropout=False, positional_embeddings=None,
                 num_positional_embeddings=None):
        super().__init__()
        self.norm_type = norm_type

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim, norm_eps)
        else:
            self.norm1 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=True)

        # diffusers Attention uses F.scaled_dot_product_attention internally.
        # No _sdpa_context() CUDA guard — unifolm-vla's DiT is clean.
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
        )
        self.norm3 = nn.LayerNorm(dim, norm_eps, elementwise_affine=True)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn,
                              final_dropout=final_dropout)
        self.final_dropout = nn.Dropout(dropout) if final_dropout else None

    def forward(self, hidden_states, attention_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                temb=None):
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class DiT(ModelMixin, ConfigMixin):
    """
    Cross-attention DiT transformer for flow matching.

    Matches the architecture in unifolm-vla cross_attention_dit.py.
    Export-clean: no CUDA guards, no device-specific ops in forward path.
    """

    @register_to_config
    def __init__(self, num_attention_heads=8, attention_head_dim=64, output_dim=26,
                 num_layers=12, dropout=0.1, attention_bias=True,
                 activation_fn="gelu-approximate", norm_type="ada_norm",
                 norm_elementwise_affine=False, norm_eps=1e-5,
                 max_num_positional_embeddings=512, final_dropout=True,
                 positional_embeddings="sinusoidal", cross_attention_dim=None, **kwargs):
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim
        self.timestep_encoder = TimestepEncoder(embedding_dim=self.inner_dim)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                self.inner_dim, num_attention_heads, attention_head_dim,
                dropout=dropout, activation_fn=activation_fn,
                attention_bias=attention_bias, norm_type=norm_type,
                norm_eps=norm_eps,
                positional_embeddings=positional_embeddings,
                num_positional_embeddings=max_num_positional_embeddings,
                final_dropout=final_dropout, cross_attention_dim=cross_attention_dim,
            )
            for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, output_dim)

    def forward(self, hidden_states, encoder_hidden_states, timestep):
        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )

        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden_states)
