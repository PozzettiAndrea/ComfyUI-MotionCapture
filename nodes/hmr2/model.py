"""
Consolidated HMR2 model — all neural network classes for the HMR2 backbone
and SMPL head, migrated to ComfyUI-native format.

Absorbed from:
  - network/hmr2/vit.py
  - network/hmr2/smpl_head.py
  - network/hmr2/components/pose_transformer.py
  - network/hmr2/components/t_cond_mlp.py
  - network/hmr2/hmr2.py
  - network/hmr2/__init__.py
"""

import copy
import math
from functools import partial
from inspect import isfunction
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from einops.layers.torch import Rearrange
from yacs.config import CfgNode

import comfy.ops
from comfy.ldm.modules.attention import optimized_attention

ops = comfy.ops.manual_cast

# ---------------------------------------------------------------------------
# Inlined timm utilities
# ---------------------------------------------------------------------------

def _to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def _drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


# ---------------------------------------------------------------------------
# Geometry utilities (from network/hmr2/utils/geometry.py, used by smpl_head)
# ---------------------------------------------------------------------------

def aa_to_rotmat(theta: torch.Tensor):
    norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return _quat_to_rotmat(quat)


def _quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    B = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rotMat = torch.stack(
        [w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
         2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
         2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2],
        dim=1,
    ).view(B, 3, 3)
    return rotMat


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


# ---------------------------------------------------------------------------
# Small helpers (from pose_transformer.py)
# ---------------------------------------------------------------------------

def _exists(val):
    return val is not None


def _default(val, d):
    if _exists(val):
        return val
    return d() if isfunction(d) else d


# ============================================================================
# From t_cond_mlp.py — Adaptive normalization, MLP builders
# ============================================================================

class AdaptiveLayerNorm1D(nn.Module):
    def __init__(self, data_dim: int, norm_cond_dim: int,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        if data_dim <= 0:
            raise ValueError(f"data_dim must be positive, but got {data_dim}")
        if norm_cond_dim <= 0:
            raise ValueError(f"norm_cond_dim must be positive, but got {norm_cond_dim}")
        self.norm = operations.LayerNorm(data_dim, dtype=dtype, device=device)
        self.linear = operations.Linear(norm_cond_dim, 2 * data_dim, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        alpha, beta = self.linear(t).chunk(2, dim=-1)
        if x.dim() > 2:
            alpha = alpha.view(alpha.shape[0], *([1] * (x.dim() - 2)), alpha.shape[1])
            beta = beta.view(beta.shape[0], *([1] * (x.dim() - 2)), beta.shape[1])
        return x * (1 + alpha) + beta


class SequentialCond(nn.Sequential):
    def forward(self, input, *args, **kwargs):
        for module in self:
            if isinstance(module, (AdaptiveLayerNorm1D, SequentialCond, ResidualMLPBlock)):
                input = module(input, *args, **kwargs)
            else:
                input = module(input)
        return input


def normalization_layer(norm: Optional[str], dim: int, norm_cond_dim: int = -1,
                        dtype=None, device=None, operations=ops):
    if norm == "batch":
        return nn.BatchNorm1d(dim)
    elif norm == "layer":
        return operations.LayerNorm(dim, dtype=dtype, device=device)
    elif norm == "ada":
        assert norm_cond_dim > 0, f"norm_cond_dim must be positive, got {norm_cond_dim}"
        return AdaptiveLayerNorm1D(dim, norm_cond_dim,
                                   dtype=dtype, device=device, operations=operations)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm: {norm}")


def linear_norm_activ_dropout(
    input_dim: int,
    output_dim: int,
    activation: nn.Module = nn.ReLU(),
    bias: bool = True,
    norm: Optional[str] = "layer",
    dropout: float = 0.0,
    norm_cond_dim: int = -1,
    dtype=None, device=None, operations=ops,
) -> SequentialCond:
    layers = []
    layers.append(operations.Linear(input_dim, output_dim, bias=bias,
                                    dtype=dtype, device=device))
    if norm is not None:
        layers.append(normalization_layer(norm, output_dim, norm_cond_dim,
                                          dtype=dtype, device=device, operations=operations))
    layers.append(copy.deepcopy(activation))
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return SequentialCond(*layers)


def create_simple_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: nn.Module = nn.ReLU(),
    bias: bool = True,
    norm: Optional[str] = "layer",
    dropout: float = 0.0,
    norm_cond_dim: int = -1,
    dtype=None, device=None, operations=ops,
) -> SequentialCond:
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend(
            linear_norm_activ_dropout(
                prev_dim, hidden_dim, activation, bias, norm, dropout, norm_cond_dim,
                dtype=dtype, device=device, operations=operations,
            )
        )
        prev_dim = hidden_dim
    layers.append(operations.Linear(prev_dim, output_dim, bias=bias,
                                    dtype=dtype, device=device))
    return SequentialCond(*layers)


class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
        activation: nn.Module = nn.ReLU(),
        bias: bool = True,
        norm: Optional[str] = "layer",
        dropout: float = 0.0,
        norm_cond_dim: int = -1,
        dtype=None, device=None, operations=ops,
    ):
        super().__init__()
        if not (input_dim == output_dim == hidden_dim):
            raise NotImplementedError(
                f"input_dim {input_dim} != output_dim {output_dim} is not implemented"
            )
        layers = []
        prev_dim = input_dim
        for i in range(num_hidden_layers):
            layers.append(
                linear_norm_activ_dropout(
                    prev_dim, hidden_dim, activation, bias, norm, dropout, norm_cond_dim,
                    dtype=dtype, device=device, operations=operations,
                )
            )
            prev_dim = hidden_dim
        self.model = SequentialCond(*layers)
        self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + self.model(x, *args, **kwargs)


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
        activation: nn.Module = nn.ReLU(),
        bias: bool = True,
        norm: Optional[str] = "layer",
        dropout: float = 0.0,
        num_blocks: int = 1,
        norm_cond_dim: int = -1,
        dtype=None, device=None, operations=ops,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.model = SequentialCond(
            linear_norm_activ_dropout(
                input_dim, hidden_dim, activation, bias, norm, dropout, norm_cond_dim,
                dtype=dtype, device=device, operations=operations,
            ),
            *[
                ResidualMLPBlock(
                    hidden_dim, hidden_dim, num_hidden_layers, hidden_dim,
                    activation, bias, norm, dropout, norm_cond_dim,
                    dtype=dtype, device=device, operations=operations,
                )
                for _ in range(num_blocks)
            ],
            operations.Linear(hidden_dim, output_dim, bias=bias,
                              dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.model(x, *args, **kwargs)


class FrequencyEmbedder(nn.Module):
    def __init__(self, num_frequencies, max_freq_log2):
        super().__init__()
        frequencies = 2 ** torch.linspace(0, max_freq_log2, steps=num_frequencies)
        self.register_buffer("frequencies", frequencies)

    def forward(self, x):
        N = x.size(0)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x_unsqueezed = x.unsqueeze(-1)
        scaled = self.frequencies.view(1, 1, -1) * x_unsqueezed
        s = torch.sin(scaled)
        c = torch.cos(scaled)
        embedded = torch.cat([s, c, x_unsqueezed], dim=-1).view(N, -1)
        return embedded


# ============================================================================
# From pose_transformer.py — Attention, CrossAttention, Transformer blocks
# ============================================================================

class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: Callable, norm: str = "layer", norm_cond_dim: int = -1,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.norm = normalization_layer(norm, dim, norm_cond_dim,
                                        dtype=dtype, device=device, operations=operations)
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if isinstance(self.norm, AdaptiveLayerNorm1D):
            return self.fn(self.norm(x, *args), **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.net = nn.Sequential(
            operations.Linear(dim, hidden_dim, dtype=dtype, device=device),
            nn.GELU(),
            nn.Dropout(dropout),
            operations.Linear(hidden_dim, dim, dtype=dtype, device=device),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PoseAttention(nn.Module):
    """Self-attention for the pose transformer (from components/pose_transformer.py Attention)."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads

        self.to_qkv = operations.Linear(dim, inner_dim * 3, bias=False,
                                        dtype=dtype, device=device)
        self.to_out = (
            nn.Sequential(
                operations.Linear(inner_dim, dim, dtype=dtype, device=device),
                nn.Dropout(dropout),
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        out = optimized_attention(q, k, v, heads=self.heads,
                                  skip_reshape=True, skip_output_reshape=True)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads

        context_dim = _default(context_dim, dim)
        self.to_kv = operations.Linear(context_dim, inner_dim * 2, bias=False,
                                       dtype=dtype, device=device)
        self.to_q = operations.Linear(dim, inner_dim, bias=False,
                                      dtype=dtype, device=device)
        self.to_out = (
            nn.Sequential(
                operations.Linear(inner_dim, dim, dtype=dtype, device=device),
                nn.Dropout(dropout),
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x, context=None):
        context = _default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = self.to_q(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), [q, k, v])
        out = optimized_attention(q, k, v, heads=self.heads,
                                  skip_reshape=True, skip_output_reshape=True)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int,
                 dropout: float = 0.0, norm: str = "layer", norm_cond_dim: int = -1,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            sa = PoseAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                               dtype=dtype, device=device, operations=operations)
            ff = FeedForward(dim, mlp_dim, dropout=dropout,
                             dtype=dtype, device=device, operations=operations)
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim,
                            dtype=dtype, device=device, operations=operations),
                    PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim,
                            dtype=dtype, device=device, operations=operations),
                ])
            )

    def forward(self, x: torch.Tensor, *args):
        for attn, ff in self.layers:
            x = attn(x, *args) + x
            x = ff(x, *args) + x
        return x


class TransformerCrossAttn(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int,
                 dropout: float = 0.0, norm: str = "layer", norm_cond_dim: int = -1,
                 context_dim: Optional[int] = None,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            sa = PoseAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                               dtype=dtype, device=device, operations=operations)
            ca = CrossAttention(dim, context_dim=context_dim, heads=heads,
                                dim_head=dim_head, dropout=dropout,
                                dtype=dtype, device=device, operations=operations)
            ff = FeedForward(dim, mlp_dim, dropout=dropout,
                             dtype=dtype, device=device, operations=operations)
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim,
                            dtype=dtype, device=device, operations=operations),
                    PreNorm(dim, ca, norm=norm, norm_cond_dim=norm_cond_dim,
                            dtype=dtype, device=device, operations=operations),
                    PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim,
                            dtype=dtype, device=device, operations=operations),
                ])
            )

    def forward(self, x: torch.Tensor, *args, context=None, context_list=None):
        if context_list is None:
            context_list = [context] * len(self.layers)
        if len(context_list) != len(self.layers):
            raise ValueError(
                f"len(context_list) != len(self.layers) ({len(context_list)} != {len(self.layers)})"
            )
        for i, (self_attn, cross_attn, ff) in enumerate(self.layers):
            x = self_attn(x, *args) + x
            x = cross_attn(x, *args, context=context_list[i]) + x
            x = ff(x, *args) + x
        return x


class DropTokenDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p

    def forward(self, x: torch.Tensor):
        if self.training and self.p > 0:
            zero_mask = torch.full_like(x[0, :, 0], self.p).bernoulli().bool()
            if zero_mask.any():
                x = x[:, ~zero_mask, :]
        return x


class ZeroTokenDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p

    def forward(self, x: torch.Tensor):
        if self.training and self.p > 0:
            zero_mask = torch.full_like(x[:, :, 0], self.p).bernoulli().bool()
            x[zero_mask, :] = 0
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_tokens: int, token_dim: int, dim: int, depth: int,
                 heads: int, mlp_dim: int, dim_head: int = 64, dropout: float = 0.0,
                 emb_dropout: float = 0.0, emb_dropout_type: str = "drop",
                 emb_dropout_loc: str = "token", norm: str = "layer",
                 norm_cond_dim: int = -1, token_pe_numfreq: int = -1,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        if token_pe_numfreq > 0:
            token_dim_new = token_dim * (2 * token_pe_numfreq + 1)
            self.to_token_embedding = nn.Sequential(
                Rearrange("b n d -> (b n) d", n=num_tokens, d=token_dim),
                FrequencyEmbedder(token_pe_numfreq, token_pe_numfreq - 1),
                Rearrange("(b n) d -> b n d", n=num_tokens, d=token_dim_new),
                operations.Linear(token_dim_new, dim, dtype=dtype, device=device),
            )
        else:
            self.to_token_embedding = operations.Linear(token_dim, dim,
                                                        dtype=dtype, device=device)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        if emb_dropout_type == "drop":
            self.dropout = DropTokenDropout(emb_dropout)
        elif emb_dropout_type == "zero":
            self.dropout = ZeroTokenDropout(emb_dropout)
        else:
            raise ValueError(f"Unknown emb_dropout_type: {emb_dropout_type}")
        self.emb_dropout_loc = emb_dropout_loc

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, norm=norm,
            norm_cond_dim=norm_cond_dim,
            dtype=dtype, device=device, operations=operations,
        )

    def forward(self, inp: torch.Tensor, *args, **kwargs):
        x = inp
        if self.emb_dropout_loc == "input":
            x = self.dropout(x)
        x = self.to_token_embedding(x)
        if self.emb_dropout_loc == "token":
            x = self.dropout(x)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        if self.emb_dropout_loc == "token_afterpos":
            x = self.dropout(x)
        x = self.transformer(x, *args)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_tokens: int, token_dim: int, dim: int, depth: int,
                 heads: int, mlp_dim: int, dim_head: int = 64, dropout: float = 0.0,
                 emb_dropout: float = 0.0, emb_dropout_type: str = "drop",
                 norm: str = "layer", norm_cond_dim: int = -1,
                 context_dim: Optional[int] = None,
                 skip_token_embedding: bool = False,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        if not skip_token_embedding:
            self.to_token_embedding = operations.Linear(token_dim, dim,
                                                        dtype=dtype, device=device)
        else:
            self.to_token_embedding = nn.Identity()
            if token_dim != dim:
                raise ValueError(
                    f"token_dim ({token_dim}) != dim ({dim}) when skip_token_embedding is True"
                )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        if emb_dropout_type == "drop":
            self.dropout = DropTokenDropout(emb_dropout)
        elif emb_dropout_type == "zero":
            self.dropout = ZeroTokenDropout(emb_dropout)
        elif emb_dropout_type == "normal":
            self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerCrossAttn(
            dim, depth, heads, dim_head, mlp_dim, dropout,
            norm=norm, norm_cond_dim=norm_cond_dim, context_dim=context_dim,
            dtype=dtype, device=device, operations=operations,
        )

    def forward(self, inp: torch.Tensor, *args, context=None, context_list=None):
        x = self.to_token_embedding(inp)
        b, n, _ = x.shape
        x = self.dropout(x)
        x += self.pos_embedding[:, :n]
        x = self.transformer(x, *args, context=context, context_list=context_list)
        return x


# ============================================================================
# From vit.py — ViT backbone
# ============================================================================

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = operations.Linear(in_features, hidden_features,
                                     dtype=dtype, device=device)
        self.act = act_layer()
        self.fc2 = operations.Linear(hidden_features, out_features,
                                     dtype=dtype, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ViTAttention(nn.Module):
    """Multi-head attention for the ViT backbone."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., attn_head_dim=None,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.qkv = operations.Linear(dim, all_head_dim * 3, bias=qkv_bias,
                                     dtype=dtype, device=device)
        self.proj = operations.Linear(all_head_dim, dim,
                                      dtype=dtype, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B, H, N, D)

        x = optimized_attention(q, k, v, heads=self.num_heads,
                                skip_reshape=True, skip_output_reshape=True)
        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=None, attn_head_dim=None,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        if norm_layer is None:
            norm_layer = lambda d: operations.LayerNorm(d, eps=1e-6,
                                                        dtype=dtype, device=device)
        self.norm1 = norm_layer(dim)
        self.attn = ViTAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
            dtype=dtype, device=device, operations=operations,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop,
                       dtype=dtype, device=device, operations=operations)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio),
                            int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]),
                                   int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = operations.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size,
            stride=(patch_size[0] // ratio),
            padding=4 + 2 * (ratio // 2 - 1),
            dtype=dtype, device=device,
        )

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


def get_abs_pos(abs_pos, h, w, ori_h, ori_w, has_cls_token=True):
    cls_token = None
    B, L, C = abs_pos.shape
    if has_cls_token:
        cls_token = abs_pos[:, 0:1]
        abs_pos = abs_pos[:, 1:]
    if ori_h != h or ori_w != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
            size=(h, w), mode="bicubic", align_corners=False,
        ).permute(0, 2, 3, 1).reshape(B, -1, C)
    else:
        new_abs_pos = abs_pos
    if cls_token is not None:
        new_abs_pos = torch.cat([cls_token, new_abs_pos], dim=1)
    return new_abs_pos


class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
                 use_checkpoint=False, frozen_stages=-1, ratio=1, last_norm=True,
                 patch_padding='pad', freeze_attn=False, freeze_ffn=False,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        if norm_layer is None:
            norm_layer = lambda dim: operations.LayerNorm(dim, eps=1e-6,
                                                          dtype=dtype, device=device)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, ratio=ratio,
            dtype=dtype, device=device, operations=operations,
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i],
                dtype=dtype, device=device, operations=operations,
            )
            for i in range(depth)
        ])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None and self.pos_embed.device.type != "meta":
            nn.init.trunc_normal_(self.pos_embed, std=.02)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False
        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.last_norm(x)
        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
        return xp

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()


# ============================================================================
# From smpl_head.py — SMPL decoder head
# ============================================================================

class SMPLTransformerDecoderHead(nn.Module):
    def __init__(self, cfg, dtype=None, device=None, operations=ops):
        super().__init__()
        self.cfg = cfg
        self.joint_rep_type = cfg.MODEL.SMPL_HEAD.get("JOINT_REP", "6d")
        self.joint_rep_dim = {"6d": 6, "aa": 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * (cfg.SMPL.NUM_BODY_JOINTS + 1)
        self.npose = npose
        self.input_is_mean_shape = cfg.MODEL.SMPL_HEAD.get("TRANSFORMER_INPUT", "zero") == "mean_shape"
        transformer_args = dict(
            num_tokens=1,
            token_dim=(npose + 10 + 3) if self.input_is_mean_shape else 1,
            dim=1024,
        )
        transformer_args.update(**dict(cfg.MODEL.SMPL_HEAD.TRANSFORMER_DECODER))
        self.transformer = TransformerDecoder(
            **transformer_args,
            dtype=dtype, device=device, operations=operations,
        )
        dim = transformer_args["dim"]
        self.decpose = operations.Linear(dim, npose, dtype=dtype, device=device)
        self.decshape = operations.Linear(dim, 10, dtype=dtype, device=device)
        self.deccam = operations.Linear(dim, 3, dtype=dtype, device=device)

        mean_params = np.load(cfg.SMPL.MEAN_PARAMS)
        init_body_pose = torch.from_numpy(mean_params["pose"].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params["shape"].astype("float32")).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params["cam"].astype(np.float32)).unsqueeze(0)
        self.register_buffer("init_body_pose", init_body_pose)
        self.register_buffer("init_betas", init_betas)
        self.register_buffer("init_cam", init_cam)

    def forward(self, x, only_return_token_out=False):
        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = rearrange(x, "b c h w -> b (h w) c")

        init_body_pose = self.init_body_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        if self.joint_rep_type == "aa":
            raise NotImplementedError

        pred_body_pose = init_body_pose
        pred_betas = init_betas
        pred_cam = init_cam
        pred_body_pose_list = []
        pred_betas_list = []
        pred_cam_list = []
        for i in range(self.cfg.MODEL.SMPL_HEAD.get("IEF_ITERS", 1)):
            assert i == 0, "Only support 1 iteration for now"

            if self.input_is_mean_shape:
                token = torch.cat([pred_body_pose, pred_betas, pred_cam], dim=1)[:, None, :]
            else:
                token = torch.zeros(batch_size, 1, 1, device=x.device, dtype=x.dtype)

            token_out = self.transformer(token, context=x)
            token_out = token_out.squeeze(1)

            if only_return_token_out:
                return token_out
            else:
                pred_body_pose = self.decpose(token_out) + pred_body_pose
                pred_betas = self.decshape(token_out) + pred_betas
                pred_cam = self.deccam(token_out) + pred_cam
                pred_body_pose_list.append(pred_body_pose)
                pred_betas_list.append(pred_betas)
                pred_cam_list.append(pred_cam)

        joint_conversion_fn = {
            "6d": rot6d_to_rotmat,
            "aa": lambda x: aa_to_rotmat(x.view(-1, 3).contiguous()),
        }[self.joint_rep_type]

        pred_smpl_params_list = {}
        pred_smpl_params_list["body_pose"] = torch.cat(
            [joint_conversion_fn(pbp).view(batch_size, -1, 3, 3)[:, 1:, :, :]
             for pbp in pred_body_pose_list], dim=0
        )
        pred_smpl_params_list["betas"] = torch.cat(pred_betas_list, dim=0)
        pred_smpl_params_list["cam"] = torch.cat(pred_cam_list, dim=0)
        pred_body_pose = joint_conversion_fn(pred_body_pose).view(
            batch_size, self.cfg.SMPL.NUM_BODY_JOINTS + 1, 3, 3
        )

        pred_smpl_params = {
            "global_orient": pred_body_pose[:, [0]],
            "body_pose": pred_body_pose[:, 1:],
            "betas": pred_betas,
        }
        return pred_smpl_params, pred_cam, pred_smpl_params_list, token_out


# ============================================================================
# From hmr2.py — HMR2 model (converted from pl.LightningModule to nn.Module)
# ============================================================================

class HMR2(nn.Module):
    def __init__(self, cfg: CfgNode,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.cfg = cfg
        self.backbone = ViT(
            img_size=(256, 192),
            patch_size=16,
            embed_dim=1280,
            depth=32,
            num_heads=16,
            ratio=1,
            use_checkpoint=False,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.55,
            dtype=dtype, device=device, operations=operations,
        )
        self.smpl_head = SMPLTransformerDecoderHead(
            cfg, dtype=dtype, device=device, operations=operations,
        )

    def forward(self, batch, feat_mode=True):
        x = batch["img"][:, :, :, 32:-32]
        vit_feats = self.backbone(x)
        if feat_mode:
            token_out = self.smpl_head(vit_feats, only_return_token_out=True)
            return token_out
        from ..motion_utils.pytorch3d_shim import matrix_to_axis_angle
        from ..motion_utils.hmr_cam import compute_transl_full_cam
        pred_smpl_params, pred_cam, _, token_out = self.smpl_head(
            vit_feats, only_return_token_out=False
        )
        output = {}
        output["token_out"] = token_out
        output["smpl_params"] = {
            "body_pose": matrix_to_axis_angle(pred_smpl_params["body_pose"]).flatten(-2),
            "betas": pred_smpl_params["betas"],
            "global_orient": matrix_to_axis_angle(pred_smpl_params["global_orient"])[:, 0],
            "transl": compute_transl_full_cam(pred_cam, batch["bbx_xys"], batch["K_fullimg"]),
        }
        return output


# ============================================================================
# From network/hmr2/__init__.py — load_hmr2()
# ============================================================================

def load_hmr2(checkpoint_path=None):
    from .configs import get_config

    if checkpoint_path is None:
        import folder_paths
        checkpoint_path = Path(folder_paths.models_dir) / "motion_capture" / "hmr2.safetensors"

    model_cfg = str((Path(__file__).parent / "configs" / "model_config.yaml").resolve())
    model_cfg = get_config(model_cfg)

    if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and ("BBOX_SHAPE" not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, \
            f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()

    with torch.device("meta"):
        model = HMR2(model_cfg)

    import comfy.utils
    state_dict = comfy.utils.load_torch_file(str(checkpoint_path))
    keys = [k for k in state_dict.keys() if k.split(".")[0] in ["backbone", "smpl_head"]]
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    model.load_state_dict(state_dict, strict=False, assign=True)

    return model
