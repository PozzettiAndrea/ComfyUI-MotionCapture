"""
Shared ViT building blocks used by both ViTPose and HMR2 models.

Contains: Mlp, Attention, Block, PatchEmbed, ViT backbone, and small utilities.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

import comfy.ops
from comfy.ldm.modules.attention import optimized_attention

ops = comfy.ops.manual_cast


# ---------------------------------------------------------------------------
# Small utilities (inlined from timm)
# ---------------------------------------------------------------------------

def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Truncated normal initialization (inlined from timm)."""
    with torch.no_grad():
        l_ = (1.0 + math.erf((a - mean) / (std * math.sqrt(2.0)))) / 2.0
        u_ = (1.0 + math.erf((b - mean) / (std * math.sqrt(2.0)))) / 2.0
        tensor.uniform_(2 * l_ - 1, 2 * u_ - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


# ---------------------------------------------------------------------------
# ViT building blocks
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., dtype=None, device=None,
                 operations=ops):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = operations.Linear(in_features, hidden_features, dtype=dtype, device=device)
        self.act = act_layer()
        self.fc2 = operations.Linear(hidden_features, out_features, dtype=dtype, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0., attn_head_dim=None,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.qkv = operations.Linear(dim, all_head_dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.proj = operations.Linear(all_head_dim, dim, dtype=dtype, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        out = optimized_attention(q, k, v, heads=self.num_heads,
                                  skip_reshape=True, skip_output_reshape=True)
        x = out.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=None, attn_head_dim=None,
                 dtype=None, device=None, operations=ops):
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(operations.LayerNorm, dtype=dtype, device=device)

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
            dtype=dtype, device=device, operations=operations,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop,
            dtype=dtype, device=device, operations=operations,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 ratio=1, dtype=None, device=None, operations=ops):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        self.patch_shape = (
            int(img_size[0] // patch_size[0] * ratio),
            int(img_size[1] // patch_size[1] * ratio),
        )
        self.origin_patch_shape = (
            int(img_size[0] // patch_size[0]),
            int(img_size[1] // patch_size[1]),
        )
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


class ViT(nn.Module):
    """Vision Transformer backbone (shared by ViTPose and HMR2)."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, use_checkpoint=False,
                 ratio=1, last_norm=True,
                 dtype=None, device=None, operations=ops):
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(operations.LayerNorm, eps=1e-6, dtype=dtype, device=device)

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        self.depth = depth

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, ratio=ratio,
            dtype=dtype, device=device, operations=operations,
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim, dtype=dtype, device=device)
        )

        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i],
                dtype=dtype, device=device, operations=operations,
            )
            for i in range(depth)
        ])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None and self.pos_embed.device.type != "meta":
            trunc_normal_(self.pos_embed, std=.02)

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
        return self.forward_features(x)
