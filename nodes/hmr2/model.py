"""
Consolidated HMR2 model — ViT backbone (shared) + SMPL transformer decoder head,
migrated to ComfyUI-native format. Inference only.

Absorbed from:
  - network/hmr2/vit.py
  - network/hmr2/smpl_head.py
  - network/hmr2/components/pose_transformer.py
  - network/hmr2/components/t_cond_mlp.py
  - network/hmr2/hmr2.py
  - network/hmr2/__init__.py
"""

from functools import partial
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import comfy.ops
from comfy.ldm.modules.attention import optimized_attention

from ..shared_vit import ViT, ops


# ---------------------------------------------------------------------------
# Geometry utilities (from network/hmr2/utils/geometry.py)
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


# ============================================================================
# Adaptive normalization & MLP builders (from t_cond_mlp.py)
# ============================================================================

class AdaptiveLayerNorm1D(nn.Module):
    def __init__(self, data_dim: int, norm_cond_dim: int,
                 dtype=None, device=None, operations=ops):
        super().__init__()
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
    import copy
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


class ResidualMLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int,
                 output_dim: int, activation: nn.Module = nn.ReLU(),
                 bias: bool = True, norm: Optional[str] = "layer",
                 dropout: float = 0.0, norm_cond_dim: int = -1,
                 dtype=None, device=None, operations=ops):
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

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + self.model(x, *args, **kwargs)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int,
                 output_dim: int, activation: nn.Module = nn.ReLU(),
                 bias: bool = True, norm: Optional[str] = "layer",
                 dropout: float = 0.0, num_blocks: int = 1,
                 norm_cond_dim: int = -1,
                 dtype=None, device=None, operations=ops):
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


# ============================================================================
# Pose transformer blocks (from pose_transformer.py)
# ============================================================================

class PreNorm(nn.Module):
    def __init__(self, dim: int, fn, norm: str = "layer", norm_cond_dim: int = -1,
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
    """Self-attention for the pose transformer."""
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

        context_dim = context_dim if context_dim is not None else dim
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
        context = context if context is not None else x
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
        for i, (self_attn, cross_attn, ff) in enumerate(self.layers):
            x = self_attn(x, *args) + x
            x = cross_attn(x, *args, context=context_list[i]) + x
            x = ff(x, *args) + x
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

        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim,
                                                      dtype=dtype, device=device))
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
# SMPL decoder head
# ============================================================================

_SMPL_MEAN_PARAMS_PATH = Path(__file__).parent / "smpl_mean_params.npz"
_NUM_BODY_JOINTS = 23


class SMPLTransformerDecoderHead(nn.Module):
    def __init__(self, dtype=None, device=None, operations=ops):
        super().__init__()
        self.joint_rep_dim = 6  # 6d rotation representation
        npose = self.joint_rep_dim * (_NUM_BODY_JOINTS + 1)
        self.npose = npose
        self.transformer = TransformerDecoder(
            num_tokens=1,
            token_dim=1,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=1024,
            dim_head=64,
            dropout=0.0,
            emb_dropout=0.0,
            norm="layer",
            context_dim=1280,
            dtype=dtype, device=device, operations=operations,
        )
        self.decpose = operations.Linear(1024, npose, dtype=dtype, device=device)
        self.decshape = operations.Linear(1024, 10, dtype=dtype, device=device)
        self.deccam = operations.Linear(1024, 3, dtype=dtype, device=device)

        mean_params = np.load(_SMPL_MEAN_PARAMS_PATH)
        init_body_pose = torch.from_numpy(mean_params["pose"].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params["shape"].astype(np.float32)).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params["cam"].astype(np.float32)).unsqueeze(0)
        self.register_buffer("init_body_pose", init_body_pose)
        self.register_buffer("init_betas", init_betas)
        self.register_buffer("init_cam", init_cam)

    def forward(self, x, only_return_token_out=False):
        batch_size = x.shape[0]
        x = rearrange(x, "b c h w -> b (h w) c")

        init_body_pose = self.init_body_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        token = torch.zeros(batch_size, 1, 1, device=x.device, dtype=x.dtype)

        token_out = self.transformer(token, context=x)
        token_out = token_out.squeeze(1)

        if only_return_token_out:
            return token_out

        pred_body_pose = self.decpose(token_out) + init_body_pose
        pred_betas = self.decshape(token_out) + init_betas
        pred_cam = self.deccam(token_out) + init_cam

        joint_conversion_fn = rot6d_to_rotmat
        pred_smpl_params_list = {
            "body_pose": joint_conversion_fn(pred_body_pose).view(batch_size, -1, 3, 3)[:, 1:, :, :],
            "betas": pred_betas,
            "cam": pred_cam,
        }

        pred_body_pose_mat = joint_conversion_fn(pred_body_pose).view(
            batch_size, _NUM_BODY_JOINTS + 1, 3, 3
        )
        pred_smpl_params = {
            "global_orient": pred_body_pose_mat[:, [0]],
            "body_pose": pred_body_pose_mat[:, 1:],
            "betas": pred_betas,
        }
        return pred_smpl_params, pred_cam, pred_smpl_params_list, token_out


# ============================================================================
# HMR2 model
# ============================================================================

class HMR2(nn.Module):
    def __init__(self, dtype=None, device=None, operations=ops):
        super().__init__()
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
            dtype=dtype, device=device, operations=operations,
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
# load_hmr2()
# ============================================================================

def load_hmr2(checkpoint_path=None):
    if checkpoint_path is None:
        import folder_paths
        checkpoint_path = Path(folder_paths.models_dir) / "motion_capture" / "hmr2.safetensors"

    with torch.device("meta"):
        model = HMR2()

    import comfy.utils
    state_dict = comfy.utils.load_torch_file(str(checkpoint_path))
    keys = [k for k in state_dict.keys() if k.split(".")[0] in ["backbone", "smpl_head"]]
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    model.load_state_dict(state_dict, strict=False, assign=True)

    return model
