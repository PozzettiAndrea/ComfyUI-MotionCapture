"""
Consolidated GVHMR model -- all neural network classes for the GVHMR
temporal transformer, pipeline, and demo wrapper, migrated to ComfyUI-native format.

Absorbed from:
  - network/base_arch/embeddings/rotary_embedding.py
  - network/base_arch/transformer/encoder_rope.py
  - network/base_arch/transformer/layer.py
  - network/gvhmr/relative_transformer.py
  - model/gvhmr/pipeline/gvhmr_pipeline.py
  - model/gvhmr/gvhmr_pl_demo.py
"""

import logging
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.amp import autocast

import comfy.ops
from comfy.ldm.modules.attention import optimized_attention

from ..motion_utils.net_utils import length_to_mask, gaussian_smooth
from ..motion_utils.pylogger import Log
from ..motion_utils.hmr_cam import (
    compute_bbox_info_bedlam,
    compute_transl_full_cam,
    normalize_kp2d,
)
from ..motion_utils.hmr_global import (
    rollout_local_transl_vel,
    get_tgtcoord_rootparam,
)
from ..motion_utils.pytorch3d_shim import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)
from .postprocess import (
    pp_static_joint,
    process_ik,
    pp_static_joint_cam,
)
from . import stats as stats_compose

ops = comfy.ops.manual_cast
log = logging.getLogger("motioncapture")

# ============================================================================
# Rotary Embedding (from network/base_arch/embeddings/rotary_embedding.py)
# ============================================================================

def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@autocast("cuda", enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:].to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], \
        f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim=-1)


def get_encoding(d_model, max_seq_len=4096):
    """Return: (L, D)"""
    t = torch.arange(max_seq_len).float()
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    freqs = torch.einsum("i, j -> i j", t, freqs)
    freqs = repeat(freqs, "i j -> i (j r)", r=2)
    return freqs


class ROPE(nn.Module):
    """Minimal rotary positional encoding."""
    def __init__(self, d_model, max_seq_len=4096):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        with torch.device("cpu"):
            encoding = get_encoding(d_model, max_seq_len)
            self.register_buffer("encoding", encoding, False)

    def rotate_queries_or_keys(self, x):
        """x: (B, H, L, D) -> (B, H, L, D)"""
        seq_len, d_model = x.shape[-2:]
        assert d_model == self.d_model
        if seq_len > self.max_seq_len:
            encoding = get_encoding(d_model, seq_len).to(x)
        else:
            encoding = self.encoding[:seq_len]
        return apply_rotary_emb(encoding, x, seq_dim=-2)


# ============================================================================
# Utility (from network/base_arch/transformer/layer.py)
# ============================================================================

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


# ============================================================================
# Local Mlp (replaces timm.models.vision_transformer.Mlp)
# ============================================================================

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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============================================================================
# RoPE Attention & Encoder Block (from encoder_rope.py)
# ============================================================================

class RoPEAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.rope = ROPE(self.head_dim, max_seq_len=4096)

        self.query = operations.Linear(embed_dim, embed_dim,
                                       dtype=dtype, device=device)
        self.key = operations.Linear(embed_dim, embed_dim,
                                     dtype=dtype, device=device)
        self.value = operations.Linear(embed_dim, embed_dim,
                                       dtype=dtype, device=device)
        self.proj = operations.Linear(embed_dim, embed_dim,
                                      dtype=dtype, device=device)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x: (B, L, C)
        # attn_mask: (L, L) boolean -- True = masked
        # key_padding_mask: (B, L) boolean -- True = padded
        B, L, _ = x.shape
        xq, xk, xv = self.query(x), self.key(x), self.value(x)

        xq = xq.reshape(B, L, self.num_heads, -1).transpose(1, 2)  # (B, H, L, D)
        xk = xk.reshape(B, L, self.num_heads, -1).transpose(1, 2)
        xv = xv.reshape(B, L, self.num_heads, -1).transpose(1, 2)

        xq = self.rope.rotate_queries_or_keys(xq)
        xk = self.rope.rotate_queries_or_keys(xk)

        # Build combined float additive mask for SDPA (0 = attend, -inf = mask)
        combined_mask = None
        if attn_mask is not None or key_padding_mask is not None:
            combined_mask = torch.zeros(B, 1, L, L, device=x.device, dtype=xq.dtype)
            if attn_mask is not None:
                combined_mask = combined_mask.masked_fill(
                    attn_mask.reshape(1, 1, L, L), float("-inf")
                )
            if key_padding_mask is not None:
                combined_mask = combined_mask.masked_fill(
                    key_padding_mask.reshape(B, 1, 1, L), float("-inf")
                )

        output = optimized_attention(
            xq, xk, xv, heads=self.num_heads,
            mask=combined_mask, skip_reshape=True, skip_output_reshape=True,
        )
        output = output.transpose(1, 2).reshape(B, L, -1)
        output = self.proj(output)
        return output


class EncoderRoPEBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1,
                 dtype=None, device=None, operations=ops, **block_kwargs):
        super().__init__()
        self.norm1 = operations.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6,
                                          dtype=dtype, device=device)
        self.attn = RoPEAttention(hidden_size, num_heads, dropout,
                                  dtype=dtype, device=device, operations=operations)
        self.norm2 = operations.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6,
                                          dtype=dtype, device=device)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,
                       act_layer=approx_gelu, drop=dropout,
                       dtype=dtype, device=device, operations=operations)

        self.gate_msa = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.gate_mlp = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x, attn_mask=None, tgt_key_padding_mask=None):
        x = x + self.gate_msa * self._sa_block(
            self.norm1(x), attn_mask=attn_mask, key_padding_mask=tgt_key_padding_mask
        )
        x = x + self.gate_mlp * self.mlp(self.norm2(x))
        return x

    def _sa_block(self, x, attn_mask=None, key_padding_mask=None):
        return self.attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)


# ============================================================================
# NetworkEncoderRoPE (from network/gvhmr/relative_transformer.py)
# ============================================================================

class NetworkEncoderRoPE(nn.Module):
    def __init__(
        self,
        output_dim=151,
        max_len=120,
        cliffcam_dim=3,
        cam_angvel_dim=6,
        imgseq_dim=1024,
        latent_dim=512,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4.0,
        pred_cam_dim=3,
        static_conf_dim=6,
        dropout=0.1,
        avgbeta=True,
        dtype=None, device=None, operations=ops,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.max_len = max_len
        self.cliffcam_dim = cliffcam_dim
        self.cam_angvel_dim = cam_angvel_dim
        self.imgseq_dim = imgseq_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # Input (Kp2d)
        self.learned_pos_linear = operations.Linear(2, 32, dtype=dtype, device=device)
        self.learned_pos_params = nn.Parameter(torch.randn(17, 32), requires_grad=True)
        self.embed_noisyobs = Mlp(
            17 * 32, hidden_features=self.latent_dim * 2,
            out_features=self.latent_dim, drop=dropout,
            dtype=dtype, device=device, operations=operations,
        )

        # Condition embedders
        self.cliffcam_embedder = nn.Sequential(
            operations.Linear(self.cliffcam_dim, latent_dim, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(operations.Linear(latent_dim, latent_dim, dtype=dtype, device=device)),
        )
        if self.cam_angvel_dim > 0:
            self.cam_angvel_embedder = nn.Sequential(
                operations.Linear(self.cam_angvel_dim, latent_dim, dtype=dtype, device=device),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(operations.Linear(latent_dim, latent_dim, dtype=dtype, device=device)),
            )
        if self.imgseq_dim > 0:
            self.imgseq_embedder = nn.Sequential(
                operations.LayerNorm(self.imgseq_dim, dtype=dtype, device=device),
                zero_module(operations.Linear(self.imgseq_dim, latent_dim, dtype=dtype, device=device)),
            )

        # Transformer
        self.blocks = nn.ModuleList([
            EncoderRoPEBlock(self.latent_dim, self.num_heads, mlp_ratio=mlp_ratio,
                             dropout=dropout,
                             dtype=dtype, device=device, operations=operations)
            for _ in range(self.num_layers)
        ])

        # Output heads
        self.final_layer = Mlp(self.latent_dim, out_features=self.output_dim,
                               dtype=dtype, device=device, operations=operations)
        self.pred_cam_head = pred_cam_dim > 0
        if self.pred_cam_head:
            self.pred_cam_head = Mlp(self.latent_dim, out_features=pred_cam_dim,
                                    dtype=dtype, device=device, operations=operations)
            with torch.device("cpu"):
                self.register_buffer("pred_cam_mean", torch.tensor([1.0606, -0.0027, 0.2702]), False)
                self.register_buffer("pred_cam_std", torch.tensor([0.1784, 0.0956, 0.0764]), False)

        self.static_conf_head = static_conf_dim > 0
        if self.static_conf_head:
            self.static_conf_head = Mlp(self.latent_dim, out_features=static_conf_dim,
                                        dtype=dtype, device=device, operations=operations)

        self.avgbeta = avgbeta

    def forward(self, length, obs=None, f_cliffcam=None, f_cam_angvel=None, f_imgseq=None):
        B, L, J, C = obs.shape
        assert J == 17 and C == 3

        # Main token from observation (2D pose)
        obs = obs.clone()
        visible_mask = obs[..., [2]] > 0.5
        obs[~visible_mask[..., 0]] = 0
        f_obs = self.learned_pos_linear(obs[..., :2])
        f_obs = f_obs * visible_mask + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask
        x = self.embed_noisyobs(f_obs.view(B, L, -1))

        # Condition
        f_to_add = []
        f_to_add.append(self.cliffcam_embedder(f_cliffcam))
        if hasattr(self, "cam_angvel_embedder"):
            f_to_add.append(self.cam_angvel_embedder(f_cam_angvel))
        if f_imgseq is not None and hasattr(self, "imgseq_embedder"):
            f_to_add.append(self.imgseq_embedder(f_imgseq))

        for f_delta in f_to_add:
            x = x + f_delta

        # Setup length and make padding mask
        assert B == length.size(0)
        pmask = ~length_to_mask(length, L)

        if L > self.max_len:
            attnmask = torch.ones((L, L), device=x.device, dtype=torch.bool)
            for i in range(L):
                min_ind = max(0, i - self.max_len // 2)
                max_ind = min(L, i + self.max_len // 2)
                max_ind = max(self.max_len, max_ind)
                min_ind = min(L - self.max_len, min_ind)
                attnmask[i, min_ind:max_ind] = False
        else:
            attnmask = None

        # Transformer
        for block in self.blocks:
            x = block(x, attn_mask=attnmask, tgt_key_padding_mask=pmask)

        # Output
        sample = self.final_layer(x)
        if self.avgbeta:
            betas = (sample[..., 126:136] * (~pmask[..., None])).sum(1) / length[:, None]
            betas = repeat(betas, "b c -> b l c", l=L)
            sample = torch.cat([sample[..., :126], betas, sample[..., 136:]], dim=-1)

        # Output (extra)
        pred_cam = None
        if self.pred_cam_head:
            pred_cam = self.pred_cam_head(x)
            pred_cam = pred_cam * self.pred_cam_std + self.pred_cam_mean
            torch.clamp_min_(pred_cam[..., 0], 0.25)

        static_conf_logits = None
        if self.static_conf_head:
            static_conf_logits = self.static_conf_head(x)

        output = {
            "pred_context": x,
            "pred_x": sample,
            "pred_cam": pred_cam,
            "static_conf_logits": static_conf_logits,
        }
        return output


# ============================================================================
# Pipeline (from model/gvhmr/pipeline/gvhmr_pipeline.py -- inference only)
# ============================================================================

@autocast("cuda", enabled=False)
def get_smpl_params_w_Rt_v2(
    global_orient_gv,
    local_transl_vel,
    global_orient_c,
    cam_angvel,
):
    """Get global R,t in GV0(ay)"""
    def as_identity(R):
        is_I = matrix_to_axis_angle(R).norm(dim=-1) < 1e-5
        R[is_I] = torch.eye(3)[None].expand(is_I.sum(), -1, -1).to(R)
        return R

    B = cam_angvel.shape[0]
    R_t_to_tp1 = rotation_6d_to_matrix(cam_angvel)
    R_t_to_tp1 = as_identity(R_t_to_tp1)

    R_gv = axis_angle_to_matrix(global_orient_gv)
    R_c = axis_angle_to_matrix(global_orient_c)

    R_c2gv = R_gv @ R_c.mT
    view_axis_gv = R_c2gv[:, :, :, 2]

    R_cnext2gv = R_c2gv @ R_t_to_tp1.mT
    view_axis_gv_next = R_cnext2gv[..., 2]

    vec1_xyz = view_axis_gv.clone()
    vec1_xyz[..., 1] = 0
    vec1_xyz = F.normalize(vec1_xyz, dim=-1)
    vec2_xyz = view_axis_gv_next.clone()
    vec2_xyz[..., 1] = 0
    vec2_xyz = F.normalize(vec2_xyz, dim=-1)

    aa_tp1_to_t = vec2_xyz.cross(vec1_xyz, dim=-1)
    aa_tp1_to_t_angle = torch.acos(
        torch.clamp((vec1_xyz * vec2_xyz).sum(dim=-1, keepdim=True), -1.0, 1.0)
    )
    aa_tp1_to_t = F.normalize(aa_tp1_to_t, dim=-1) * aa_tp1_to_t_angle

    aa_tp1_to_t = gaussian_smooth(aa_tp1_to_t, dim=-2)
    R_tp1_to_t = axis_angle_to_matrix(aa_tp1_to_t).mT

    R_t_to_0 = [torch.eye(3)[None].expand(B, -1, -1).to(R_t_to_tp1)]
    for i in range(1, R_t_to_tp1.shape[1]):
        R_t_to_0.append(R_t_to_0[-1] @ R_tp1_to_t[:, i])
    R_t_to_0 = torch.stack(R_t_to_0, dim=1)
    R_t_to_0 = as_identity(R_t_to_0)

    global_orient = matrix_to_axis_angle(R_t_to_0 @ R_gv)
    transl = rollout_local_transl_vel(local_transl_vel, global_orient)
    global_orient, transl, _ = get_tgtcoord_rootparam(global_orient, transl, tsf="any->ay")

    return {"global_orient": global_orient, "transl": transl}


class Pipeline(nn.Module):
    def __init__(self, denoiser3d, endecoder, normalize_cam_angvel=True,
                 weights=None, static_conf=None, **kwargs):
        super().__init__()
        self.args = SimpleNamespace(
            weights=weights,
            normalize_cam_angvel=normalize_cam_angvel,
            static_conf=static_conf,
        )
        self.weights = weights
        self.denoiser3d = denoiser3d
        self.endecoder = endecoder

        if normalize_cam_angvel:
            cam_angvel_stats = stats_compose.cam_angvel["manual"]
            with torch.device("cpu"):
                self.register_buffer("cam_angvel_mean",
                                     torch.tensor(cam_angvel_stats["mean"]), persistent=False)
                self.register_buffer("cam_angvel_std",
                                     torch.tensor(cam_angvel_stats["std"]), persistent=False)

    def forward(self, inputs, train=False, postproc=False, static_cam=False):
        outputs = dict()
        length = inputs["length"]

        # Conditions
        cliff_cam = compute_bbox_info_bedlam(inputs["bbx_xys"], inputs["K_fullimg"])
        f_cam_angvel = inputs["cam_angvel"]
        if self.args.normalize_cam_angvel:
            f_cam_angvel = (f_cam_angvel - self.cam_angvel_mean) / self.cam_angvel_std
        f_condition = {
            "obs": inputs["obs"],
            "f_cliffcam": cliff_cam,
            "f_cam_angvel": f_cam_angvel,
            "f_imgseq": inputs["f_imgseq"],
        }

        # Forward & output
        model_output = self.denoiser3d(length=length, **f_condition)
        decode_dict = self.endecoder.decode(model_output["pred_x"])
        outputs.update({"model_output": model_output, "decode_dict": decode_dict})

        # Cast to float32 for post-processing
        def _to_f32(v):
            return v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != torch.float32 else v
        model_output = {k: _to_f32(v) for k, v in model_output.items()}
        inputs = {k: _to_f32(v) for k, v in inputs.items()}

        # Post-processing
        outputs["pred_smpl_params_incam"] = {
            "body_pose": decode_dict["body_pose"],
            "betas": decode_dict["betas"],
            "global_orient": decode_dict["global_orient"],
            "transl": compute_transl_full_cam(
                model_output["pred_cam"], inputs["bbx_xys"], inputs["K_fullimg"]
            ),
        }

        pred_smpl_params_global = get_smpl_params_w_Rt_v2(
            global_orient_gv=decode_dict["global_orient_gv"],
            local_transl_vel=decode_dict["local_transl_vel"],
            global_orient_c=decode_dict["global_orient"],
            cam_angvel=inputs["cam_angvel"],
        )
        outputs["pred_smpl_params_global"] = {
            "body_pose": decode_dict["body_pose"],
            "betas": decode_dict["betas"],
            **pred_smpl_params_global,
        }
        outputs["static_conf_logits"] = model_output["static_conf_logits"]

        if postproc:
            if static_cam:
                outputs["pred_smpl_params_global"]["transl"] = pp_static_joint_cam(
                    outputs, self.endecoder
                )
            else:
                outputs["pred_smpl_params_global"]["transl"] = pp_static_joint(
                    outputs, self.endecoder
                )
            body_pose = process_ik(outputs, self.endecoder)
            decode_dict["body_pose"] = body_pose
            outputs["pred_smpl_params_global"]["body_pose"] = body_pose
            outputs["pred_smpl_params_incam"]["body_pose"] = body_pose

        return outputs


# ============================================================================
# DemoPL (from model/gvhmr/gvhmr_pl_demo.py -- converted to nn.Module)
# ============================================================================

class DemoPL(nn.Module):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    @torch.no_grad()
    def predict(self, data, static_cam=False):
        batch = {
            "length": data["length"][None],
            "obs": normalize_kp2d(data["kp2d"], data["bbx_xys"])[None],
            "bbx_xys": data["bbx_xys"][None],
            "K_fullimg": data["K_fullimg"][None],
            "cam_angvel": data["cam_angvel"][None],
            "f_imgseq": data["f_imgseq"][None],
        }
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        batch = {
            k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)
            for k, v in batch.items()
        }
        outputs = self.pipeline.forward(batch, train=False, postproc=True, static_cam=static_cam)

        pred = {
            "smpl_params_global": {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()},
            "smpl_params_incam": {k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()},
            "K_fullimg": data["K_fullimg"],
            "net_outputs": outputs,
        }
        return pred

    def load_pretrained_model(self, ckpt_path):
        Log.info(f"[PL-Trainer] Loading ckpt type: {ckpt_path}")
        import comfy.utils
        state_dict = comfy.utils.load_torch_file(str(ckpt_path))
        missing, unexpected = self.load_state_dict(state_dict, strict=False, assign=True)
        if len(missing) > 0:
            Log.warn(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            Log.warn(f"Unexpected keys: {unexpected}")
