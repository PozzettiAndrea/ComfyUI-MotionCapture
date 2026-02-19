"""
Consolidated ViTPose model -- ViT backbone, heatmap heads, and model builder,
migrated to ComfyUI-native format with operations= threading.

Absorbed from:
  - builder/backbones/vit.py
  - builder/heads/topdown_heatmap_base_head.py
  - builder/heads/topdown_heatmap_simple_head.py
  - model_builder.py
"""

import logging
import math
from abc import ABCMeta, abstractmethod
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import comfy.ops
from comfy.ldm.modules.attention import optimized_attention

ops = comfy.ops.manual_cast

log = logging.getLogger("motioncapture")


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


def _trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
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
# ViT backbone (from builder/backbones/vit.py)
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


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

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = operations.Linear(dim, all_head_dim * 3, bias=qkv_bias, dtype=dtype, device=device)

        self.proj = operations.Linear(all_head_dim, dim, dtype=dtype, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Use ComfyUI optimized attention
        # q, k, v are (B, H, N, D) -- use skip_reshape=True since already reshaped
        out = optimized_attention(q, k, v, heads=self.num_heads,
                                  skip_reshape=True, skip_output_reshape=True)
        # out is (B, H, N, D), reshape to (B, N, C)
        x = out.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=None, attn_head_dim=None,
                 dtype=None, device=None, operations=ops):
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(operations.LayerNorm, dtype=dtype, device=device)

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
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
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)
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
    """Vision Transformer backbone for ViTPose."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, use_checkpoint=False,
                 frozen_stages=-1, ratio=1, last_norm=True,
                 patch_padding='pad', freeze_attn=False, freeze_ffn=False,
                 dtype=None, device=None, operations=ops):
        super(ViT, self).__init__()

        if norm_layer is None:
            norm_layer = partial(operations.LayerNorm, eps=1e-6, dtype=dtype, device=device)

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

        # since the pretraining model has class token
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim, dtype=dtype, device=device)
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer,
                dtype=dtype, device=device, operations=operations,
            )
            for i in range(depth)
        ])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            _trunc_normal_(self.pos_embed, std=0.02)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

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


# ---------------------------------------------------------------------------
# Heatmap heads (from builder/heads/)
# ---------------------------------------------------------------------------

class TopdownHeatmapBaseHead(nn.Module):
    """Base class for top-down heatmap heads."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_loss(self, **kwargs):
        """Gets the loss."""

    @abstractmethod
    def get_accuracy(self, **kwargs):
        """Gets the accuracy."""

    @abstractmethod
    def forward(self, **kwargs):
        """Forward function."""

    @abstractmethod
    def inference_model(self, **kwargs):
        """Inference function."""

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')
        return deconv_kernel, padding, output_padding


class TopdownHeatmapSimpleHead(TopdownHeatmapBaseHead):
    """Top-down heatmap simple head with deconv layers and a final conv layer.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_deconv_layers (int): Number of deconv layers (>= 0)
        num_deconv_filters (list|tuple): Number of filters
        num_deconv_kernels (list|tuple): Kernel sizes
        extra (dict|None): Extra configuration
    """

    def __init__(self, in_channels, out_channels, num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4), extra=None, in_index=0,
                 input_transform=None, align_corners=False,
                 loss_keypoint=None, train_cfg=None, test_cfg=None,
                 upsample=0, dtype=None, device=None, operations=ops):
        super().__init__()

        self.in_channels = in_channels
        self.loss = None
        self.upsample = upsample
        self.dtype = dtype
        self.device = device
        self.operations = operations

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')

        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers, num_deconv_filters, num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')

        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            conv_channels = num_deconv_filters[-1] if num_deconv_layers > 0 else self.in_channels

            layers = []
            if extra is not None:
                num_conv_layers = extra.get('num_conv_layers', 0)
                num_conv_kernels = extra.get('num_conv_kernels',
                                             [1] * num_conv_layers)
                for i in range(num_conv_layers):
                    layers.append(
                        operations.Conv2d(
                            in_channels=conv_channels,
                            out_channels=conv_channels,
                            kernel_size=num_conv_kernels[i],
                            stride=1,
                            padding=(num_conv_kernels[i] - 1) // 2,
                            dtype=dtype, device=device,
                        ))
                    layers.append(nn.BatchNorm2d(conv_channels))
                    layers.append(nn.ReLU(inplace=True))

            layers.append(
                operations.Conv2d(
                    in_channels=conv_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dtype=dtype, device=device,
                ))

            if len(layers) > 1:
                self.final_layer = nn.Sequential(*layers)
            else:
                self.final_layer = layers[0]

    def get_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss."""
        losses = dict()
        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 4 and target_weight.dim() == 3
        losses['heatmap_loss'] = self.loss(output, target, target_weight)
        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss."""
        accuracy = dict()
        return accuracy

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    def inference_model(self, x, flip_pairs=None):
        """Inference function."""
        output = self.forward(x)
        if flip_pairs is not None:
            output_heatmap = output.detach().cpu().numpy()
        else:
            output_heatmap = output.detach().cpu().numpy()
        return output_heatmap

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms."""
        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder."""
        if not isinstance(inputs, list):
            if self.upsample > 0:
                inputs = F.interpolate(
                    F.relu(inputs),
                    scale_factor=self.upsample,
                    mode='bilinear',
                    align_corners=self.align_corners,
                )
            return inputs

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                F.interpolate(
                    x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners,
                ) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = (f'num_layers({num_layers}) '
                         f'!= length of num_filters({len(num_filters)})')
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = (f'num_layers({num_layers}) '
                         f'!= length of num_kernels({len(num_kernels)})')
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                ))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Model builder (from model_builder.py)
# ---------------------------------------------------------------------------

models = {
    "ViTPose_huge_coco_256x192": dict(
        type="TopDown",
        pretrained=None,
        backbone=dict(
            type="ViT",
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
        ),
        keypoint_head=dict(
            type="TopdownHeatmapSimpleHead",
            in_channels=1280,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1),
            out_channels=17,
            loss_keypoint=dict(type="JointsMSELoss", use_target_weight=True),
        ),
        train_cfg=dict(),
        test_cfg=dict(),
    ),
    "ViTPose_base_coco_256x192": dict(
        type="TopDown",
        pretrained=None,
        backbone=dict(
            type="ViT",
            img_size=(256, 192),
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            ratio=1,
            use_checkpoint=False,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.3,
        ),
        keypoint_head=dict(
            type="TopdownHeatmapSimpleHead",
            in_channels=768,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1),
            out_channels=17,
            loss_keypoint=dict(type="JointsMSELoss", use_target_weight=True),
        ),
        train_cfg=dict(),
        test_cfg=dict(),
    ),
    "ViTPose_base_simple_coco_256x192": dict(
        type="TopDown",
        pretrained=None,
        backbone=dict(
            type="ViT",
            img_size=(256, 192),
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            ratio=1,
            use_checkpoint=False,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.3,
        ),
        keypoint_head=dict(
            type="TopdownHeatmapSimpleHead",
            in_channels=768,
            num_deconv_layers=0,
            num_deconv_filters=[],
            num_deconv_kernels=[],
            upsample=4,
            extra=dict(final_conv_kernel=3),
            out_channels=17,
            loss_keypoint=dict(type="JointsMSELoss", use_target_weight=True),
        ),
        train_cfg=dict(),
        test_cfg=dict(
            flip_test=True,
            post_process="default",
            shift_heatmap=False,
            target_type="GaussianHeatmap",
            modulate_kernel=11,
            use_udp=True,
        ),
    ),
}


class VitPoseModel(nn.Module):
    """Combined ViTPose model: backbone + keypoint head."""

    def __init__(self, backbone, keypoint_head):
        super(VitPoseModel, self).__init__()
        self.backbone = backbone
        self.keypoint_head = keypoint_head

    def forward(self, x):
        x = self.backbone(x)
        x = self.keypoint_head(x)
        return x


def build_model(model_name, checkpoint=None, dtype=None, device=None, operations=ops):
    """Build a ViTPose model from a config name.

    Args:
        model_name: Key into the ``models`` dict (e.g. "ViTPose_huge_coco_256x192").
        checkpoint: Optional path to a checkpoint file to load weights from.
        dtype: Optional torch dtype.
        device: Optional torch device.
        operations: ComfyUI operations namespace (default: comfy.ops.manual_cast).

    Returns:
        A VitPoseModel instance.
    """
    try:
        model = models[model_name]
    except Exception as e:
        log.warning("Unknown VitPose model name %r: %s", model_name, e)
        raise ValueError("not a correct config")

    head = TopdownHeatmapSimpleHead(
        in_channels=model["keypoint_head"]["in_channels"],
        out_channels=model["keypoint_head"]["out_channels"],
        num_deconv_filters=model["keypoint_head"]["num_deconv_filters"],
        num_deconv_kernels=model["keypoint_head"]["num_deconv_kernels"],
        num_deconv_layers=model["keypoint_head"]["num_deconv_layers"],
        extra=model["keypoint_head"]["extra"],
        upsample=model["keypoint_head"].get("upsample", 0),
        dtype=dtype, device=device, operations=operations,
    )

    backbone = ViT(
        img_size=model["backbone"]["img_size"],
        patch_size=model["backbone"]["patch_size"],
        embed_dim=model["backbone"]["embed_dim"],
        depth=model["backbone"]["depth"],
        num_heads=model["backbone"]["num_heads"],
        ratio=model["backbone"]["ratio"],
        mlp_ratio=model["backbone"]["mlp_ratio"],
        qkv_bias=model["backbone"]["qkv_bias"],
        drop_path_rate=model["backbone"]["drop_path_rate"],
        dtype=dtype, device=device, operations=operations,
    )

    if checkpoint is not None:
        with torch.device("meta"):
            pose = VitPoseModel(backbone, head)

        import comfy.utils
        state_dict = comfy.utils.load_torch_file(str(checkpoint))
        # ViTPose checkpoints wrap weights under "state_dict" key
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        pose.load_state_dict(state_dict, strict=False, assign=True)
    else:
        pose = VitPoseModel(backbone, head)

    return pose
