"""
ViTPose model — ViT backbone + heatmap head, inference only.

Uses shared ViT blocks from nodes/shared_vit.py.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops

from ..shared_vit import ViT, ops

log = logging.getLogger("motioncapture")


# ---------------------------------------------------------------------------
# Heatmap head (inference only, training code removed)
# ---------------------------------------------------------------------------

class TopdownHeatmapSimpleHead(nn.Module):
    """Top-down heatmap head with deconv layers and a final conv layer."""

    def __init__(self, in_channels, out_channels, num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4), extra=None,
                 upsample=0, dtype=None, device=None, operations=ops):
        super().__init__()

        self.in_channels = in_channels
        self.upsample = upsample

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers, num_deconv_filters, num_deconv_kernels,
                dtype=dtype, device=device, operations=operations,
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

    def forward(self, x):
        if not isinstance(x, list):
            if self.upsample > 0:
                x = F.interpolate(
                    F.relu(x),
                    scale_factor=self.upsample,
                    mode='bilinear',
                    align_corners=False,
                )
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
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

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels,
                           dtype=None, device=None, operations=ops):
        if num_layers != len(num_filters):
            raise ValueError(f'num_layers({num_layers}) != length of num_filters({len(num_filters)})')
        if num_layers != len(num_kernels):
            raise ValueError(f'num_layers({num_layers}) != length of num_kernels({len(num_kernels)})')

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            layers.append(
                operations.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                    dtype=dtype, device=device,
                ))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

MODELS = {
    "ViTPose_huge_coco_256x192": dict(
        backbone=dict(
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
            in_channels=1280,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1),
            out_channels=17,
        ),
    ),
}


class VitPoseModel(nn.Module):
    """Combined ViTPose model: backbone + keypoint head."""

    def __init__(self, backbone, keypoint_head):
        super().__init__()
        self.backbone = backbone
        self.keypoint_head = keypoint_head

    def forward(self, x):
        x = self.backbone(x)
        x = self.keypoint_head(x)
        return x


def build_model(model_name, checkpoint=None, dtype=None, device=None, operations=ops):
    """Build a ViTPose model from a config name."""
    try:
        model = MODELS[model_name]
    except KeyError:
        raise ValueError(f"Unknown ViTPose model name: {model_name!r}")

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
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        pose.load_state_dict(state_dict, strict=False, assign=True)
    else:
        pose = VitPoseModel(backbone, head)

    return pose
