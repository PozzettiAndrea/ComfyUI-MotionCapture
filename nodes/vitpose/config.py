"""
ViTPose_huge_coco_256x192 config as a Python dict.

Absorbed from:
  - builder/configs/coco/ViTPose_huge_coco_256x192.py
"""

model = dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='ViT',
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
        type='TopdownHeatmapSimpleHead',
        in_channels=1280,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1),
        out_channels=17,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
    ),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type='GaussianHeatmap',
        modulate_kernel=11,
        use_udp=True,
    ),
)
