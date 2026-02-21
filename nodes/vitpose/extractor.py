"""
VitPoseExtractor and heatmap utility functions.

Absorbed from:
  - vendor/hmr4d/utils/preproc/vitpose.py
"""

import logging

import torch
import numpy as np
from pathlib import Path
import comfy.model_management

log = logging.getLogger("motioncapture")

from .model import build_model
from .feat_extractor import get_batch
from tqdm import tqdm

from ..motion_utils.kp2d_utils import keypoints_from_heatmaps
from ..motion_utils.flip_utils import flip_heatmap_coco17


class VitPoseExtractor:
    def __init__(self, tqdm_leave=True, dtype=None, ckpt_path=None):
        self.device = comfy.model_management.get_torch_device()
        if ckpt_path is None:
            import folder_paths
            ckpt_path = Path(folder_paths.models_dir) / "motion_capture" / "vitpose.safetensors"
        self.pose = build_model("ViTPose_huge_coco_256x192", str(ckpt_path))
        self.dtype = dtype
        # Keep on CPU -- ModelPatcher handles device placement via load_models_gpu()
        if dtype is not None:
            self.pose.to(dtype=dtype)
        self.pose.eval()

        self.flip_test = True
        self.tqdm_leave = tqdm_leave

    @torch.no_grad()
    def extract(self, video_path, bbx_xys, img_ds=0.5):
        # Get the batch
        if isinstance(video_path, str):
            imgs, bbx_xys = get_batch(video_path, bbx_xys, img_ds=img_ds)
        else:
            assert isinstance(video_path, torch.Tensor)
            imgs = video_path

        # Inference
        L, _, H, W = imgs.shape  # (L, 3, H, W)
        batch_size = 8  # Reduced from 16 for lower memory usage
        vitpose = []
        for j in tqdm(range(0, L, batch_size), desc="ViTPose", leave=self.tqdm_leave):
            # Heat map
            imgs_batch = imgs[j : j + batch_size, :, :, 32:224].to(device=self.device, dtype=self.dtype)
            if self.flip_test:
                heatmap, heatmap_flipped = self.pose(torch.cat([imgs_batch, imgs_batch.flip(3)], dim=0)).chunk(2)
                heatmap_flipped = flip_heatmap_coco17(heatmap_flipped)
                heatmap = (heatmap + heatmap_flipped) * 0.5
                del heatmap_flipped
            else:
                heatmap = self.pose(imgs_batch.clone())  # (B, J, 64, 48)

            # postprocess from mmpose
            bbx_xys_batch = bbx_xys[j : j + batch_size]
            heatmap = heatmap.clone().cpu().float().numpy()
            center = bbx_xys_batch[:, :2].numpy()
            scale = (torch.cat((bbx_xys_batch[:, [2]] * 24 / 32, bbx_xys_batch[:, [2]]), dim=1) / 200).numpy()
            preds, maxvals = keypoints_from_heatmaps(heatmaps=heatmap, center=center, scale=scale, use_udp=True)
            kp2d = np.concatenate((preds, maxvals), axis=-1)
            kp2d = torch.from_numpy(kp2d)

            vitpose.append(kp2d.detach().cpu().clone())

            # Periodic memory cleanup to prevent fragmentation
            if j > 0 and j % (batch_size * 4) == 0:
                comfy.model_management.soft_empty_cache()

        vitpose = torch.cat(vitpose, dim=0).clone()  # (F, 17, 3)
        return vitpose
