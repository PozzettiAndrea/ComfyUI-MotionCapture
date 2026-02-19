"""
HMR2 feature extractor and batch preparation for ViTPose.

Absorbed from:
  - vendor/hmr4d/utils/preproc/vitfeat_extractor.py
"""

import torch
import cv2
import numpy as np
from tqdm import tqdm
import comfy.model_management

from ..hmr2 import load_hmr2, HMR2
from ..motion_utils.video_io_utils import read_video_np


# ---------------------------------------------------------------------------
# Inlined from vendor/hmr4d/network/hmr2/utils/preproc.py
# ---------------------------------------------------------------------------

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225])


def crop_and_resize(img, bbx_xy, bbx_s, dst_size=256, enlarge_ratio=1.2):
    """
    Args:
        img: (H, W, 3)
        bbx_xy: (2,)
        bbx_s: scalar
    """
    hs = bbx_s * enlarge_ratio / 2
    src = np.stack(
        [
            bbx_xy - hs,  # left-up corner
            bbx_xy + np.array([hs, -hs]),  # right-up corner
            bbx_xy,  # center
        ]
    ).astype(np.float32)
    dst = np.array(
        [[0, 0], [dst_size - 1, 0], [dst_size / 2 - 0.5, dst_size / 2 - 0.5]],
        dtype=np.float32,
    )
    A = cv2.getAffineTransform(src, dst)
    img_crop = cv2.warpAffine(img, A, (dst_size, dst_size), flags=cv2.INTER_LINEAR)
    bbx_xys_final = np.array([*bbx_xy, bbx_s * enlarge_ratio])
    return img_crop, bbx_xys_final


# ---------------------------------------------------------------------------
# Batch preparation
# ---------------------------------------------------------------------------

def get_batch(input_path, bbx_xys, img_ds=0.5, img_dst_size=256, path_type="video"):
    if path_type == "video":
        imgs = read_video_np(input_path, scale=img_ds)
    elif path_type == "image":
        imgs = cv2.imread(str(input_path))[..., ::-1]
        imgs = cv2.resize(imgs, (0, 0), fx=img_ds, fy=img_ds)
        imgs = imgs[None]
    elif path_type == "np":
        assert isinstance(input_path, np.ndarray)
        assert img_ds == 1.0  # this is safe
        imgs = input_path

    gt_center = bbx_xys[:, :2]
    gt_bbx_size = bbx_xys[:, 2]

    # Blur image to avoid aliasing artifacts
    if True:
        gt_bbx_size_ds = gt_bbx_size * img_ds
        ds_factors = ((gt_bbx_size_ds * 1.0) / img_dst_size / 2.0).numpy()
        imgs = np.stack(
            [
                cv2.GaussianBlur(v, (5, 5), (d - 1) / 2) if d > 1.1 else v
                for v, d in zip(imgs, ds_factors)
            ]
        )

    # Output
    imgs_list = []
    bbx_xys_ds_list = []
    for i in range(len(imgs)):
        img, bbx_xys_ds = crop_and_resize(
            imgs[i],
            gt_center[i] * img_ds,
            gt_bbx_size[i] * img_ds,
            img_dst_size,
            enlarge_ratio=1.0,
        )
        imgs_list.append(img)
        bbx_xys_ds_list.append(bbx_xys_ds)
    imgs = torch.from_numpy(np.stack(imgs_list))  # (F, 256, 256, 3), RGB
    bbx_xys = torch.from_numpy(np.stack(bbx_xys_ds_list)) / img_ds  # (F, 3)

    imgs = ((imgs / 255.0 - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2)  # (F, 3, 256, 256)
    return imgs, bbx_xys


# ---------------------------------------------------------------------------
# HMR2 Feature Extractor
# ---------------------------------------------------------------------------

class Extractor:
    def __init__(self, tqdm_leave=True, dtype=None, ckpt_path=None):
        self.device = comfy.model_management.get_torch_device()
        self.dtype = dtype
        model = load_hmr2(checkpoint_path=ckpt_path) if ckpt_path else load_hmr2()
        # Keep on CPU -- ModelPatcher handles device placement via load_models_gpu()
        if dtype is not None:
            self.extractor: HMR2 = model.to(dtype=dtype).eval()
        else:
            self.extractor: HMR2 = model.eval()
        self.tqdm_leave = tqdm_leave

    def extract_video_features(self, video_path, bbx_xys, img_ds=0.5):
        """
        img_ds makes the image smaller, which is useful for faster processing
        """
        # Get the batch
        if isinstance(video_path, str):
            imgs, bbx_xys = get_batch(video_path, bbx_xys, img_ds=img_ds)
        else:
            assert isinstance(video_path, torch.Tensor)
            imgs = video_path

        # Inference
        F, _, H, W = imgs.shape  # (F, 3, H, W)
        # Keep imgs on CPU, only move batch to GPU (saves memory for long videos)
        batch_size = 8  # Reduced from 16 for lower memory usage (~2.5GB GPU)
        features = []
        for j in tqdm(range(0, F, batch_size), desc="HMR2 Feature", leave=self.tqdm_leave):
            imgs_batch = imgs[j : j + batch_size].to(device=self.device, dtype=self.dtype)

            with torch.no_grad():
                feature = self.extractor({"img": imgs_batch})
                features.append(feature.detach().cpu())

            # Periodic memory cleanup to prevent fragmentation
            if j > 0 and j % (batch_size * 4) == 0:
                comfy.model_management.soft_empty_cache()

        features = torch.cat(features, dim=0).clone()  # (F, 1024)
        return features
