import torch
from hmr4d.network.hmr2 import load_hmr2, HMR2


from hmr4d.utils.video_io_utils import read_video_np
import cv2
import numpy as np

from hmr4d.network.hmr2.utils.preproc import crop_and_resize, IMAGE_MEAN, IMAGE_STD
from tqdm import tqdm


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
        for i in range(len(imgs)):
            d = ds_factors[i]
            if d > 1.1:
                imgs[i] = cv2.GaussianBlur(imgs[i], (5, 5), (d - 1) / 2)

    # Output (pre-allocate to avoid large temporary lists)
    imgs_out = torch.empty((len(imgs), 3, img_dst_size, img_dst_size), dtype=torch.float32)
    bbx_xys_ds_out = torch.empty((len(imgs), 3), dtype=torch.float32)
    
    for i in range(len(imgs)):
        img, bbx_xys_ds = crop_and_resize(
            imgs[i],
            gt_center[i] * img_ds,
            gt_bbx_size[i] * img_ds,
            img_dst_size,
            enlarge_ratio=1.0,
        )
        # Normalize and store directly in output tensor
        # (img is uint8 H,W,3 RGB)
        img_tensor = torch.from_numpy(img).float()
        imgs_out[i] = ((img_tensor / 255.0 - IMAGE_MEAN) / IMAGE_STD).permute(2, 0, 1)
        bbx_xys_ds_out[i] = torch.from_numpy(bbx_xys_ds)

    bbx_xys = bbx_xys_ds_out / img_ds
    return imgs_out, bbx_xys


class Extractor:
    def __init__(self, tqdm_leave=True, device="cuda", batch_size=16):
        self.device = device
        self.batch_size = batch_size
        self.extractor: HMR2 = load_hmr2().to(self.device).eval()
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
        batch_size = self.batch_size
        features = []
        for j in tqdm(range(0, F, batch_size), desc="HMR2 Feature", leave=self.tqdm_leave):
            imgs_batch = imgs[j : j + batch_size].to(self.device)

            with torch.no_grad():
                feature = self.extractor({"img": imgs_batch})
                features.append(feature.detach().cpu())

        features = torch.cat(features, dim=0).clone()  # (F, 1024)
        return features
