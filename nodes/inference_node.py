"""
GVHMRInference Node - Performs motion capture inference on video with SAM3 masks
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import cv2
from typing import Dict, Tuple
from tqdm import tqdm

# Add vendor path for GVHMR
VENDOR_PATH = Path(__file__).parent.parent / "vendor"
sys.path.insert(0, str(VENDOR_PATH))

# Import GVHMR components
from hmr4d.utils.pylogger import Log
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, create_camera_sensor
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.net_utils import to_cuda

# Import local utilities
from .utils import (
    extract_bboxes_from_masks,
    bbox_to_xyxy,
    expand_bbox,
    normalize_image_tensor,
    validate_masks,
    validate_images,
)


class GVHMRInference:
    """
    ComfyUI node for GVHMR motion capture inference.
    Takes video frames and SAM3 masks, outputs SMPL parameters and 3D mesh.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # ComfyUI IMAGE tensor (B, H, W, C)
                "masks": ("MASK",),  # SAM3 masks (B, H, W)
                "model": ("GVHMR_MODEL",),  # Model bundle from LoadGVHMRModels
                "static_camera": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Set to True if camera is stationary (skips visual odometry)"
                }),
            },
            "optional": {
                "focal_length_mm": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 300,
                    "tooltip": "Camera focal length in mm (0 = auto-estimate). For phones: 13-77mm typical"
                }),
                "bbox_scale": ("FLOAT", {
                    "default": 1.2,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Expand bounding box by this factor to ensure full person capture"
                }),
            }
        }

    RETURN_TYPES = ("SMPL_PARAMS", "IMAGE", "STRING")
    RETURN_NAMES = ("smpl_params", "visualization", "info")
    FUNCTION = "run_inference"
    CATEGORY = "MotionCapture/GVHMR"

    def prepare_data_from_masks(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        model_bundle: Dict,
        static_camera: bool,
        focal_length_mm: int,
        bbox_scale: float,
    ) -> Dict:
        """
        Prepare data dictionary for GVHMR inference from images and masks.
        """
        # Validate inputs
        validate_images(images)
        validate_masks(masks)

        device = model_bundle["device"]
        batch_size, height, width, channels = images.shape

        # Convert images to numpy for processing
        images_np = (images.cpu().numpy() * 255).astype(np.uint8)

        # Extract bounding boxes from masks
        Log.info("[GVHMRInference] Extracting bounding boxes from SAM3 masks...")
        bboxes_xywh = extract_bboxes_from_masks(masks)

        # Expand bounding boxes
        bboxes_xywh = [
            expand_bbox(bbox, scale=bbox_scale, img_width=width, img_height=height)
            for bbox in bboxes_xywh
        ]

        # Convert to xyxy format for processing
        bboxes_xyxy = torch.tensor([bbox_to_xyxy(bbox) for bbox in bboxes_xywh], dtype=torch.float32)

        # Get bbx_xys format (used by GVHMR)
        bbx_xys = get_bbx_xys_from_xyxy(bboxes_xyxy, base_enlarge=1.0).float()  # Already expanded above

        # Extract ViTPose 2D keypoints
        Log.info("[GVHMRInference] Extracting 2D pose with ViTPose...")
        vitpose_extractor = model_bundle["vitpose_extractor"]

        # ViTPose expects video frames as numpy array (F, H, W, 3)
        kp2d_list = []
        for i, img in enumerate(images_np):
            # Crop to bbox
            x, y, w, h = bboxes_xywh[i]
            cropped = img[y:y+h, x:x+w]

            # Resize to 256x256 for ViTPose
            resized = cv2.resize(cropped, (256, 256))

            # Extract keypoints (this is a simplified approach; actual implementation may vary)
            # The real ViTPose extractor processes the whole video at once
            kp2d_list.append(resized)

        # Use the actual extractor (note: this expects a video path, so we may need to adapt)
        # For now, we'll create a temporary processing pipeline
        kp2d = vitpose_extractor.extract_from_images(images_np, bbx_xys)

        # Extract ViT features
        Log.info("[GVHMRInference] Extracting image features...")
        feature_extractor = model_bundle["feature_extractor"]
        f_imgseq = feature_extractor.extract_from_images(images_np, bbx_xys)

        # Estimate camera intrinsics
        if focal_length_mm > 0:
            K_fullimg = create_camera_sensor(width, height, focal_length_mm)[2].repeat(batch_size, 1, 1)
        else:
            K_fullimg = estimate_K(width, height).repeat(batch_size, 1, 1)

        # Handle camera motion
        if static_camera:
            R_w2c = torch.eye(3).repeat(batch_size, 1, 1)
        else:
            # For moving camera, we would need to run visual odometry
            # For now, default to static if not implemented
            Log.warn("[GVHMRInference] Moving camera mode not fully implemented, using static")
            R_w2c = torch.eye(3).repeat(batch_size, 1, 1)

        cam_angvel = compute_cam_angvel(R_w2c)

        # Prepare data dictionary
        data = {
            "length": torch.tensor(batch_size),
            "bbx_xys": bbx_xys,
            "kp2d": kp2d,
            "K_fullimg": K_fullimg,
            "cam_angvel": cam_angvel,
            "f_imgseq": f_imgseq,
        }

        return data, K_fullimg

    def run_inference(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        model: Dict,
        static_camera: bool = True,
        focal_length_mm: int = 0,
        bbox_scale: float = 1.2,
    ):
        """
        Run GVHMR inference on images with SAM3 masks.
        """
        try:
            Log.info("[GVHMRInference] Starting GVHMR inference...")
            Log.info(f"[GVHMRInference] Input shape: {images.shape}, Masks shape: {masks.shape}")

            # Prepare data
            data, K_fullimg = self.prepare_data_from_masks(
                images, masks, model, static_camera, focal_length_mm, bbox_scale
            )

            # Run GVHMR inference
            Log.info("[GVHMRInference] Running GVHMR model...")
            gvhmr_model = model["gvhmr"]
            device = model["device"]

            with torch.no_grad():
                pred = gvhmr_model.predict(data, static_cam=static_camera)

            # Extract SMPL parameters
            smpl_params = {
                "global": pred["smpl_params_global"],
                "incam": pred["smpl_params_incam"],
                "K_fullimg": K_fullimg,
            }

            # Create visualization
            Log.info("[GVHMRInference] Rendering visualization...")
            viz_frames = self.render_visualization(images, smpl_params, model)

            # Create info string
            num_frames = images.shape[0]
            info = (
                f"GVHMR Inference Complete\n"
                f"Frames processed: {num_frames}\n"
                f"Static camera: {static_camera}\n"
                f"Device: {device}\n"
            )

            Log.info("[GVHMRInference] Inference complete!")
            return (smpl_params, viz_frames, info)

        except Exception as e:
            error_msg = f"GVHMR Inference failed: {str(e)}"
            Log.error(error_msg)
            import traceback
            traceback.print_exc()
            # Return empty results on error
            return (None, images, error_msg)

    def render_visualization(
        self,
        images: torch.Tensor,
        smpl_params: Dict,
        model: Dict,
    ) -> torch.Tensor:
        """
        Render SMPL mesh overlay on input images.
        """
        try:
            device = model["device"]
            batch_size, height, width, channels = images.shape

            # Initialize SMPL model
            smplx = make_smplx("supermotion").to(device)
            smplx2smpl = torch.load(
                str(Path(__file__).parent.parent / "vendor" / "hmr4d" / "utils" / "body_model" / "smplx2smpl_sparse.pt")
            ).to(device)

            # Get SMPL vertices
            smplx_out = smplx(**to_cuda(smpl_params["incam"]))
            pred_verts = torch.stack([torch.matmul(smplx2smpl, v) for v in smplx_out.vertices])

            # Initialize renderer
            from hmr4d.utils.vis.renderer import Renderer
            faces_smpl = make_smplx("smpl").faces
            K = smpl_params["K_fullimg"][0]
            renderer = Renderer(width, height, device=device, faces=faces_smpl, K=K)

            # Render each frame
            rendered_frames = []
            images_np = (images.cpu().numpy() * 255).astype(np.uint8)

            for i in range(batch_size):
                img_rendered = renderer.render_mesh(
                    pred_verts[i].to(device),
                    images_np[i],
                    [0.8, 0.8, 0.8]  # Mesh color
                )
                rendered_frames.append(img_rendered)

            # Convert back to torch tensor
            rendered_tensor = torch.from_numpy(np.stack(rendered_frames)).float() / 255.0

            return rendered_tensor

        except Exception as e:
            Log.error(f"Visualization rendering failed: {e}")
            # Return original images if rendering fails
            return images


# Node registration
NODE_CLASS_MAPPINGS = {
    "GVHMRInference": GVHMRInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GVHMRInference": "GVHMR Inference",
}
