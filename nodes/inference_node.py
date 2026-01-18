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
from hmr4d.utils.preproc.relpose.simple_vo import SimpleVO

# Import local utilities
from .utils import (
    extract_bboxes_from_masks,
    bbox_to_xyxy,
    expand_bbox,
    normalize_image_tensor,
    validate_masks,
    validate_images,
)


def _save_temp_video(images_np: np.ndarray, fps: int = 30) -> str:
    """Save numpy images to temporary video file for SimpleVO."""
    import tempfile
    temp_path = tempfile.mktemp(suffix=".mp4")
    h, w = images_np.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))
    for frame in images_np:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return temp_path


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
                    "tooltip": "Camera focal length in mm (0 = auto-estimate). Ignored if intrinsics input is connected. Phones: 13-77mm typical"
                }),
                "bbox_scale": ("FLOAT", {
                    "default": 1.2,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Expand bounding box by this factor to ensure full person capture"
                }),
                "vo_scale": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Scale factor for visual odometry processing (lower = faster, only used when static_camera=False)"
                }),
                "vo_step": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 30,
                    "tooltip": "Frame step for feature matching in VO (higher = faster but less accurate, only used when static_camera=False)"
                }),
                "intrinsics": ("INTRINSICS", {
                    "tooltip": "Camera intrinsics matrix (3x3). Connect from DepthAnything V3 or other source. Overrides focal_length_mm if provided."
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
        vo_scale: float,
        vo_step: int,
        intrinsics: torch.Tensor = None,
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

        # Use get_batch to preprocess images for extractors
        from hmr4d.utils.preproc.vitfeat_extractor import get_batch

        # Prepare images in the right format for get_batch (expects (F, H, W, 3) RGB numpy array)
        imgs_tensor, bbx_xys_processed = get_batch(images_np, bbx_xys, img_ds=1.0, path_type="np")

        # Extract 2D keypoints with ViTPose
        kp2d = vitpose_extractor.extract(imgs_tensor, bbx_xys_processed, img_ds=1.0)

        # Extract ViT features
        Log.info("[GVHMRInference] Extracting image features...")
        feature_extractor = model_bundle["feature_extractor"]
        f_imgseq = feature_extractor.extract_video_features(imgs_tensor, bbx_xys_processed, img_ds=1.0)

        # Camera intrinsics: use provided K, or estimate from focal_length_mm, or auto-estimate
        if intrinsics is not None:
            Log.info(f"[GVHMRInference] Using provided camera intrinsics, input shape: {intrinsics.shape}")
            # Squeeze extra dimensions (DA3 outputs [1, 1, 3, 3])
            K = intrinsics.squeeze()
            while K.dim() > 3:
                K = K.squeeze(0)
            # Handle different input shapes
            if K.dim() == 2:  # (3, 3) - single matrix
                K_fullimg = K.unsqueeze(0).repeat(batch_size, 1, 1)
            elif K.dim() == 3 and K.shape[0] == 1:  # (1, 3, 3)
                K_fullimg = K.repeat(batch_size, 1, 1)
            elif K.dim() == 3 and K.shape[0] == batch_size:  # (B, 3, 3)
                K_fullimg = K
            else:
                Log.warn(f"[GVHMRInference] Unexpected intrinsics shape {K.shape}, falling back to estimation")
                K_fullimg = estimate_K(width, height).repeat(batch_size, 1, 1)
            Log.info(f"[GVHMRInference] Intrinsics fx={K_fullimg[0,0,0]:.1f}, fy={K_fullimg[0,1,1]:.1f}, cx={K_fullimg[0,0,2]:.1f}, cy={K_fullimg[0,1,2]:.1f}")
        elif focal_length_mm > 0:
            Log.info(f"[GVHMRInference] Using focal length: {focal_length_mm}mm")
            K_fullimg = create_camera_sensor(width, height, focal_length_mm)[2].repeat(batch_size, 1, 1)
        else:
            Log.info("[GVHMRInference] Auto-estimating camera intrinsics (53° FOV)")
            K_fullimg = estimate_K(width, height).repeat(batch_size, 1, 1)

        # Handle camera motion
        t_w2c = None  # Camera translation (for future use)
        if static_camera:
            R_w2c = torch.eye(3).repeat(batch_size, 1, 1)
        else:
            # Run visual odometry to estimate camera motion
            try:
                Log.info("[GVHMRInference] Running visual odometry for moving camera...")

                # Save frames to temp video (SimpleVO requires video path)
                temp_video = _save_temp_video(images_np)

                # Run SimpleVO
                f_mm = focal_length_mm if focal_length_mm > 0 else 24
                vo = SimpleVO(temp_video, scale=vo_scale, step=vo_step, method="sift", f_mm=f_mm)
                T_w2c_list = vo.compute()

                # Clean up temp file
                os.remove(temp_video)

                # Extract rotation matrices (upper-left 3x3 of each 4x4 transform)
                R_w2c = torch.tensor(np.stack([T[:3, :3] for T in T_w2c_list]), dtype=torch.float32)
                # Extract translation vectors (right column of each 4x4 transform)
                t_w2c = torch.tensor(np.stack([T[:3, 3] for T in T_w2c_list]), dtype=torch.float32)

                Log.info(f"[GVHMRInference] Visual odometry complete: {len(T_w2c_list)} poses estimated")
                Log.info(f"[GVHMRInference] Camera translation range: {t_w2c.abs().max().item():.3f}m")

            except Exception as e:
                Log.warn(f"[GVHMRInference] Visual odometry failed: {e}, falling back to static camera")
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

        # Store camera transforms for potential future use
        camera_data = {
            "R_w2c": R_w2c,
            "t_w2c": t_w2c,
            "K_fullimg": K_fullimg,
        }

        return data, camera_data

    def run_inference(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        model: Dict,
        static_camera: bool = True,
        focal_length_mm: int = 0,
        bbox_scale: float = 1.2,
        vo_scale: float = 0.5,
        vo_step: int = 8,
        intrinsics: torch.Tensor = None,
    ):
        """
        Run GVHMR inference on images with SAM3 masks.
        """
        try:
            Log.info("[GVHMRInference] Starting GVHMR inference...")
            Log.info(f"[GVHMRInference] Input shape: {images.shape}, Masks shape: {masks.shape}")

            # Prepare data
            data, camera_data = self.prepare_data_from_masks(
                images, masks, model, static_camera, focal_length_mm, bbox_scale, vo_scale, vo_step, intrinsics
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
                "K_fullimg": camera_data["K_fullimg"],
                "R_w2c": camera_data["R_w2c"],
                "t_w2c": camera_data["t_w2c"],
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
            # Check if rendering is available
            from hmr4d.utils.vis.renderer import PYTORCH3D_AVAILABLE
            if not PYTORCH3D_AVAILABLE:
                Log.warn("[GVHMRInference] PyTorch3D not available - skipping visualization rendering")
                Log.info("[GVHMRInference] Returning original images (SMPL parameters were extracted successfully)")
                return images

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
            Log.warn(f"[GVHMRInference] Visualization rendering failed: {e}")
            Log.info("[GVHMRInference] Returning original images (SMPL parameters were extracted successfully)")
            # Return original images if rendering fails
            return images


# Node registration
NODE_CLASS_MAPPINGS = {
    "GVHMRInference": GVHMRInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GVHMRInference": "GVHMR Inference",
}
