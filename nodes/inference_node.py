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
import gc

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

        # Convert images to numpy for processing (granularly to avoid system RAM spike)
        images_np = np.empty((batch_size, height, width, channels), dtype=np.uint8)
        for i in range(batch_size):
            images_np[i] = (images[i] * 255).to(torch.uint8).cpu().numpy()

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
        is_lazy = model_bundle.get("lazy", False)
        low_vram = model_bundle.get("low_vram_mode", True)
        batch_size_proc = model_bundle.get("batch_size", 16)
        
        if is_lazy:
            from hmr4d.utils.preproc import VitPoseExtractor
            vitpose_extractor = VitPoseExtractor(device=device, batch_size=batch_size_proc)
        else:
            vitpose_extractor = model_bundle["vitpose_extractor"]
            if low_vram and device != "cpu":
                vitpose_extractor.pose.to(device)

        # Use get_batch to preprocess images for extractors
        from hmr4d.utils.preproc.vitfeat_extractor import get_batch

        # Prepare images in the right format for get_batch (expects (F, H, W, 3) RGB numpy array)
        imgs_tensor, bbx_xys_processed = get_batch(images_np, bbx_xys, img_ds=1.0, path_type="np")

        # Extract 2D keypoints with ViTPose
        kp2d = vitpose_extractor.extract(imgs_tensor, bbx_xys_processed, img_ds=1.0)
        
        if is_lazy:
            del vitpose_extractor
            gc.collect()
        elif low_vram and device != "cpu":
            vitpose_extractor.pose.to("cpu")
            
        if device != "cpu":
            torch.cuda.empty_cache()

        # Extract ViT features
        Log.info("[GVHMRInference] Extracting image features...")
        if is_lazy:
            from hmr4d.utils.preproc import Extractor
            feature_extractor = Extractor(device=device, batch_size=batch_size_proc)
        else:
            feature_extractor = model_bundle["feature_extractor"]
            if low_vram and device != "cpu":
                feature_extractor.extractor.to(device)

        f_imgseq = feature_extractor.extract_video_features(imgs_tensor, bbx_xys_processed, img_ds=1.0)
        
        if is_lazy:
            del feature_extractor
            gc.collect()
        elif low_vram and device != "cpu":
            feature_extractor.extractor.to("cpu")
            
        if device != "cpu":
            torch.cuda.empty_cache()

        # Cleanup intermediate tensors
        del imgs_tensor
        gc.collect()

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
            is_lazy = model.get("lazy", False)
            low_vram = model.get("low_vram_mode", True)
            device = model["device"]

            if is_lazy:
                from hydra.utils import instantiate
                from omegaconf import OmegaConf
                cfg = model["config"]
                # Instantiate DemoPL with pipeline from config
                model_cfg_dict = OmegaConf.to_container(cfg.model, resolve=True)
                model_cfg = OmegaConf.create(model_cfg_dict)
                gvhmr_model = instantiate(model_cfg, _recursive_=False)
                # Load pretrained weights
                gvhmr_model.load_pretrained_model(model["paths"]["gvhmr"])
                gvhmr_model.eval()
                gvhmr_model.to(device)
            else:
                gvhmr_model = model["gvhmr"]
                if low_vram and device != "cpu":
                    gvhmr_model.to(device)

            with torch.no_grad():
                pred = gvhmr_model.predict(data, static_cam=static_camera)

            if is_lazy:
                del gvhmr_model
                gc.collect()
            elif low_vram and device != "cpu":
                gvhmr_model.to("cpu")
                
            if device != "cpu":
                torch.cuda.empty_cache()

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
            smplx2smpl_path = Path(__file__).parent.parent / "vendor" / "hmr4d" / "utils" / "body_model" / "smplx2smpl_sparse.pt"
            smplx2smpl = torch.load(str(smplx2smpl_path), map_location=device).to(device)

            # Get SMPL vertices
            smpl_incam = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in smpl_params["incam"].items()}
            smplx_out = smplx(**smpl_incam)
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
