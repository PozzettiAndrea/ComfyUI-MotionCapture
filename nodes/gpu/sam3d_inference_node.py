"""
SAM3DVideoInference Node - Performs SAM 3D Body inference on video with temporal smoothing

Outputs MHR 70-keypoint skeleton (vs SMPL's 24 joints) with full hand detail,
and applies GVHMR-style temporal smoothing for jitter-free motion.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from tqdm import tqdm

# Add vendor path
VENDOR_PATH = Path(__file__).parent / "vendor"
sys.path.insert(0, str(VENDOR_PATH))

# Import GVHMR utilities
from hmr4d.utils.pylogger import Log
from hmr4d.utils.net_utils import gaussian_smooth, moving_average_smooth

# Import local utilities
from .utils import (
    extract_bboxes_from_masks,
    bbox_to_xyxy,
    expand_bbox,
    validate_masks,
    validate_images,
)


class SAM3DVideoInference:
    """
    ComfyUI node for SAM 3D Body video inference with temporal smoothing.

    Processes video frames through SAM 3D Body for MHR skeleton extraction,
    then applies temporal smoothing to reduce jitter.

    Output: MHR_PARAMS with 70 keypoints including full hand articulation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # ComfyUI IMAGE tensor (B, H, W, C)
                "masks": ("MASK",),    # SAM3 masks (B, H, W)
                "model": ("SAM3D_MODEL",),  # Model bundle from LoadSAM3DBodyModels
            },
            "optional": {
                "inference_type": (["full", "body"], {
                    "default": "full",
                    "tooltip": "full: body + hands (70 keypoints), body: body only (faster)"
                }),
                "temporal_smoothing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply temporal smoothing to reduce jitter"
                }),
                "smoothing_sigma": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "Gaussian smoothing kernel width (higher = smoother)"
                }),
                "smoothing_method": (["gaussian", "moving_average"], {
                    "default": "gaussian",
                    "tooltip": "Smoothing algorithm"
                }),
                "bbox_scale": ("FLOAT", {
                    "default": 1.2,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Expand bounding box by this factor"
                }),
            }
        }

    RETURN_TYPES = ("MHR_PARAMS", "IMAGE", "STRING")
    RETURN_NAMES = ("mhr_params", "visualization", "info")
    FUNCTION = "run_inference"
    CATEGORY = "MotionCapture/SAM3D"

    def smooth_mhr_sequence(
        self,
        joint_positions: torch.Tensor,
        joint_rotations: Optional[torch.Tensor],
        sigma: float = 3.0,
        method: str = "gaussian",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply temporal smoothing to MHR skeleton sequence.

        Args:
            joint_positions: [F, N, 3] - N keypoints per frame
            joint_rotations: [F, M, 3, 3] - M rotation matrices per frame (optional)
            sigma: smoothing kernel width
            method: "gaussian" or "moving_average"

        Returns:
            Smoothed positions and rotations
        """
        # Smooth positions - shape [F, N, 3], smooth along dim 0 (frames)
        if method == "gaussian":
            smoothed_pos = gaussian_smooth(joint_positions, sigma=sigma, dim=0)
        else:
            window = int(sigma * 2 + 1)
            smoothed_pos = moving_average_smooth(joint_positions, window_size=window, dim=0)

        # Smooth rotations if provided
        smoothed_rot = None
        if joint_rotations is not None:
            # Convert rotation matrices to 6D representation for smoother interpolation
            # 6D representation: first two columns of rotation matrix
            F, M, _, _ = joint_rotations.shape
            rot_6d = joint_rotations[:, :, :, :2].reshape(F, M, 6)  # [F, M, 6]

            if method == "gaussian":
                smoothed_6d = gaussian_smooth(rot_6d, sigma=sigma, dim=0)
            else:
                window = int(sigma * 2 + 1)
                smoothed_6d = moving_average_smooth(smoothed_6d, window_size=window, dim=0)

            # Convert back to rotation matrices using Gram-Schmidt
            smoothed_rot = self._rotation_6d_to_matrix(smoothed_6d)

        return smoothed_pos, smoothed_rot

    def _rotation_6d_to_matrix(self, rot_6d: torch.Tensor) -> torch.Tensor:
        """
        Convert 6D rotation representation to rotation matrix.
        Uses Gram-Schmidt orthogonalization.

        Args:
            rot_6d: [..., 6] tensor

        Returns:
            [..., 3, 3] rotation matrix
        """
        shape = rot_6d.shape[:-1]
        rot_6d = rot_6d.reshape(-1, 6)

        a1 = rot_6d[:, :3]
        a2 = rot_6d[:, 3:]

        # Gram-Schmidt
        b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
        b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
        b3 = torch.cross(b1, b2, dim=-1)

        rot_mat = torch.stack([b1, b2, b3], dim=-1)  # [..., 3, 3]
        return rot_mat.reshape(*shape, 3, 3)

    def run_inference(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        model: Dict,
        inference_type: str = "full",
        temporal_smoothing: bool = True,
        smoothing_sigma: float = 3.0,
        smoothing_method: str = "gaussian",
        bbox_scale: float = 1.2,
    ):
        """
        Run SAM 3D Body inference on video frames with temporal smoothing.
        """
        try:
            Log.info("[SAM3DVideoInference] Starting SAM 3D Body inference...")
            Log.info(f"[SAM3DVideoInference] Input: {images.shape[0]} frames, {images.shape[1]}x{images.shape[2]}")

            # Validate inputs
            validate_images(images)
            validate_masks(masks)

            device = model["device"]
            batch_size, height, width, channels = images.shape

            # Convert images to numpy (RGB, 0-255)
            images_np = (images.cpu().numpy() * 255).astype(np.uint8)
            masks_np = masks.cpu().numpy()

            # Extract bounding boxes from masks
            Log.info("[SAM3DVideoInference] Extracting bounding boxes from masks...")
            bboxes_xywh = extract_bboxes_from_masks(masks)

            # Expand bounding boxes
            bboxes_xywh = [
                expand_bbox(bbox, scale=bbox_scale, img_width=width, img_height=height)
                for bbox in bboxes_xywh
            ]

            # Convert to xyxy format for SAM3D
            bboxes_xyxy = np.array([bbox_to_xyxy(bbox) for bbox in bboxes_xywh])

            # Create SAM3D estimator
            from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator

            estimator = SAM3DBodyEstimator(
                sam_3d_body_model=model["model"],
                model_cfg=model["model_cfg"],
                human_detector=None,  # Use provided masks/bboxes
                human_segmentor=None,
            )

            # Process each frame
            Log.info(f"[SAM3DVideoInference] Processing {batch_size} frames...")
            all_outputs = []

            for frame_idx in tqdm(range(batch_size), desc="SAM3D Inference"):
                frame = images_np[frame_idx]  # RGB, 0-255
                mask = masks_np[frame_idx]    # Single mask for this frame
                bbox = bboxes_xyxy[frame_idx:frame_idx+1]  # [1, 4]

                # Run SAM3D inference on single frame
                outputs = estimator.process_one_image(
                    img=frame,
                    bboxes=bbox,
                    masks=mask[np.newaxis, :, :] if mask.ndim == 2 else mask,
                    use_mask=True,
                    inference_type=inference_type,
                )

                if len(outputs) > 0:
                    all_outputs.append(outputs[0])
                else:
                    # No detection - use previous frame or zeros
                    Log.warn(f"[SAM3DVideoInference] No detection in frame {frame_idx}")
                    if all_outputs:
                        all_outputs.append(all_outputs[-1])  # Copy previous
                    else:
                        all_outputs.append(None)

            # Stack results into temporal sequence
            Log.info("[SAM3DVideoInference] Stacking results...")
            mhr_params = self._stack_outputs(all_outputs, device)

            # Apply temporal smoothing
            if temporal_smoothing and batch_size > 1:
                Log.info(f"[SAM3DVideoInference] Applying {smoothing_method} smoothing (sigma={smoothing_sigma})...")
                mhr_params = self._apply_smoothing(mhr_params, smoothing_sigma, smoothing_method)

            # Create visualization
            Log.info("[SAM3DVideoInference] Rendering visualization...")
            viz_frames = self._render_visualization(images, mhr_params, model)

            # Create info string
            info = (
                f"SAM 3D Body Inference Complete\n"
                f"Frames processed: {batch_size}\n"
                f"Inference type: {inference_type}\n"
                f"Keypoints: 70 (MHR format)\n"
                f"Temporal smoothing: {temporal_smoothing}\n"
                f"Device: {device}\n"
            )

            Log.info("[SAM3DVideoInference] Inference complete!")
            return (mhr_params, viz_frames, info)

        except Exception as e:
            error_msg = f"SAM3D Inference failed: {str(e)}"
            Log.error(error_msg)
            import traceback
            traceback.print_exc()
            return (None, images, error_msg)

    def _stack_outputs(self, outputs: list, device: str) -> Dict:
        """Stack per-frame outputs into temporal sequence."""
        # Filter out None values
        valid_outputs = [o for o in outputs if o is not None]
        if not valid_outputs:
            raise ValueError("No valid outputs from SAM3D inference")

        num_frames = len(outputs)

        # Initialize with first valid output shape
        first = valid_outputs[0]

        # Stack arrays
        keypoints_3d = []
        keypoints_3d_cam = []
        keypoints_2d = []
        vertices = []
        cam_t = []
        global_rots = []
        joint_coords = []
        body_pose = []
        hand_pose = []
        shape_params = []
        scale_params = []

        for i, out in enumerate(outputs):
            if out is None:
                # Use previous valid frame
                out = valid_outputs[min(i, len(valid_outputs)-1)]

            keypoints_3d.append(out["pred_keypoints_3d"])
            keypoints_3d_cam.append(out.get("pred_keypoints_3d_cam", out["pred_keypoints_3d"] + out["pred_cam_t"]))
            keypoints_2d.append(out["pred_keypoints_2d"])
            vertices.append(out["pred_vertices"])
            cam_t.append(out["pred_cam_t"])
            global_rots.append(out.get("pred_global_rots", np.eye(3)))
            joint_coords.append(out.get("pred_joint_coords", out["pred_keypoints_3d"]))
            body_pose.append(out.get("body_pose_params", np.zeros(260)))
            hand_pose.append(out.get("hand_pose_params", np.zeros(108)))
            shape_params.append(out.get("shape_params", np.zeros(45)))
            scale_params.append(out.get("scale_params", np.zeros(28)))

        mhr_params = {
            "type": "mhr",
            "num_frames": num_frames,
            "keypoints_3d": torch.from_numpy(np.stack(keypoints_3d)).float(),  # [F, 70, 3]
            "keypoints_3d_cam": torch.from_numpy(np.stack(keypoints_3d_cam)).float(),  # [F, 70, 3] camera-space
            "keypoints_2d": torch.from_numpy(np.stack(keypoints_2d)).float(),  # [F, 70, 2]
            "vertices": torch.from_numpy(np.stack(vertices)).float(),          # [F, V, 3]
            "cam_t": torch.from_numpy(np.stack(cam_t)).float(),                # [F, 3]
            "global_rots": torch.from_numpy(np.stack(global_rots)).float() if global_rots[0] is not None else None,
            "joint_coords": torch.from_numpy(np.stack(joint_coords)).float(),  # [F, J, 3]
            "body_pose": torch.from_numpy(np.stack(body_pose)).float(),        # [F, 260]
            "hand_pose": torch.from_numpy(np.stack(hand_pose)).float(),        # [F, 108]
            "shape_params": torch.from_numpy(np.stack(shape_params)).float(),  # [F, 45]
            "scale_params": torch.from_numpy(np.stack(scale_params)).float(),  # [F, 28]
            "focal_length": outputs[0]["focal_length"] if outputs[0] else None,
        }

        return mhr_params

    def _apply_smoothing(self, mhr_params: Dict, sigma: float, method: str) -> Dict:
        """Apply temporal smoothing to MHR parameters."""
        # Smooth keypoints (main output)
        smoothed_kp3d, _ = self.smooth_mhr_sequence(
            mhr_params["keypoints_3d"],
            None,
            sigma=sigma,
            method=method,
        )
        mhr_params["keypoints_3d"] = smoothed_kp3d

        # Smooth camera-space keypoints
        if mhr_params.get("keypoints_3d_cam") is not None:
            smoothed_kp3d_cam, _ = self.smooth_mhr_sequence(
                mhr_params["keypoints_3d_cam"],
                None,
                sigma=sigma,
                method=method,
            )
            mhr_params["keypoints_3d_cam"] = smoothed_kp3d_cam

        # Smooth joint coords
        if mhr_params.get("joint_coords") is not None:
            smoothed_joints, _ = self.smooth_mhr_sequence(
                mhr_params["joint_coords"],
                None,
                sigma=sigma,
                method=method,
            )
            mhr_params["joint_coords"] = smoothed_joints

        # Smooth vertices
        if mhr_params.get("vertices") is not None:
            smoothed_verts, _ = self.smooth_mhr_sequence(
                mhr_params["vertices"],
                None,
                sigma=sigma,
                method=method,
            )
            mhr_params["vertices"] = smoothed_verts

        # Smooth camera translation
        if mhr_params.get("cam_t") is not None:
            if method == "gaussian":
                mhr_params["cam_t"] = gaussian_smooth(mhr_params["cam_t"], sigma=sigma, dim=0)
            else:
                window = int(sigma * 2 + 1)
                mhr_params["cam_t"] = moving_average_smooth(mhr_params["cam_t"], window_size=window, dim=0)

        return mhr_params

    def _render_visualization(
        self,
        images: torch.Tensor,
        mhr_params: Dict,
        model: Dict,
    ) -> torch.Tensor:
        """Render skeleton overlay on input images."""
        try:
            batch_size = images.shape[0]
            images_np = (images.cpu().numpy() * 255).astype(np.uint8)

            # Get 2D keypoints for visualization
            keypoints_2d = mhr_params["keypoints_2d"].cpu().numpy()  # [F, 70, 2]

            rendered_frames = []

            # MHR skeleton connections for visualization (body only for clarity)
            skeleton_connections = [
                # Body
                (5, 7), (7, 62),   # Left arm: shoulder -> elbow -> wrist
                (6, 8), (8, 41),   # Right arm: shoulder -> elbow -> wrist
                (5, 6),            # Shoulders
                (5, 9), (6, 10),   # Torso
                (9, 10),           # Hips
                (9, 11), (11, 13), # Left leg
                (10, 12), (12, 14), # Right leg
                (0, 69),           # Nose to neck
                # Feet
                (13, 15), (13, 17), # Left ankle to toes/heel
                (14, 18), (14, 20), # Right ankle to toes/heel
            ]

            for i in range(batch_size):
                img = images_np[i].copy()
                kp = keypoints_2d[i]  # [70, 2]

                # Draw skeleton
                for start_idx, end_idx in skeleton_connections:
                    if start_idx < len(kp) and end_idx < len(kp):
                        pt1 = tuple(kp[start_idx].astype(int))
                        pt2 = tuple(kp[end_idx].astype(int))
                        cv2.line(img, pt1, pt2, (0, 255, 0), 2)

                # Draw keypoints
                for j, pt in enumerate(kp[:21]):  # Body keypoints
                    pt_int = tuple(pt.astype(int))
                    cv2.circle(img, pt_int, 3, (255, 0, 0), -1)

                # Draw hand keypoints (smaller)
                for j, pt in enumerate(kp[21:]):  # Hand keypoints
                    pt_int = tuple(pt.astype(int))
                    cv2.circle(img, pt_int, 2, (0, 255, 255), -1)

                rendered_frames.append(img)

            # Convert back to torch tensor
            rendered_tensor = torch.from_numpy(np.stack(rendered_frames)).float() / 255.0

            return rendered_tensor

        except Exception as e:
            Log.warn(f"[SAM3DVideoInference] Visualization rendering failed: {e}")
            return images


# Node registration
NODE_CLASS_MAPPINGS = {
    "SAM3DVideoInference": SAM3DVideoInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DVideoInference": "SAM3D Video Inference",
}
