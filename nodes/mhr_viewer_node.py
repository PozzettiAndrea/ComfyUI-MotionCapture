"""
MHRViewer Node - Visualizes MHR 70-keypoint skeleton in an interactive 3D viewer
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add vendor path
VENDOR_PATH = Path(__file__).parent.parent / "vendor"
sys.path.insert(0, str(VENDOR_PATH))

from hmr4d.utils.pylogger import Log


# MHR skeleton connections with colors (from mhr70.py skeleton_info)
# Format: (from_idx, to_idx, color_rgb)
MHR_SKELETON = [
    # Body - left side (green)
    (13, 11, [0, 255, 0]),     # left_ankle -> left_knee
    (11, 9, [0, 255, 0]),      # left_knee -> left_hip
    (5, 7, [0, 255, 0]),       # left_shoulder -> left_elbow
    (7, 62, [0, 255, 0]),      # left_elbow -> left_wrist
    (5, 9, [51, 153, 255]),    # left_shoulder -> left_hip
    (3, 5, [51, 153, 255]),    # left_ear -> left_shoulder

    # Body - right side (orange)
    (14, 12, [255, 128, 0]),   # right_ankle -> right_knee
    (12, 10, [255, 128, 0]),   # right_knee -> right_hip
    (6, 8, [255, 128, 0]),     # right_shoulder -> right_elbow
    (8, 41, [255, 128, 0]),    # right_elbow -> right_wrist
    (6, 10, [51, 153, 255]),   # right_shoulder -> right_hip
    (4, 6, [51, 153, 255]),    # right_ear -> right_shoulder

    # Body - center (blue)
    (9, 10, [51, 153, 255]),   # left_hip -> right_hip
    (5, 6, [51, 153, 255]),    # left_shoulder -> right_shoulder
    (1, 2, [51, 153, 255]),    # left_eye -> right_eye
    (0, 1, [51, 153, 255]),    # nose -> left_eye
    (0, 2, [51, 153, 255]),    # nose -> right_eye
    (1, 3, [51, 153, 255]),    # left_eye -> left_ear
    (2, 4, [51, 153, 255]),    # right_eye -> right_ear

    # Feet - left (green)
    (13, 15, [0, 255, 0]),     # left_ankle -> left_big_toe
    (13, 16, [0, 255, 0]),     # left_ankle -> left_small_toe
    (13, 17, [0, 255, 0]),     # left_ankle -> left_heel

    # Feet - right (orange)
    (14, 18, [255, 128, 0]),   # right_ankle -> right_big_toe
    (14, 19, [255, 128, 0]),   # right_ankle -> right_small_toe
    (14, 20, [255, 128, 0]),   # right_ankle -> right_heel

    # Left hand - thumb (orange)
    (62, 45, [255, 128, 0]),   # left_wrist -> left_thumb_third_joint
    (45, 44, [255, 128, 0]),   # left_thumb_third_joint -> left_thumb2
    (44, 43, [255, 128, 0]),   # left_thumb2 -> left_thumb3
    (43, 42, [255, 128, 0]),   # left_thumb3 -> left_thumb4 (tip)

    # Left hand - index finger (pink)
    (62, 49, [255, 153, 255]), # left_wrist -> left_forefinger_third_joint
    (49, 48, [255, 153, 255]), # left_forefinger_third_joint -> left_forefinger2
    (48, 47, [255, 153, 255]), # left_forefinger2 -> left_forefinger3
    (47, 46, [255, 153, 255]), # left_forefinger3 -> left_forefinger4 (tip)

    # Left hand - middle finger (light blue)
    (62, 53, [102, 178, 255]), # left_wrist -> left_middle_finger_third_joint
    (53, 52, [102, 178, 255]), # left_middle_finger_third_joint -> left_middle_finger2
    (52, 51, [102, 178, 255]), # left_middle_finger2 -> left_middle_finger3
    (51, 50, [102, 178, 255]), # left_middle_finger3 -> left_middle_finger4 (tip)

    # Left hand - ring finger (red)
    (62, 57, [255, 51, 51]),   # left_wrist -> left_ring_finger_third_joint
    (57, 56, [255, 51, 51]),   # left_ring_finger_third_joint -> left_ring_finger2
    (56, 55, [255, 51, 51]),   # left_ring_finger2 -> left_ring_finger3
    (55, 54, [255, 51, 51]),   # left_ring_finger3 -> left_ring_finger4 (tip)

    # Left hand - pinky finger (green)
    (62, 61, [0, 255, 0]),     # left_wrist -> left_pinky_finger_third_joint
    (61, 60, [0, 255, 0]),     # left_pinky_finger_third_joint -> left_pinky_finger2
    (60, 59, [0, 255, 0]),     # left_pinky_finger2 -> left_pinky_finger3
    (59, 58, [0, 255, 0]),     # left_pinky_finger3 -> left_pinky_finger4 (tip)

    # Right hand - thumb (orange)
    (41, 24, [255, 128, 0]),   # right_wrist -> right_thumb_third_joint
    (24, 23, [255, 128, 0]),   # right_thumb_third_joint -> right_thumb2
    (23, 22, [255, 128, 0]),   # right_thumb2 -> right_thumb3
    (22, 21, [255, 128, 0]),   # right_thumb3 -> right_thumb4 (tip)

    # Right hand - index finger (pink)
    (41, 28, [255, 153, 255]), # right_wrist -> right_forefinger_third_joint
    (28, 27, [255, 153, 255]), # right_forefinger_third_joint -> right_forefinger2
    (27, 26, [255, 153, 255]), # right_forefinger2 -> right_forefinger3
    (26, 25, [255, 153, 255]), # right_forefinger3 -> right_forefinger4 (tip)

    # Right hand - middle finger (light blue)
    (41, 32, [102, 178, 255]), # right_wrist -> right_middle_finger_third_joint
    (32, 31, [102, 178, 255]), # right_middle_finger_third_joint -> right_middle_finger2
    (31, 30, [102, 178, 255]), # right_middle_finger2 -> right_middle_finger3
    (30, 29, [102, 178, 255]), # right_middle_finger3 -> right_middle_finger4 (tip)

    # Right hand - ring finger (red)
    (41, 36, [255, 51, 51]),   # right_wrist -> right_ring_finger_third_joint
    (36, 35, [255, 51, 51]),   # right_ring_finger_third_joint -> right_ring_finger2
    (35, 34, [255, 51, 51]),   # right_ring_finger2 -> right_ring_finger3
    (34, 33, [255, 51, 51]),   # right_ring_finger3 -> right_ring_finger4 (tip)

    # Right hand - pinky finger (green)
    (41, 40, [0, 255, 0]),     # right_wrist -> right_pinky_finger_third_joint
    (40, 39, [0, 255, 0]),     # right_pinky_finger_third_joint -> right_pinky_finger2
    (39, 38, [0, 255, 0]),     # right_pinky_finger2 -> right_pinky_finger3
    (38, 37, [0, 255, 0]),     # right_pinky_finger3 -> right_pinky_finger4 (tip)
]

# MHR keypoint names
MHR_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
    # Right hand (21-41)
    "right_thumb_tip", "right_thumb_first", "right_thumb_second", "right_thumb_third",
    "right_index_tip", "right_index_first", "right_index_second", "right_index_third",
    "right_middle_tip", "right_middle_first", "right_middle_second", "right_middle_third",
    "right_ring_tip", "right_ring_first", "right_ring_second", "right_ring_third",
    "right_pinky_tip", "right_pinky_first", "right_pinky_second", "right_pinky_third",
    "right_wrist",
    # Left hand (42-62)
    "left_thumb_tip", "left_thumb_first", "left_thumb_second", "left_thumb_third",
    "left_index_tip", "left_index_first", "left_index_second", "left_index_third",
    "left_middle_tip", "left_middle_first", "left_middle_second", "left_middle_third",
    "left_ring_tip", "left_ring_first", "left_ring_second", "left_ring_third",
    "left_pinky_tip", "left_pinky_first", "left_pinky_second", "left_pinky_third",
    "left_wrist",
    # Extra (63-69)
    "left_olecranon", "right_olecranon",
    "left_cubital_fossa", "right_cubital_fossa",
    "left_acromion", "right_acromion", "neck",
]


class MHRViewer:
    """
    ComfyUI node for visualizing MHR 70-keypoint skeleton in an interactive 3D viewer.
    Uses Canvas 2D for rendering (no external 3D libraries).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "mhr_params": ("MHR_PARAMS",),
                "npz_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to .npz file with MHR parameters (alternative to mhr_params input)"
                }),
                "frame_skip": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Skip every N frames to reduce data size (1 = no skip)"
                }),
                "joint_size": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 1.0,
                    "tooltip": "Joint circle radius in pixels"
                }),
                "bone_width": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "Bone line width in pixels"
                }),
                "show_hands": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show hand skeleton (can be hidden for cleaner view)"
                }),
            }
        }

    RETURN_TYPES = ("MHR_VIEWER",)
    RETURN_NAMES = ("viewer_data",)
    FUNCTION = "create_viewer_data"
    CATEGORY = "MotionCapture/SAM3D"
    OUTPUT_NODE = True

    def create_viewer_data(
        self,
        mhr_params=None,
        npz_path="",
        frame_skip=1,
        joint_size=5.0,
        bone_width=2.0,
        show_hands=True,
    ):
        """
        Generate skeleton data from MHR parameters for web visualization.
        """
        Log.info("[MHRViewer] Generating skeleton data for visualization...")

        # Handle input sources
        if mhr_params is not None:
            Log.info("[MHRViewer] Using MHR parameters from node input")
            keypoints_3d = mhr_params["keypoints_3d"]  # [F, 70, 3]
        elif npz_path and npz_path.strip():
            Log.info(f"[MHRViewer] Loading MHR parameters from: {npz_path}")
            file_path = Path(npz_path)
            if not file_path.exists():
                raise FileNotFoundError(f"NPZ file not found: {file_path}")

            data = np.load(str(file_path))
            keypoints_3d = torch.from_numpy(data["keypoints_3d"])
            Log.info(f"[MHRViewer] Loaded keypoints with shape {keypoints_3d.shape}")
        else:
            raise ValueError("Either 'mhr_params' or 'npz_path' must be provided")

        # Convert to numpy if tensor
        if isinstance(keypoints_3d, torch.Tensor):
            keypoints_3d = keypoints_3d.cpu().numpy()

        num_frames = keypoints_3d.shape[0]
        Log.info(f"[MHRViewer] Processing {num_frames} frames (skip={frame_skip})")

        # Apply frame skip
        keypoints_subset = keypoints_3d[::frame_skip]  # [F', 70, 3]
        output_frames = keypoints_subset.shape[0]

        # Filter skeleton for hands option
        if show_hands:
            skeleton_data = [
                {"from": int(f), "to": int(t), "color": c}
                for f, t, c in MHR_SKELETON
            ]
        else:
            # Only body skeleton (indices 0-20)
            skeleton_data = [
                {"from": int(f), "to": int(t), "color": c}
                for f, t, c in MHR_SKELETON
                if f < 21 and t < 21
            ]

        Log.info(f"[MHRViewer] Generated {output_frames} frames, "
                 f"{keypoints_subset.shape[1]} keypoints, "
                 f"{len(skeleton_data)} bones")

        # Prepare viewer data
        viewer_data = {
            "frames": output_frames,
            "num_keypoints": 70,
            "keypoints": keypoints_subset.tolist(),  # [F, 70, 3]
            "skeleton": skeleton_data,
            "keypoint_names": MHR_KEYPOINT_NAMES,
            "joint_size": joint_size,
            "bone_width": bone_width,
            "fps": 30 // frame_skip,
        }

        Log.info("[MHRViewer] Viewer data prepared successfully!")

        return {
            "ui": {
                "mhr_skeleton": [viewer_data]  # Matches JavaScript: message.mhr_skeleton
            },
            "result": (viewer_data,)
        }


# Node registration
NODE_CLASS_MAPPINGS = {
    "MHRViewer": MHRViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MHRViewer": "MHR Skeleton Viewer",
}
