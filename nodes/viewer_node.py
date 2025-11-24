"""
SMPLViewer Node - Visualizes SMPL motion capture data in an interactive 3D viewer
"""

import os
import sys
import json
import base64
from pathlib import Path
import torch
import numpy as np

# Add vendor path for GVHMR
VENDOR_PATH = Path(__file__).parent.parent / "vendor"
sys.path.insert(0, str(VENDOR_PATH))

from hmr4d.utils.body_model.smplx_lite import SmplxLite
from hmr4d.utils.pylogger import Log


class SMPLViewer:
    """
    ComfyUI node for visualizing SMPL motion capture sequences in an interactive 3D viewer.
    Uses Three.js for real-time playback and camera controls.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "smpl_params": ("SMPL_PARAMS",),
                "npz_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to .npz file with SMPL parameters (alternative to smpl_params input)"
                }),
                "frame_skip": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Skip every N frames to reduce data size (1 = no skip)"
                }),
                "mesh_color": ("STRING", {
                    "default": "#4a9eff",
                    "tooltip": "Hex color for the mesh (e.g. #4a9eff for blue)"
                }),
            }
        }

    RETURN_TYPES = ("SMPL_VIEWER",)
    RETURN_NAMES = ("viewer_data",)
    FUNCTION = "create_viewer_data"
    CATEGORY = "MotionCapture/GVHMR"
    OUTPUT_NODE = True

    def create_viewer_data(self, smpl_params=None, npz_path="", frame_skip=1, mesh_color="#4a9eff"):
        """
        Generate 3D mesh data from SMPL parameters for web visualization.

        Args:
            smpl_params: Dictionary with 'global' key containing SMPL parameters (optional)
            npz_path: Path to .npz file with SMPL parameters (optional)
            frame_skip: Skip every N frames to reduce data size
            mesh_color: Hex color for the mesh

        Returns:
            Dictionary containing mesh geometry and metadata for Three.js viewer
        """
        Log.info("[SMPLViewer] Generating 3D mesh data for visualization...")

        # Handle input sources - priority: smpl_params > npz_path
        if smpl_params is not None:
            # Use provided SMPL_PARAMS from node connection
            Log.info("[SMPLViewer] Using SMPL parameters from node input")
            params = smpl_params['global']
        elif npz_path and npz_path.strip():
            # Load from npz file
            Log.info(f"[SMPLViewer] Loading SMPL parameters from: {npz_path}")
            file_path = Path(npz_path)
            if not file_path.exists():
                raise FileNotFoundError(f"NPZ file not found: {file_path}")

            # Load npz file
            data = np.load(str(file_path))
            params = {}
            for key in data.files:
                params[key] = torch.from_numpy(data[key])

            Log.info(f"[SMPLViewer] Loaded {len(data.files)} parameter arrays from npz")
        else:
            raise ValueError("Either 'smpl_params' or 'npz_path' must be provided")

        # Extract SMPL parameters (now from either source)
        # params now contains the 'global' parameters
        body_pose = params['body_pose']  # (F, 63)
        betas = params['betas']  # (F, 10)
        global_orient = params['global_orient']  # (F, 3)
        transl = params.get('transl', None)  # (F, 3) or None

        num_frames = body_pose.shape[0]
        Log.info(f"[SMPLViewer] Processing {num_frames} frames (skip={frame_skip})")

        # Initialize SMPL model
        smpl_model = SmplxLite(gender="neutral", num_betas=10)
        smpl_model.eval()

        # Move to same device as parameters
        device = body_pose.device
        smpl_model = smpl_model.to(device)

        # Generate mesh for each frame
        vertices_list = []
        with torch.no_grad():
            for frame_idx in range(0, num_frames, frame_skip):
                # Get parameters for this frame
                bp = body_pose[frame_idx:frame_idx+1]  # (1, 63)
                b = betas[frame_idx:frame_idx+1]  # (1, 10)
                go = global_orient[frame_idx:frame_idx+1]  # (1, 3)
                t = transl[frame_idx:frame_idx+1] if transl is not None else None  # (1, 3)

                # Generate vertices for this frame
                verts = smpl_model.forward(
                    body_pose=bp,
                    betas=b,
                    global_orient=go,
                    transl=t,
                    rotation_type="aa"
                )  # (1, V, 3)

                vertices_list.append(verts[0].cpu().numpy())  # (V, 3)

        vertices_array = np.stack(vertices_list, axis=0)  # (F', V, 3)

        # Get faces (same for all frames)
        faces = smpl_model.faces.astype(np.int32)  # (F, 3)

        Log.info(f"[SMPLViewer] Generated mesh: {vertices_array.shape[0]} frames, "
                 f"{vertices_array.shape[1]} vertices, {faces.shape[0]} faces")

        # Prepare data for JavaScript viewer
        viewer_data = {
            "frames": vertices_array.shape[0],
            "num_vertices": vertices_array.shape[1],
            "num_faces": faces.shape[0],
            "vertices": vertices_array.tolist(),  # (F, V, 3)
            "faces": faces.tolist(),  # (F, 3)
            "mesh_color": mesh_color,
            "fps": 30 // frame_skip,  # Adjust FPS based on frame skip
        }

        Log.info("[SMPLViewer] Viewer data prepared successfully!")

        # Return in SAM3 pattern: ui dict for frontend, result tuple for outputs
        return {
            "ui": {
                "smpl_mesh": [viewer_data]  # Matches JavaScript: message.smpl_mesh
            },
            "result": (viewer_data,)
        }


# Node registration
NODE_CLASS_MAPPINGS = {
    "SMPLViewer": SMPLViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SMPLViewer": "SMPL 3D Viewer",
}
