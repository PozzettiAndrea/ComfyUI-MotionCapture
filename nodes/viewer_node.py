"""
SMPLViewer Node - Visualizes SMPL motion capture data in an interactive 3D viewer
Runs in main ComfyUI process (not isolated) to avoid IPC size limits.
"""

import logging
from pathlib import Path
import torch
import numpy as np
import smplx

logger = logging.getLogger("SMPLViewer")


class SMPLViewer:
    """
    ComfyUI node for visualizing SMPL motion capture sequences in an interactive 3D viewer.
    Uses Three.js for real-time playback and camera controls.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "npz_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to .npz file with SMPL parameters (from GVHMR Inference)"
                }),
            },
            "optional": {
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

    def create_viewer_data(self, npz_path="", frame_skip=1, mesh_color="#4a9eff"):
        """
        Generate 3D mesh data from SMPL parameters for web visualization.
        """
        logger.info("[SMPLViewer] Generating 3D mesh data for visualization...")

        # Load from npz file
        if not npz_path or not npz_path.strip():
            raise ValueError("npz_path is required")

        logger.info(f"[SMPLViewer] Loading SMPL parameters from: {npz_path}")
        file_path = Path(npz_path)
        if not file_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {file_path}")

        # Load npz file
        data = np.load(str(file_path))
        params = {}
        for key in data.files:
            params[key] = torch.from_numpy(data[key])

        logger.info(f"[SMPLViewer] Loaded {len(data.files)} parameter arrays from npz")

        # Extract SMPL parameters
        body_pose = params['body_pose']  # (F, 63)
        betas = params['betas']  # (F, 10)
        global_orient = params['global_orient']  # (F, 3)
        transl = params.get('transl', None)  # (F, 3) or None

        # Debug: log actual shapes
        logger.info(f"[SMPLViewer] body_pose shape: {body_pose.shape}")
        logger.info(f"[SMPLViewer] betas shape: {betas.shape}")
        logger.info(f"[SMPLViewer] global_orient shape: {global_orient.shape}")
        if transl is not None:
            logger.info(f"[SMPLViewer] transl shape: {transl.shape}")

        num_frames = body_pose.shape[0]
        logger.info(f"[SMPLViewer] Processing {num_frames} frames (skip={frame_skip})")

        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Get paths to data files
        data_dir = Path(__file__).parent / "data"

        # Path to body models (in ComfyUI models directory)
        models_dir = Path(__file__).parent.parent.parent.parent / "models" / "motion_capture" / "body_models"

        # Initialize SMPL-X model
        logger.info("[SMPLViewer] Loading SMPLX model...")
        smplx_model = smplx.create(
            model_path=str(models_dir),
            model_type='smplx',
            gender='neutral',
            num_pca_comps=12,
            flat_hand_mean=False,
        ).to(device)
        smplx_model.eval()

        # Load SMPL-X to SMPL vertex conversion matrix
        smplx2smpl_path = data_dir / "smplx2smpl_sparse.pt"
        smplx2smpl = torch.load(str(smplx2smpl_path), weights_only=True).to(device)

        # Get SMPL faces from pre-saved file
        smpl_faces_path = data_dir / "smpl_faces.npy"
        faces = np.load(str(smpl_faces_path))

        # Process frames in batches
        vertices_list = []
        with torch.no_grad():
            for frame_idx in range(0, num_frames, frame_skip):
                # Get parameters for this frame
                bp = body_pose[frame_idx:frame_idx+1].to(device)  # (1, 63)
                b = betas[frame_idx:frame_idx+1].to(device)  # (1, 10)
                go = global_orient[frame_idx:frame_idx+1].to(device)  # (1, 3)
                t = transl[frame_idx:frame_idx+1].to(device) if transl is not None else None

                # Generate SMPL-X vertices
                smplx_out = smplx_model(
                    body_pose=bp,
                    betas=b,
                    global_orient=go,
                    transl=t,
                )

                # Convert SMPL-X vertices to SMPL vertices
                smpl_verts = torch.matmul(smplx2smpl, smplx_out.vertices[0])  # (V_smpl, 3)
                vertices_list.append(smpl_verts.cpu().numpy())

        vertices_array = np.stack(vertices_list, axis=0)  # (F', V, 3)

        logger.info(f"[SMPLViewer] Generated mesh: {vertices_array.shape[0]} frames, "
                 f"{vertices_array.shape[1]} vertices, {faces.shape[0]} faces")

        # Prepare data for JavaScript viewer
        viewer_data = {
            "frames": vertices_array.shape[0],
            "num_vertices": vertices_array.shape[1],
            "num_faces": faces.shape[0],
            "vertices": vertices_array.tolist(),
            "faces": faces.tolist(),
            "mesh_color": mesh_color,
            "fps": 30 // frame_skip,
        }

        logger.info("[SMPLViewer] Viewer data prepared successfully!")

        # Return in SAM3 pattern: ui dict for frontend, result tuple for outputs
        return {
            "ui": {
                "smpl_mesh": [viewer_data]
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
