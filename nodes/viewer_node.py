"""
SMPLViewer Node - Visualizes SMPL motion capture data in an interactive 3D viewer.
Writes mesh data to a binary file and sends the filename to the JS viewer via IPC.
"""

import logging
from pathlib import Path
import torch
import numpy as np
import smplx
import folder_paths

logger = logging.getLogger("SMPLViewer")


def _next_sequential_filename(directory, prefix, ext):
    """Find the next sequential filename like prefix_0001.ext, prefix_0002.ext, etc."""
    existing = sorted(directory.glob(f"{prefix}_*{ext}"))
    max_num = 0
    for f in existing:
        stem = f.stem
        suffix = stem[len(prefix) + 1:]
        try:
            max_num = max(max_num, int(suffix))
        except ValueError:
            pass
    return f"{prefix}_{max_num + 1:04d}{ext}"


class SMPLViewer:
    """
    ComfyUI node for visualizing SMPL motion capture sequences in an interactive 3D viewer.
    Writes mesh to a .bin file; the JS widget fetches it via /view?filename=...&type=output.
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

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mesh_file",)
    FUNCTION = "create_viewer_data"
    CATEGORY = "MotionCapture/GVHMR"
    OUTPUT_NODE = True

    def create_viewer_data(self, npz_path="", frame_skip=1, mesh_color="#4a9eff"):
        """
        Generate 3D mesh data from SMPL parameters, write to .bin file,
        and return the filename for the JS viewer to fetch.
        """
        logger.info("[SMPLViewer] Generating 3D mesh data for visualization...")

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

        # Extract SMPL parameters
        body_pose = params['body_pose']  # (F, 63)
        betas = params['betas']  # (F, 10)
        global_orient = params['global_orient']  # (F, 3)
        transl = params.get('transl', None)  # (F, 3) or None

        num_frames = body_pose.shape[0]
        logger.info(f"[SMPLViewer] Processing {num_frames} frames (skip={frame_skip})")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        data_dir = Path(__file__).parent / "data"
        models_dir = Path(folder_paths.models_dir) / "motion_capture" / "body_models"

        # Initialize SMPL-X model
        smplx_model = smplx.create(
            model_path=str(models_dir),
            model_type='smplx',
            gender='neutral',
            num_pca_comps=12,
            flat_hand_mean=False,
        ).to(device)
        smplx_model.eval()

        # Load SMPL-X to SMPL vertex conversion matrix
        smplx2smpl = torch.load(
            str(data_dir / "smplx2smpl_sparse.pt"), weights_only=True
        ).to(device)

        # Get SMPL faces
        faces = np.load(str(data_dir / "smpl_faces.npy"))

        # Generate vertices
        vertices_list = []
        with torch.no_grad():
            for frame_idx in range(0, num_frames, frame_skip):
                bp = body_pose[frame_idx:frame_idx+1].to(device)
                b = betas[frame_idx:frame_idx+1].to(device)
                go = global_orient[frame_idx:frame_idx+1].to(device)
                t = transl[frame_idx:frame_idx+1].to(device) if transl is not None else None

                smplx_out = smplx_model(
                    body_pose=bp, betas=b, global_orient=go, transl=t,
                )
                smpl_verts = torch.matmul(smplx2smpl, smplx_out.vertices[0])
                vertices_list.append(smpl_verts.cpu().numpy())

        vertices_array = np.stack(vertices_list, axis=0).astype(np.float32)  # (F', V, 3)
        faces_u32 = faces.astype(np.uint32)
        fps = 30 // frame_skip

        logger.info(f"[SMPLViewer] Generated mesh: {vertices_array.shape[0]} frames, "
                     f"{vertices_array.shape[1]} vertices, {faces_u32.shape[0]} faces")

        # Write binary file to ComfyUI output directory
        # Format: Magic(4) | Frames(u32) | Verts(u32) | Faces(u32) | FPS(f32) | mesh_color(64 bytes) | vertex_data | face_data
        output_dir = Path(folder_paths.get_output_directory())
        mesh_filename = _next_sequential_filename(output_dir, "smpl_mesh", ".bin")
        mesh_filepath = output_dir / mesh_filename

        with open(mesh_filepath, "wb") as f:
            f.write(b"SMPL")
            f.write(np.array([vertices_array.shape[0]], dtype=np.uint32).tobytes())
            f.write(np.array([vertices_array.shape[1]], dtype=np.uint32).tobytes())
            f.write(np.array([faces_u32.shape[0]], dtype=np.uint32).tobytes())
            f.write(np.array([float(fps)], dtype=np.float32).tobytes())
            # mesh_color as fixed 64-byte UTF-8 field (padded with nulls)
            color_bytes = mesh_color.encode("utf-8")[:64]
            f.write(color_bytes + b"\x00" * (64 - len(color_bytes)))
            f.write(vertices_array.tobytes())
            f.write(faces_u32.tobytes())

        size_mb = mesh_filepath.stat().st_size / (1024 * 1024)
        logger.info(f"[SMPLViewer] Wrote {mesh_filename} ({size_mb:.1f} MB)")

        return {
            "ui": {
                "smpl_mesh_file": [mesh_filename]
            },
            "result": (mesh_filename,)
        }


# Node registration
NODE_CLASS_MAPPINGS = {
    "SMPLViewer": SMPLViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SMPLViewer": "SMPL 3D Viewer",
}
