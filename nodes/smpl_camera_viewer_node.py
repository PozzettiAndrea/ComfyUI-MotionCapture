"""
SMPLCameraViewer Node - Visualizes SMPL mesh from the estimated camera trajectory.
Writes mesh + camera data to a binary file (SMPC format) for the JS viewer.
"""

import logging
from pathlib import Path
import torch
import numpy as np
import cv2
import smplx
import folder_paths

logger = logging.getLogger("SMPLCameraViewer")


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


class SMPLCameraViewer:
    """
    ComfyUI node for visualizing SMPL mesh from the estimated camera perspective.
    Supports both trajectory camera (from GVHMR moving camera) and free orbit mode.
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
                "camera_npz_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to camera trajectory .npz (from GVHMR moving camera output). If empty, checks main NPZ for camera data."
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
                "images": ("IMAGE", {
                    "tooltip": "Reference video frames to display side-by-side with the 3D mesh"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mesh_file",)
    FUNCTION = "create_viewer_data"
    CATEGORY = "MotionCapture/GVHMR"
    OUTPUT_NODE = True

    def create_viewer_data(self, npz_path="", camera_npz_path="", frame_skip=1, mesh_color="#4a9eff", images=None):
        logger.info("[SMPLCameraViewer] Generating 3D mesh + camera data...")

        if not npz_path or not npz_path.strip():
            raise ValueError("npz_path is required")

        file_path = Path(npz_path)
        if not file_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {file_path}")

        # Load SMPL params
        data = np.load(str(file_path))
        params = {key: torch.from_numpy(data[key]) for key in data.files}

        body_pose = params['body_pose']
        betas = params['betas']
        global_orient = params['global_orient']
        transl = params.get('transl', None)

        # Load incam params for through-camera rendering
        has_incam = 'global_orient_incam' in params and 'transl_incam' in params
        if has_incam:
            global_orient_incam = params['global_orient_incam']
            transl_incam = params['transl_incam']
            logger.info("[SMPLCameraViewer] Found incam params for through-camera rendering")

        num_frames = body_pose.shape[0]
        logger.info(f"[SMPLCameraViewer] Processing {num_frames} frames (skip={frame_skip})")

        # Load camera data: try separate NPZ first, then check main NPZ
        has_camera = False
        R_cam2world_np = None
        t_cam2world_np = None
        K_fullimg_np = None
        img_width = 1920
        img_height = 1080

        # Helper to extract camera data from an npz dict, supporting both new and old formats
        def _extract_camera(src, is_tensor=False):
            """Extract R_cam2world, t_cam2world, K_fullimg from npz data.
            Supports new format (R_cam2world) and old format (R_w2c)."""
            def _get(key):
                v = src[key]
                if is_tensor:
                    v = v.numpy()
                return v.astype(np.float32)

            if 'R_cam2world' in src and 't_cam2world' in src and 'K_fullimg' in src:
                return _get('R_cam2world'), _get('t_cam2world'), _get('K_fullimg'), 'new'
            elif 'R_w2c' in src and 't_w2c' in src and 'K_fullimg' in src:
                # Old format: convert world-to-cam → cam-to-world by inversion
                R_w2c = _get('R_w2c')
                t_w2c = _get('t_w2c')
                R_c2w = R_w2c.transpose(0, 2, 1)  # (F, 3, 3)
                t_c2w = -np.einsum('fij,fj->fi', R_c2w, t_w2c)  # (F, 3)
                logger.warning("[SMPLCameraViewer] Using old R_w2c/t_w2c format — re-run GVHMR inference for best results")
                return R_c2w, t_c2w, _get('K_fullimg'), 'old'
            return None, None, None, None

        if camera_npz_path and camera_npz_path.strip():
            cam_path = Path(camera_npz_path)
            if cam_path.exists():
                cam_data = np.load(str(cam_path))
                R, t, K, fmt = _extract_camera(cam_data)
                if R is not None:
                    R_cam2world_np, t_cam2world_np, K_fullimg_np = R, t, K
                    if 'img_width' in cam_data:
                        img_width = int(cam_data['img_width'][0])
                    if 'img_height' in cam_data:
                        img_height = int(cam_data['img_height'][0])
                    has_camera = True
                    logger.info(f"[SMPLCameraViewer] Loaded camera trajectory ({fmt} format) from: {cam_path}")

        if not has_camera:
            R, t, K, fmt = _extract_camera(params, is_tensor=True)
            if R is not None:
                R_cam2world_np, t_cam2world_np, K_fullimg_np = R, t, K
                if 'img_width' in params:
                    img_width = int(params['img_width'].item())
                if 'img_height' in params:
                    img_height = int(params['img_height'].item())
                has_camera = True
                logger.info(f"[SMPLCameraViewer] Loaded camera trajectory ({fmt} format) from main NPZ")

        if not has_camera:
            logger.info("[SMPLCameraViewer] No camera trajectory found, viewer will use orbit mode")

        # Run SMPL-X forward pass to get vertices
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data_dir = Path(__file__).parent / "data"
        models_dir = Path(folder_paths.models_dir) / "motion_capture" / "body_models"

        smplx_model = smplx.create(
            model_path=str(models_dir),
            model_type='smplx',
            gender='neutral',
            num_pca_comps=12,
            flat_hand_mean=False,
        ).to(device)
        smplx_model.eval()

        smplx2smpl = torch.load(
            str(data_dir / "smplx2smpl_sparse.pt"), weights_only=True
        ).to(device)

        faces = np.load(str(data_dir / "smpl_faces.npy"))

        vertices_list = []
        incam_vertices_list = []
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

                # Compute incam vertices for through-camera rendering
                if has_incam:
                    go_incam = global_orient_incam[frame_idx:frame_idx+1].to(device)
                    t_incam = transl_incam[frame_idx:frame_idx+1].to(device)
                    smplx_out_incam = smplx_model(
                        body_pose=bp, betas=b, global_orient=go_incam, transl=t_incam,
                    )
                    smpl_verts_incam = torch.matmul(smplx2smpl, smplx_out_incam.vertices[0])
                    incam_vertices_list.append(smpl_verts_incam.cpu().numpy())

        vertices_array = np.stack(vertices_list, axis=0).astype(np.float32)
        incam_vertices_array = np.stack(incam_vertices_list, axis=0).astype(np.float32) if incam_vertices_list else None
        faces_u32 = faces.astype(np.uint32)
        num_output_frames = vertices_array.shape[0]
        num_verts = vertices_array.shape[1]
        fps = 30 // frame_skip

        logger.info(f"[SMPLCameraViewer] Generated mesh: {num_output_frames} frames, "
                     f"{num_verts} vertices, {faces_u32.shape[0]} faces")

        # Subsample camera data to match frame_skip
        if has_camera and frame_skip > 1:
            indices = list(range(0, num_frames, frame_skip))
            R_cam2world_np = R_cam2world_np[indices]
            t_cam2world_np = t_cam2world_np[indices]
            K_fullimg_np = K_fullimg_np[indices]

        # Write SMPC binary format
        output_dir = Path(folder_paths.get_output_directory())
        mesh_filename = _next_sequential_filename(output_dir, "smpl_camera_mesh", ".bin")
        mesh_filepath = output_dir / mesh_filename

        with open(mesh_filepath, "wb") as f:
            f.write(b"SMPC")
            f.write(np.array([num_output_frames], dtype=np.uint32).tobytes())
            f.write(np.array([num_verts], dtype=np.uint32).tobytes())
            f.write(np.array([faces_u32.shape[0]], dtype=np.uint32).tobytes())
            f.write(np.array([float(fps)], dtype=np.float32).tobytes())
            # mesh_color (64 bytes, null-padded)
            color_bytes = mesh_color.encode("utf-8")[:64]
            f.write(color_bytes + b"\x00" * (64 - len(color_bytes)))
            # has_camera flag (u32 for 4-byte alignment)
            f.write(np.array([1 if has_camera else 0], dtype=np.uint32).tobytes())
            # image dimensions
            f.write(np.array([img_width], dtype=np.uint32).tobytes())
            f.write(np.array([img_height], dtype=np.uint32).tobytes())
            # vertex data
            f.write(vertices_array.tobytes())
            # face data
            f.write(faces_u32.tobytes())
            # camera data (if available) — R_cam2world/t_cam2world are in GV Y-up frame
            if has_camera:
                f.write(R_cam2world_np.astype(np.float32).tobytes())
                f.write(t_cam2world_np.astype(np.float32).tobytes())
                f.write(K_fullimg_np.astype(np.float32).tobytes())

            # Incam vertices for through-camera rendering (optional)
            has_incam_data = has_incam and has_camera and incam_vertices_array is not None
            f.write(np.array([1 if has_incam_data else 0], dtype=np.uint32).tobytes())
            if has_incam_data:
                f.write(incam_vertices_array.tobytes())
                logger.info(f"[SMPLCameraViewer] Wrote {incam_vertices_array.shape[0]} incam vertex frames")

            # Image frames (optional) — JPEG-encoded reference video frames
            has_images = images is not None and len(images) > 0
            f.write(np.array([1 if has_images else 0], dtype=np.uint32).tobytes())
            if has_images:
                # Subsample images with frame_skip
                img_indices = list(range(0, len(images), frame_skip))
                num_img_frames = len(img_indices)
                f.write(np.array([num_img_frames], dtype=np.uint32).tobytes())
                logger.info(f"[SMPLCameraViewer] Encoding {num_img_frames} reference frames as JPEG...")

                # JPEG-encode all frames first to get sizes
                jpeg_buffers = []
                for idx in img_indices:
                    frame = images[idx]  # (H, W, C) float32 [0,1]
                    rgb = (frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    ok, jpg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ok:
                        raise RuntimeError(f"Failed to JPEG-encode frame {idx}")
                    jpeg_buffers.append(jpg.tobytes())

                # Write sizes array
                sizes = np.array([len(b) for b in jpeg_buffers], dtype=np.uint32)
                f.write(sizes.tobytes())
                # Write concatenated JPEG data
                for buf in jpeg_buffers:
                    f.write(buf)

                total_img_mb = sum(len(b) for b in jpeg_buffers) / (1024 * 1024)
                logger.info(f"[SMPLCameraViewer] Embedded {num_img_frames} JPEG frames ({total_img_mb:.1f} MB)")

        size_mb = mesh_filepath.stat().st_size / (1024 * 1024)
        logger.info(f"[SMPLCameraViewer] Wrote {mesh_filename} ({size_mb:.1f} MB), has_camera={has_camera}, has_images={has_images}")

        return {
            "ui": {
                "smpl_camera_mesh_file": [mesh_filename]
            },
            "result": (mesh_filename,)
        }


NODE_CLASS_MAPPINGS = {
    "SMPLCameraViewer": SMPLCameraViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SMPLCameraViewer": "SMPL Viewer with Camera",
}
