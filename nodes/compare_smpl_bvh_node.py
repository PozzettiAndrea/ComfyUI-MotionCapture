"""
CompareSMPLtoBVH Node - Side-by-side comparison of SMPL and BVH animations

Displays SMPL mesh and BVH skeleton side-by-side with synchronized playback
and camera controls for easy comparison.
"""

from pathlib import Path
from typing import Dict, Tuple
import torch
import numpy as np
import folder_paths

from comfy_api.latest import io

from .motion_utils.pylogger import Log

from .shared_utils import next_sequential_filename as _next_sequential_filename


class CompareSMPLtoBVH(io.ComfyNode):
    """
    Side-by-side comparison viewer for SMPL and BVH animations.
    Synchronized playback and camera for easy comparison.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CompareSMPLtoBVH",
            display_name="Compare SMPL vs BVH",
            category="MotionCapture/Comparison",
            is_output_node=True,
            inputs=[
                io.Custom("SMPL_PARAMS").Input("smpl_params"),
                io.Custom("BVH_DATA").Input("bvh_data"),
            ],
            outputs=[
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        smpl_params: Dict,
        bvh_data: Dict,
    ) -> io.NodeOutput:
        """
        Display SMPL and BVH animations side-by-side.

        Args:
            smpl_params: SMPL parameters from GVHMR or LoadSMPL
            bvh_data: BVH data from SMPLtoBVH

        Returns:
            NodeOutput with info string and UI data
        """
        try:
            Log.info("[CompareSMPLtoBVH] Loading animations for comparison...")

            # Generate SMPL mesh and save to file
            Log.info("[CompareSMPLtoBVH] Generating SMPL mesh...")

            # Import SMPL model
            from .body_model.smplx_lite import SmplxLite

            # Extract SMPL parameters
            params = smpl_params['global']
            body_pose = params['body_pose']  # (F, 63)
            betas = params['betas']  # (F, 10)
            global_orient = params['global_orient']  # (F, 3)
            transl = params.get('transl', None)  # (F, 3) or None

            num_frames = body_pose.shape[0]
            smpl_frames = num_frames

            # Initialize SMPL model
            smpl_model = SmplxLite(gender="neutral", num_betas=10)
            smpl_model.eval()
            device = body_pose.device
            smpl_model = smpl_model.to(device)

            # Generate mesh for each frame
            Log.info(f"[CompareSMPLtoBVH] Generating {num_frames} frames...")
            vertices_list = []
            with torch.no_grad():
                for frame_idx in range(num_frames):
                    bp = body_pose[frame_idx:frame_idx+1]
                    b = betas[frame_idx:frame_idx+1]
                    go = global_orient[frame_idx:frame_idx+1]
                    t = transl[frame_idx:frame_idx+1] if transl is not None else None

                    verts = smpl_model.forward(
                        body_pose=bp,
                        betas=b,
                        global_orient=go,
                        transl=t,
                        rotation_type="aa"
                    )
                    vertices_list.append(verts[0].cpu().numpy())

            vertices_array = np.stack(vertices_list, axis=0)  # (F, V, 3)
            faces = smpl_model.faces.astype(np.int32)  # (Nf, 3)

            # Save mesh to custom binary format (.bin) for easier JS loading
            output_dir = Path(folder_paths.get_output_directory())
            output_dir.mkdir(parents=True, exist_ok=True)
            mesh_filename = _next_sequential_filename(output_dir, "smpl_mesh", ".bin")
            mesh_filepath = output_dir / mesh_filename

            # Create binary header and data
            # Header: Magic(4), Frames(4), Verts(4), Faces(4), FPS(4)
            magic = b"SMPL"
            num_frames_u32 = np.array([num_frames], dtype=np.uint32)
            num_verts_u32 = np.array([vertices_array.shape[1]], dtype=np.uint32)
            num_faces_u32 = np.array([faces.shape[0]], dtype=np.uint32)
            fps_f32 = np.array([30.0], dtype=np.float32)

            with open(mesh_filepath, "wb") as f:
                f.write(magic)
                f.write(num_frames_u32.tobytes())
                f.write(num_verts_u32.tobytes())
                f.write(num_faces_u32.tobytes())
                f.write(fps_f32.tobytes())
                f.write(vertices_array.astype(np.float32).tobytes())
                f.write(faces.astype(np.uint32).tobytes())

            Log.info(f"[CompareSMPLtoBVH] Saved mesh to {mesh_filepath} ({mesh_filepath.stat().st_size / 1024 / 1024:.1f} MB)")

            # Read BVH file content
            bvh_file_path = bvh_data.get("file_path", "")
            if not bvh_file_path or not Path(bvh_file_path).exists():
                raise ValueError(f"BVH file not found: {bvh_file_path}")

            with open(bvh_file_path, 'r') as f:
                bvh_content = f.read()

            # Store data for web viewer
            smpl_mesh_filename = mesh_filename
            bvh_info = {
                "num_frames": bvh_data.get("num_frames", 0),
                "fps": bvh_data.get("fps", 30),
                "file_path": bvh_file_path,
            }

            bvh_frames = bvh_data.get("num_frames", 0)

            info = (
                f"SMPL vs BVH Comparison\n"
                f"SMPL Frames: {smpl_frames}\n"
                f"BVH Frames: {bvh_frames}\n"
                f"BVH File: {Path(bvh_file_path).name}\n"
                f"Synchronized playback and camera\n"
            )

            Log.info(f"[CompareSMPLtoBVH] Ready for comparison - SMPL: {smpl_frames} frames, BVH: {bvh_frames} frames")

            # Return data in ComfyUI OUTPUT_NODE format
            return io.NodeOutput(info, ui={
                "smpl_mesh_filename": [smpl_mesh_filename],
                "bvh_content": [bvh_content],
                "bvh_info": [bvh_info]
            })

        except Exception as e:
            error_msg = f"CompareSMPLtoBVH failed: {str(e)}"
            Log.error(error_msg, exc_info=True)
            return io.NodeOutput(error_msg, ui={
                "smpl_mesh_file": [""],
                "bvh_content": [""],
                "bvh_info": [{}]
            })

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # Always update when input changes
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "CompareSMPLtoBVH": CompareSMPLtoBVH,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CompareSMPLtoBVH": "Compare SMPL vs BVH",
}
