"""
BVHViewer Node - Interactive 3D viewer for BVH skeletal animations
"""

from pathlib import Path
from typing import Dict, Tuple

from comfy_api.latest import io

from .motion_utils.pylogger import Log


class BVHViewer(io.ComfyNode):
    """
    Display BVH skeletal animations in an interactive 3D viewer.
    Uses Three.js BVHLoader for visualization.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BVHViewer",
            display_name="BVH Animation Viewer",
            category="MotionCapture/BVH",
            is_output_node=True,
            inputs=[
                io.Custom("BVH_DATA").Input("bvh_data"),
            ],
            outputs=[
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, bvh_data: Dict) -> io.NodeOutput:
        """
        Display BVH animation in interactive viewer.

        Args:
            bvh_data: BVH data dictionary from SMPLtoBVH node

        Returns:
            NodeOutput with info string and UI data
        """
        try:
            Log.info("[BVHViewer] Loading BVH for visualization...")

            # Get BVH file path
            file_path = bvh_data.get("file_path", "")
            num_frames = bvh_data.get("num_frames", 0)
            fps = bvh_data.get("fps", 30)

            Log.info(f"[BVHViewer DEBUG] BVH file path: {file_path}")
            Log.info(f"[BVHViewer DEBUG] Num frames: {num_frames}, FPS: {fps}")
            Log.info(f"[BVHViewer DEBUG] File exists: {Path(file_path).exists() if file_path else False}")

            if not file_path or not Path(file_path).exists():
                raise ValueError(f"BVH file not found: {file_path}")

            # Read BVH file content
            with open(file_path, 'r') as f:
                bvh_content = f.read()

            Log.info(f"[BVHViewer DEBUG] BVH content length: {len(bvh_content)} bytes")
            Log.info(f"[BVHViewer DEBUG] First 200 chars: {bvh_content[:200]}")

            bvh_info = {
                "num_frames": num_frames,
                "fps": fps,
                "file_path": file_path,
            }

            Log.info(f"[BVHViewer DEBUG] Stored bvh_content ({len(bvh_content)} bytes) and bvh_info for frontend")
            Log.info(f"[BVHViewer DEBUG] bvh_info: {bvh_info}")

            info = (
                f"BVH Viewer Ready\n"
                f"File: {Path(file_path).name}\n"
                f"Frames: {num_frames}\n"
                f"FPS: {fps}\n"
                f"Joints: {len(bvh_data.get('joint_names', []))}\n"
            )

            Log.info(f"[BVHViewer] Loaded BVH with {num_frames} frames")

            # Return data in ComfyUI OUTPUT_NODE format
            # The "ui" dict is sent to the frontend JavaScript
            return io.NodeOutput(info, ui={
                "bvh_content": [bvh_content],
                "bvh_info": [bvh_info]
            })

        except Exception as e:
            error_msg = f"BVHViewer failed: {str(e)}"
            Log.error(error_msg, exc_info=True)
            return io.NodeOutput(error_msg, ui={
                "bvh_content": [""],
                "bvh_info": [{}]
            })

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # Always update when input changes
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "BVHViewer": BVHViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BVHViewer": "BVH Animation Viewer",
}
