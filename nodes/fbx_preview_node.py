"""
FBX Preview Node - Interactive 3D viewer for FBX meshes using comfy-3d-viewers.

Displays rigged FBX files with Three.js viewer and skeleton visualization.
"""

import os
import logging
from pathlib import Path

from comfy_api.latest import io

try:
    import folder_paths
except ImportError:
    folder_paths = None

log = logging.getLogger("motioncapture")


class MocapPreviewRiggedMesh(io.ComfyNode):
    """
    Preview rigged mesh with interactive FBX viewer.

    Displays the rigged FBX in a Three.js viewer with skeleton visualization
    and interactive controls. Uses the shared comfy-3d-viewers infrastructure.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MocapPreviewRiggedMesh",
            display_name="Mocap: Preview Rigged Mesh",
            category="MotionCapture/Visualization",
            is_output_node=True,
            inputs=[
                io.String.Input("fbx_output_path", tooltip="FBX filename from output directory"),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, fbx_output_path):
        """Preview the rigged mesh in an interactive FBX viewer."""
        log.info("Preparing preview...")

        # Get output directory
        if folder_paths:
            output_dir = folder_paths.get_output_directory()
        else:
            output_dir = Path("output")

        fbx_path = os.path.join(output_dir, fbx_output_path)

        if not os.path.exists(fbx_path):
            log.warning("FBX file not found: %s", fbx_output_path)
            return io.NodeOutput(ui={
                "fbx_file": [fbx_output_path],
                "has_skinning": [False],
                "has_skeleton": [False],
                "error": ["File not found"],
            })

        log.info("FBX path: %s", fbx_path)

        # Assume FBX files have skinning and skeleton (retargeted animations)
        has_skinning = True
        has_skeleton = True

        log.debug("Has skinning: %s", has_skinning)
        log.debug("Has skeleton: %s", has_skeleton)

        return io.NodeOutput(ui={
            "fbx_file": [fbx_output_path],
            "has_skinning": [bool(has_skinning)],
            "has_skeleton": [bool(has_skeleton)],
        })


# Keep FBXPreview as an alias for backwards compatibility
class FBXPreview(MocapPreviewRiggedMesh):
    """Alias for MocapPreviewRiggedMesh for backwards compatibility."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FBXPreview",
            display_name="FBX 3D Preview (Legacy)",
            category="MotionCapture",
            is_output_node=True,
            inputs=[
                io.String.Input("fbx_output_path", tooltip="FBX filename from output directory"),
            ],
            outputs=[],
        )


NODE_CLASS_MAPPINGS = {
    "MocapPreviewRiggedMesh": MocapPreviewRiggedMesh,
    "FBXPreview": FBXPreview,  # Keep old name for compatibility
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MocapPreviewRiggedMesh": "Mocap: Preview Rigged Mesh",
    "FBXPreview": "FBX 3D Preview (Legacy)",
}
