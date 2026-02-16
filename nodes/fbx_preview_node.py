"""
FBX Preview Node - Interactive 3D viewer for FBX meshes using comfy-3d-viewers.

Displays rigged FBX files with Three.js viewer and skeleton visualization.
"""

import os
import logging
from pathlib import Path

try:
    import folder_paths
except ImportError:
    folder_paths = None

log = logging.getLogger("motioncapture")


class MocapPreviewRiggedMesh:
    """
    Preview rigged mesh with interactive FBX viewer.

    Displays the rigged FBX in a Three.js viewer with skeleton visualization
    and interactive controls. Uses the shared comfy-3d-viewers infrastructure.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fbx_output_path": ("STRING", {
                    "tooltip": "FBX filename from output directory"
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "MotionCapture/Visualization"

    def preview(self, fbx_output_path):
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
            return {
                "ui": {
                    "fbx_file": [fbx_output_path],
                    "has_skinning": [False],
                    "has_skeleton": [False],
                    "error": ["File not found"],
                }
            }

        log.info("FBX path: %s", fbx_path)

        # Assume FBX files have skinning and skeleton (retargeted animations)
        has_skinning = True
        has_skeleton = True

        log.debug("Has skinning: %s", has_skinning)
        log.debug("Has skeleton: %s", has_skeleton)

        return {
            "ui": {
                "fbx_file": [fbx_output_path],
                "has_skinning": [bool(has_skinning)],
                "has_skeleton": [bool(has_skeleton)],
            }
        }


# Keep FBXPreview as an alias for backwards compatibility
class FBXPreview(MocapPreviewRiggedMesh):
    """Alias for MocapPreviewRiggedMesh for backwards compatibility."""
    CATEGORY = "MotionCapture"


NODE_CLASS_MAPPINGS = {
    "MocapPreviewRiggedMesh": MocapPreviewRiggedMesh,
    "FBXPreview": FBXPreview,  # Keep old name for compatibility
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MocapPreviewRiggedMesh": "Mocap: Preview Rigged Mesh",
    "FBXPreview": "FBX 3D Preview (Legacy)",
}
