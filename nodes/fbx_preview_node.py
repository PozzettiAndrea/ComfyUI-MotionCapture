"""
FBX Preview Node - Interactive 3D viewer for FBX characters
"""

from typing import Tuple
from .utils import Log


class FBXPreview:
    """
    Display an interactive 3D preview of an FBX character.
    Shows mesh, skeleton, and allows rotation/zoom.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fbx_path": ("STRING", {
                    "forceInput": True,
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_path",)
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "MotionCapture"

    def preview(self, fbx_path: str) -> Tuple[str,]:
        """
        Display FBX preview in ComfyUI UI.

        Args:
            fbx_path: Absolute path to FBX file

        Returns:
            Tuple with fbx_path (passthrough)
        """
        try:
            Log.info(f"[FBXPreview] Displaying preview for: {fbx_path}")

            # The actual preview is handled by the web extension
            # This node just passes through the path to the frontend
            return (fbx_path,)

        except Exception as e:
            error_msg = f"FBXPreview failed: {str(e)}"
            Log.error(error_msg)
            import traceback
            traceback.print_exc()
            return ("",)


NODE_CLASS_MAPPINGS = {
    "FBXPreview": FBXPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FBXPreview": "FBX 3D Preview",
}
