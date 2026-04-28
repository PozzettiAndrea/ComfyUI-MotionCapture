"""
FBX Animation Viewer Node - Interactive animation playback for animated FBX files
"""

from typing import Tuple

from comfy_api.latest import io

from .motion_utils.pylogger import Log


class FBXAnimationViewer(io.ComfyNode):
    """
    Display an interactive animation viewer for animated FBX files.
    Shows skeletal animation playback with play/pause controls, timeline scrubber,
    and adjustable playback speed.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FBXAnimationViewer",
            display_name="FBX Animation Viewer",
            category="MotionCapture",
            is_output_node=True,
            inputs=[
                io.String.Input("fbx_path", force_input=True),
            ],
            outputs=[
                io.String.Output(display_name="fbx_path"),
            ],
        )

    @classmethod
    def execute(cls, fbx_path: str):
        """
        Display animated FBX playback in ComfyUI UI.

        Args:
            fbx_path: Absolute path to animated FBX file

        Returns:
            NodeOutput with ui data for web extension
        """
        try:
            Log.info(f"[FBXAnimationViewer] Displaying animation for: {fbx_path}")

            # The actual animation viewer is handled by the web extension
            # Return ui dict to send data to onExecuted callback
            return io.NodeOutput(ui={
                "fbx_path": [fbx_path]
            })

        except Exception as e:
            error_msg = f"FBXAnimationViewer failed: {str(e)}"
            Log.error(error_msg, exc_info=True)
            return io.NodeOutput(ui={
                "fbx_path": [""]
            })


NODE_CLASS_MAPPINGS = {
    "FBXAnimationViewer": FBXAnimationViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FBXAnimationViewer": "FBX Animation Viewer",
}
