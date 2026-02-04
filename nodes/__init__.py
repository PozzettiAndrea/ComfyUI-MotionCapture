"""MotionCapture Nodes."""

from .nodes_gpu import NODE_CLASS_MAPPINGS as gpu_mappings
from .nodes_gpu import NODE_DISPLAY_NAME_MAPPINGS as gpu_display
from .nodes_blender import NODE_CLASS_MAPPINGS as blender_mappings
from .nodes_blender import NODE_DISPLAY_NAME_MAPPINGS as blender_display
from .viewer_node import NODE_CLASS_MAPPINGS as viewer_mappings
from .viewer_node import NODE_DISPLAY_NAME_MAPPINGS as viewer_display

NODE_CLASS_MAPPINGS = {
    **gpu_mappings,
    **blender_mappings,
    **viewer_mappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **gpu_display,
    **blender_display,
    **viewer_display,
}
