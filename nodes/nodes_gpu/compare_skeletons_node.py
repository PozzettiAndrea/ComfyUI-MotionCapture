"""
MotionCapture Compare Skeletons Node
Compare two skeletons side-by-side with synced rotation.
"""

import os
from typing import Tuple

try:
    import folder_paths
except ImportError:
    folder_paths = None


class CompareSkeletons:
    """
    Compare two skeletons side-by-side with synced rotation.

    Opens two FBX files in a split-view debug viewer where:
    - Both skeletons are displayed side-by-side
    - Camera rotation and zoom are synced between views
    - Clicking a bone in one view highlights the matching bone (by name) in the other
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fbx_path_left": ("STRING", {
                    "tooltip": "Path to left skeleton FBX file"
                }),
                "fbx_path_right": ("STRING", {
                    "tooltip": "Path to right skeleton FBX file"
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "compare_skeletons"
    CATEGORY = "MotionCapture/Skeleton"

    def compare_skeletons(self, fbx_path_left: str, fbx_path_right: str):
        """Open both FBX files in the comparison skeleton viewer."""
        print(f"[CompareSkeletons] Preparing skeleton comparison view...")

        if folder_paths:
            output_dir = folder_paths.get_output_directory()
            input_dir = folder_paths.get_input_directory()
        else:
            output_dir = "output"
            input_dir = "input"

        # Validate left FBX path
        if os.path.isabs(fbx_path_left):
            full_path_left = fbx_path_left
        else:
            # Check output dir first, then input dir
            if os.path.exists(os.path.join(output_dir, fbx_path_left)):
                full_path_left = os.path.join(output_dir, fbx_path_left)
            elif os.path.exists(os.path.join(input_dir, fbx_path_left)):
                full_path_left = os.path.join(input_dir, fbx_path_left)
            else:
                full_path_left = os.path.join(output_dir, fbx_path_left)

        if not os.path.exists(full_path_left):
            raise RuntimeError(f"Left FBX file not found: {fbx_path_left}")

        # Validate right FBX path
        if os.path.isabs(fbx_path_right):
            full_path_right = fbx_path_right
        else:
            if os.path.exists(os.path.join(output_dir, fbx_path_right)):
                full_path_right = os.path.join(output_dir, fbx_path_right)
            elif os.path.exists(os.path.join(input_dir, fbx_path_right)):
                full_path_right = os.path.join(input_dir, fbx_path_right)
            else:
                full_path_right = os.path.join(output_dir, fbx_path_right)

        if not os.path.exists(full_path_right):
            raise RuntimeError(f"Right FBX file not found: {fbx_path_right}")

        print(f"[CompareSkeletons] Left FBX: {full_path_left}")
        print(f"[CompareSkeletons] Right FBX: {full_path_right}")

        # For the viewer, use relative path if in output, otherwise basename
        if os.path.isabs(fbx_path_left):
            viewer_filename_left = os.path.basename(fbx_path_left)
        else:
            viewer_filename_left = fbx_path_left

        if os.path.isabs(fbx_path_right):
            viewer_filename_right = os.path.basename(fbx_path_right)
        else:
            viewer_filename_right = fbx_path_right

        return {
            "ui": {
                "fbx_file_left": [viewer_filename_left],
                "fbx_file_right": [viewer_filename_right],
            }
        }
