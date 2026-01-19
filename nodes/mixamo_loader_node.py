"""
LoadMixamoCharacter Node - Load Mixamo-rigged FBX characters from input/3d folder
"""

import os
from pathlib import Path
from typing import Tuple, List
import folder_paths

# Use print for logging to avoid dependency on hmr4d in isolated contexts


class LoadMixamoCharacter:
    """
    Load a Mixamo-rigged FBX character from input/3d folder.

    This node provides a dropdown of FBX files in ComfyUI's input/3d directory,
    which is where Mixamo characters should be placed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get files for server-side validation
        fbx_files = cls.get_mixamo_files()
        if not fbx_files:
            fbx_files = [""]  # Provide fallback to avoid empty list error

        return {
            "required": {
                "fbx_file": (fbx_files, {
                    "remote": {
                        "route": "/motioncapture/mixamo_files",
                        "refresh_button": True,
                    },
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("fbx_path", "info")
    FUNCTION = "load_mixamo"
    CATEGORY = "MotionCapture/Mixamo"

    @staticmethod
    def get_mixamo_files() -> List[str]:
        """Get all FBX files from input/3d directory."""
        try:
            input_dir = folder_paths.get_input_directory()
            input_3d_dir = os.path.join(input_dir, "3d")

            if not os.path.exists(input_3d_dir):
                print(f"[LoadMixamoCharacter] input/3d directory not found: {input_3d_dir}")
                return []

            fbx_files = []
            for root, dirs, files in os.walk(input_3d_dir):
                for file in files:
                    if file.lower().endswith('.fbx'):
                        full_path = os.path.join(root, file)
                        # Return path relative to input/3d
                        rel_path = os.path.relpath(full_path, input_3d_dir)
                        fbx_files.append(rel_path)

            return sorted(fbx_files)
        except Exception as e:
            print(f"[LoadMixamoCharacter] Error scanning 3d directory: {e}")
            return []

    def load_mixamo(self, fbx_file: str) -> Tuple[str, str]:
        """
        Load Mixamo FBX character and return path.

        Args:
            fbx_file: Relative path to FBX file within input/3d

        Returns:
            Tuple of (absolute_fbx_path, info_string)
        """
        try:
            print(f"[LoadMixamoCharacter] Loading Mixamo FBX: {fbx_file}")

            input_dir = folder_paths.get_input_directory()
            input_3d_dir = os.path.join(input_dir, "3d")
            fbx_path = os.path.join(input_3d_dir, fbx_file)
            fbx_path = os.path.abspath(fbx_path)

            if not os.path.exists(fbx_path):
                raise FileNotFoundError(f"Mixamo FBX file not found: {fbx_path}")

            file_size = os.path.getsize(fbx_path) / (1024 * 1024)  # MB

            info = (
                f"Mixamo Character Loaded\n"
                f"File: {fbx_file}\n"
                f"Source: input/3d\n"
                f"Full path: {fbx_path}\n"
                f"Size: {file_size:.2f} MB\n"
            )

            print(f"[LoadMixamoCharacter] FBX loaded: {fbx_path}")
            return (fbx_path, info)

        except Exception as e:
            error_msg = f"LoadMixamoCharacter failed: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return ("", error_msg)


NODE_CLASS_MAPPINGS = {
    "LoadMixamoCharacter": LoadMixamoCharacter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadMixamoCharacter": "Load Mixamo Character",
}
