"""
FBX Loader Node - Load rigged FBX characters with folder browsing
"""

import os
from pathlib import Path
from typing import Tuple, List
import folder_paths

from comfy_api.latest import io

from .motion_utils.pylogger import Log


class LoadFBXCharacter(io.ComfyNode):
    """
    Load a rigged FBX character from input or output folders.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadFBXCharacter",
            display_name="Load FBX Character",
            category="MotionCapture",
            inputs=[
                io.Combo.Input("source_folder", options=["input", "output"], default="input"),
                io.Combo.Input("fbx_file", options=[], remote={
                    "route": "/motioncapture/fbx_files",
                    "query": "source_folder",
                    "refresh_button": True,
                }),
            ],
            outputs=[
                io.String.Output(display_name="fbx_path"),
                io.String.Output(display_name="info"),
            ],
        )

    @staticmethod
    def get_fbx_files_from_input() -> List[str]:
        """Get all FBX files from input directory."""
        try:
            input_dir = folder_paths.get_input_directory()
            fbx_files = []

            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith('.fbx'):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, input_dir)
                        fbx_files.append(rel_path)

            return sorted(fbx_files)
        except Exception as e:
            Log.error(f"[LoadFBXCharacter] Error scanning input directory: {e}")
            return []

    @staticmethod
    def get_fbx_files_from_output() -> List[str]:
        """Get all FBX files from output directory."""
        try:
            output_dir = folder_paths.get_output_directory()
            fbx_files = []

            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.lower().endswith('.fbx'):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, output_dir)
                        fbx_files.append(rel_path)

            return sorted(fbx_files)
        except Exception as e:
            Log.error(f"[LoadFBXCharacter] Error scanning output directory: {e}")
            return []

    @classmethod
    def execute(
        cls,
        fbx_file: str,
        source_folder: str,
    ) -> io.NodeOutput:
        """
        Load FBX character and return path.

        Args:
            fbx_file: Relative path to FBX file
            source_folder: "input" or "output"

        Returns:
            NodeOutput with (absolute_fbx_path, info_string)
        """
        try:
            Log.info(f"[LoadFBXCharacter] Loading FBX: {fbx_file}")

            # Get base directory
            if source_folder == "input":
                base_dir = folder_paths.get_input_directory()
            else:
                base_dir = folder_paths.get_output_directory()

            # Construct full path
            fbx_path = os.path.join(base_dir, fbx_file)
            fbx_path = os.path.abspath(fbx_path)

            # Validate file exists
            if not os.path.exists(fbx_path):
                raise FileNotFoundError(f"FBX file not found: {fbx_path}")

            # Get file info
            file_size = os.path.getsize(fbx_path) / (1024 * 1024)  # MB

            info = (
                f"FBX Character Loaded\n"
                f"File: {fbx_file}\n"
                f"Source: {source_folder}\n"
                f"Full path: {fbx_path}\n"
                f"Size: {file_size:.2f} MB\n"
            )

            Log.info(f"[LoadFBXCharacter] FBX loaded successfully: {fbx_path}")
            return io.NodeOutput(fbx_path, info)

        except Exception as e:
            error_msg = f"LoadFBXCharacter failed: {str(e)}"
            Log.error(error_msg, exc_info=True)
            return io.NodeOutput("", error_msg)


NODE_CLASS_MAPPINGS = {
    "LoadFBXCharacter": LoadFBXCharacter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFBXCharacter": "Load FBX Character",
}
