"""
LoadMixamoCharacter Node - Load Mixamo-rigged FBX characters.

Searches both input and output folders for .fbx files.
"""

import os
import logging

from comfy_api.latest import io

import folder_paths

from .shared_utils import resolve_file_path

log = logging.getLogger("motioncapture")


class LoadMixamoCharacter(io.ComfyNode):
    """
    Load a Mixamo-rigged FBX character.

    Searches both input and output folders for .fbx files.
    Returns the resolved file path.
    """

    @classmethod
    def define_schema(cls):
        fbx_files = cls._get_fbx_files()
        if not fbx_files:
            fbx_files = ["No .fbx files found"]
        return io.Schema(
            node_id="LoadMixamoCharacter",
            display_name="Load Mixamo Character",
            category="MotionCapture/Mixamo",
            inputs=[
                io.Combo.Input("fbx_file", options=fbx_files,
                               tooltip="FBX file containing Mixamo-rigged character"),
            ],
            outputs=[
                io.String.Output(display_name="fbx_path"),
                io.String.Output(display_name="info"),
            ],
        )

    @staticmethod
    def _scan_fbx(base_dir, prefix=""):
        """Recursively scan directory for .fbx files."""
        fbx_files = []
        if not os.path.exists(base_dir):
            return fbx_files
        for root, _dirs, files in os.walk(base_dir):
            for file in sorted(files):
                if file.lower().endswith('.fbx'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, base_dir)
                    if prefix:
                        fbx_files.append(f"{prefix}{rel_path}")
                    else:
                        fbx_files.append(rel_path)
        return fbx_files

    @staticmethod
    def _get_fbx_files():
        """Get list of .fbx files in input and output folders."""
        fbx_files = []

        # Scan input folder
        input_dir = folder_paths.get_input_directory()
        fbx_files.extend(LoadMixamoCharacter._scan_fbx(input_dir))

        # Scan output folder
        output_dir = folder_paths.get_output_directory()
        fbx_files.extend(LoadMixamoCharacter._scan_fbx(output_dir, prefix="[output] "))

        return fbx_files

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        fbx_file = kwargs.get("fbx_file")
        full_path = resolve_file_path(fbx_file)
        if full_path and os.path.exists(full_path):
            return os.path.getmtime(full_path)
        return fbx_file

    @classmethod
    def execute(cls, fbx_file):
        full_path = resolve_file_path(fbx_file)
        if full_path is None:
            raise FileNotFoundError(f"Mixamo FBX file not found: {fbx_file}")

        file_size = os.path.getsize(full_path) / (1024 * 1024)  # MB
        source = "output" if fbx_file.startswith("[output] ") else "input"

        info = (
            f"Mixamo Character Loaded\n"
            f"File: {fbx_file}\n"
            f"Source: {source}\n"
            f"Full path: {full_path}\n"
            f"Size: {file_size:.2f} MB\n"
        )

        log.info("Selected: %s", full_path)
        return io.NodeOutput(full_path, info)


NODE_CLASS_MAPPINGS = {
    "LoadMixamoCharacter": LoadMixamoCharacter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadMixamoCharacter": "Load Mixamo Character",
}
