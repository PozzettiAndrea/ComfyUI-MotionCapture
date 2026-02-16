"""
LoadSMPLParams Node - Load SMPL params from disk.

Loads smpl_params_*.npz files from ComfyUI output folder.
"""

import os
import logging

import folder_paths

from .shared_utils import resolve_file_path

log = logging.getLogger("motioncapture")


class LoadSMPLParams:
    """
    Select an SMPL params .npz file (smpl_params_*.npz).

    Searches both input and output folders.
    Returns the resolved file path.
    """

    @classmethod
    def INPUT_TYPES(cls):
        npz_files = cls.get_npz_files()
        if not npz_files:
            npz_files = ["No smpl_params files found"]
        return {
            "required": {
                "file_path": (npz_files, {
                    "tooltip": "NPZ file containing SMPL parameters (smpl_params_*.npz from GVHMR Inference)"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "load"
    CATEGORY = "MotionCapture/SMPL"

    @classmethod
    def get_npz_files(cls):
        """Get list of smpl_params_*.npz files in input and output folders."""
        npz_files = []

        # Scan input folder
        input_dir = folder_paths.get_input_directory()
        if os.path.exists(input_dir):
            for file in sorted(os.listdir(input_dir)):
                if file.startswith("smpl_params_") and file.endswith(".npz"):
                    npz_files.append(file)

        # Scan output folder
        output_dir = folder_paths.get_output_directory()
        if os.path.exists(output_dir):
            for file in sorted(os.listdir(output_dir)):
                if file.startswith("smpl_params_") and file.endswith(".npz"):
                    npz_files.append(f"[output] {file}")

        return npz_files

    @classmethod
    def IS_CHANGED(cls, file_path):
        full_path = resolve_file_path(file_path)
        if full_path and os.path.exists(full_path):
            return os.path.getmtime(full_path)
        return file_path

    def load(self, file_path):
        full_path = resolve_file_path(file_path)
        if full_path is None:
            raise FileNotFoundError(f"SMPL params file not found: {file_path}")
        log.info("Selected: %s", full_path)
        return (full_path,)


NODE_CLASS_MAPPINGS = {
    "LoadSMPL": LoadSMPLParams,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSMPL": "Load SMPL Params",
}
