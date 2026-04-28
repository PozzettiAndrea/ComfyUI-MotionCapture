"""
LoadSMPLParams Node - Load SMPL params from disk.

Loads smpl_params_*.npz files from ComfyUI output folder.
"""

import os
import logging

from comfy_api.latest import io

import folder_paths

from .shared_utils import resolve_file_path

log = logging.getLogger("motioncapture")


class LoadSMPLParams(io.ComfyNode):
    """
    Select an SMPL params .npz file (smpl_params_*.npz).

    Searches both input and output folders.
    Returns the resolved file path.
    """

    @classmethod
    def define_schema(cls):
        npz_files = cls.get_npz_files()
        if not npz_files:
            npz_files = ["No smpl_params files found"]
        return io.Schema(
            node_id="LoadSMPL",
            display_name="Load SMPL Params",
            category="MotionCapture/SMPL",
            inputs=[
                io.Combo.Input("file_path", options=npz_files,
                               tooltip="NPZ file containing SMPL parameters (smpl_params_*.npz from GVHMR Inference)"),
            ],
            outputs=[
                io.String.Output(display_name="file_path"),
            ],
        )

    @staticmethod
    def get_npz_files():
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
    def fingerprint_inputs(cls, **kwargs):
        file_path = kwargs.get("file_path")
        full_path = resolve_file_path(file_path)
        if full_path and os.path.exists(full_path):
            return os.path.getmtime(full_path)
        return file_path

    @classmethod
    def execute(cls, file_path):
        full_path = resolve_file_path(file_path)
        if full_path is None:
            raise FileNotFoundError(f"SMPL params file not found: {file_path}")
        log.info("Selected: %s", full_path)
        return io.NodeOutput(full_path)


NODE_CLASS_MAPPINGS = {
    "LoadSMPL": LoadSMPLParams,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSMPL": "Load SMPL Params",
}
