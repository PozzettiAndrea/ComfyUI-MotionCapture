"""
LoadCameraTrajectory Node - Load camera trajectory from disk.

Loads camera_trajectory_*.npz files from ComfyUI output folder.
"""

import os
import logging

from comfy_api.latest import io

import folder_paths

from .shared_utils import resolve_file_path

log = logging.getLogger("motioncapture")


class LoadCameraTrajectory(io.ComfyNode):
    """
    Select a camera trajectory .npz file (camera_trajectory_*.npz).

    Searches both input and output folders.
    Returns the resolved file path.
    """

    @classmethod
    def define_schema(cls):
        npz_files = cls.get_npz_files()
        if not npz_files:
            npz_files = ["No camera_trajectory files found"]
        return io.Schema(
            node_id="LoadCameraTrajectory",
            display_name="Load Camera Trajectory",
            category="MotionCapture/SMPL",
            inputs=[
                io.Combo.Input("file_path", options=npz_files,
                               tooltip="NPZ file containing camera trajectory (camera_trajectory_*.npz from GVHMR moving camera)"),
            ],
            outputs=[
                io.String.Output(display_name="camera_npz_path"),
            ],
        )

    @staticmethod
    def get_npz_files():
        """Get list of camera_trajectory_*.npz files in input and output folders."""
        npz_files = []

        input_dir = folder_paths.get_input_directory()
        if os.path.exists(input_dir):
            for file in sorted(os.listdir(input_dir)):
                if file.startswith("camera_trajectory_") and file.endswith(".npz"):
                    npz_files.append(file)

        output_dir = folder_paths.get_output_directory()
        if os.path.exists(output_dir):
            for file in sorted(os.listdir(output_dir)):
                if file.startswith("camera_trajectory_") and file.endswith(".npz"):
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
            raise FileNotFoundError(f"Camera trajectory file not found: {file_path}")
        log.info("Selected: %s", full_path)
        return io.NodeOutput(full_path)


NODE_CLASS_MAPPINGS = {
    "LoadCameraTrajectory": LoadCameraTrajectory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadCameraTrajectory": "Load Camera Trajectory",
}
