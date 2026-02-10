"""
LoadCameraTrajectory Node - Load camera trajectory from disk.

Loads camera_trajectory_*.npz files from ComfyUI output folder.
"""

import os
import folder_paths


class LoadCameraTrajectory:
    """
    Select a camera trajectory .npz file (camera_trajectory_*.npz).

    Searches both input and output folders.
    Returns the resolved file path.
    """

    @classmethod
    def INPUT_TYPES(cls):
        npz_files = cls.get_npz_files()
        if not npz_files:
            npz_files = ["No camera_trajectory files found"]
        return {
            "required": {
                "file_path": (npz_files, {
                    "tooltip": "NPZ file containing camera trajectory (camera_trajectory_*.npz from GVHMR moving camera)"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("camera_npz_path",)
    FUNCTION = "load"
    CATEGORY = "MotionCapture/SMPL"

    @classmethod
    def get_npz_files(cls):
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
    def IS_CHANGED(cls, file_path):
        full_path = cls._resolve_file_path(file_path)
        if full_path and os.path.exists(full_path):
            return os.path.getmtime(full_path)
        return file_path

    @classmethod
    def _resolve_file_path(cls, file_path):
        if file_path.startswith("[output] "):
            clean_path = file_path.replace("[output] ", "")
            output_dir = folder_paths.get_output_directory()
            output_path = os.path.join(output_dir, clean_path)
            if os.path.exists(output_path):
                return output_path
        else:
            input_dir = folder_paths.get_input_directory()
            input_path = os.path.join(input_dir, file_path)
            if os.path.exists(input_path):
                return input_path

        if os.path.exists(file_path):
            return file_path

        return None

    def load(self, file_path):
        full_path = self._resolve_file_path(file_path)
        if full_path is None:
            raise FileNotFoundError(f"Camera trajectory file not found: {file_path}")
        print(f"[LoadCameraTrajectory] Selected: {full_path}")
        return (full_path,)


NODE_CLASS_MAPPINGS = {
    "LoadCameraTrajectory": LoadCameraTrajectory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadCameraTrajectory": "Load Camera Trajectory",
}
