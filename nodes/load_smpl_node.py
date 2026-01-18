"""
LoadSMPL Node - Load SMPL motion data from disk
"""

import os
from pathlib import Path
from typing import Dict, Tuple, List
import torch
import numpy as np
import folder_paths

from hmr4d.utils.pylogger import Log


class LoadSMPL:
    """
    Load SMPL motion parameters from .npz file.
    """

    @staticmethod
    def get_npz_files_from_input() -> List[str]:
        """Get all NPZ files from input directory recursively."""
        try:
            input_dir = folder_paths.get_input_directory()
            npz_files = []

            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith('.npz'):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, input_dir)
                        npz_files.append(rel_path)

            return sorted(npz_files)
        except Exception as e:
            Log.error(f"[LoadSMPL] Error scanning input directory: {e}")
            return []

    @staticmethod
    def get_npz_files_from_output() -> List[str]:
        """Get all NPZ files from output directory recursively."""
        try:
            output_dir = folder_paths.get_output_directory()
            npz_files = []

            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.lower().endswith('.npz'):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, output_dir)
                        npz_files.append(rel_path)

            return sorted(npz_files)
        except Exception as e:
            Log.error(f"[LoadSMPL] Error scanning output directory: {e}")
            return []

    @classmethod
    def INPUT_TYPES(cls):
        # Get files from both directories for server-side validation
        npz_files = list(set(
            cls.get_npz_files_from_output() + cls.get_npz_files_from_input()
        ))
        if not npz_files:
            npz_files = [""]  # Provide fallback to avoid empty list

        return {
            "required": {
                "npz_file": (npz_files, {
                    "remote": {
                        "route": "/motioncapture/npz_files",
                        "refresh_button": True,
                    },
                }),
                "source_folder": (["output", "input"],),
            },
        }

    RETURN_TYPES = ("SMPL_PARAMS", "STRING")
    RETURN_NAMES = ("smpl_params", "info")
    FUNCTION = "load_smpl"
    CATEGORY = "MotionCapture/SMPL"

    def load_smpl(
        self,
        npz_file: str,
        source_folder: str,
    ) -> Tuple[Dict, str]:
        """
        Load SMPL parameters from NPZ file.

        Args:
            npz_file: Relative path to NPZ file within source folder
            source_folder: Source folder ("input" or "output")

        Returns:
            Tuple of (smpl_params, info_string)
        """
        try:
            Log.info("[LoadSMPL] Loading SMPL motion data...")

            # Get base directory based on source folder
            if source_folder == "input":
                base_dir = folder_paths.get_input_directory()
            else:
                base_dir = folder_paths.get_output_directory()

            # Construct full path
            file_path = os.path.join(base_dir, npz_file)
            file_path = Path(os.path.abspath(file_path))

            # Validate input
            if not file_path.exists():
                raise FileNotFoundError(f"SMPL file not found: {file_path}")

            # Load NPZ file
            data = np.load(file_path)

            # Convert to torch tensors
            global_params = {}
            for key in data.files:
                global_params[key] = torch.from_numpy(data[key])

            # Create SMPL_PARAMS structure (matching GVHMRInference output)
            smpl_params = {
                "global": global_params,
                "incam": global_params,  # Use same for both (global coordinates)
            }

            # Get info
            num_frames = global_params.get("body_pose", torch.tensor([])).shape[0] if "body_pose" in global_params else 0
            file_size_kb = file_path.stat().st_size / 1024

            info = (
                f"LoadSMPL Complete\n"
                f"Input: {file_path}\n"
                f"Frames: {num_frames}\n"
                f"File size: {file_size_kb:.1f} KB\n"
                f"Parameters: {', '.join(global_params.keys())}\n"
            )

            Log.info(f"[LoadSMPL] Loaded {num_frames} frames from {file_path}")
            return (smpl_params, info)

        except Exception as e:
            error_msg = f"LoadSMPL failed: {str(e)}"
            Log.error(error_msg)
            import traceback
            traceback.print_exc()
            return ({}, error_msg)


NODE_CLASS_MAPPINGS = {
    "LoadSMPL": LoadSMPL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSMPL": "Load SMPL Motion",
}
