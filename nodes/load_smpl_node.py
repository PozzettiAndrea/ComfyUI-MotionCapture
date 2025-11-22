"""
LoadSMPL Node - Load SMPL motion data from disk
"""

from pathlib import Path
from typing import Dict, Tuple
import torch
import numpy as np

from .utils import Log


class LoadSMPL:
    """
    Load SMPL motion parameters from .npz file.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("SMPL_PARAMS", "STRING")
    RETURN_NAMES = ("smpl_params", "info")
    FUNCTION = "load_smpl"
    CATEGORY = "MotionCapture/SMPL"

    def load_smpl(
        self,
        file_path: str,
    ) -> Tuple[Dict, str]:
        """
        Load SMPL parameters from NPZ file.

        Args:
            file_path: Path to NPZ file

        Returns:
            Tuple of (smpl_params, info_string)
        """
        try:
            Log.info("[LoadSMPL] Loading SMPL motion data...")

            # Validate input
            file_path = Path(file_path)
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
