"""
SaveSMPL Node - Save SMPL motion data to disk for reuse
"""

from pathlib import Path
from typing import Dict, Tuple
import torch
import numpy as np

from .utils import Log


class SaveSMPL:
    """
    Save SMPL motion parameters to .npz file for caching and reuse.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "smpl_params": ("SMPL_PARAMS",),
                "output_path": ("STRING", {
                    "default": "output/motion.npz",
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_path", "info")
    FUNCTION = "save_smpl"
    CATEGORY = "MotionCapture/SMPL"

    def save_smpl(
        self,
        smpl_params: Dict,
        output_path: str,
    ) -> Tuple[str, str]:
        """
        Save SMPL parameters to NPZ file.

        Args:
            smpl_params: SMPL parameters from GVHMRInference
            output_path: Path to save NPZ file

        Returns:
            Tuple of (file_path, info_string)
        """
        try:
            Log.info("[SaveSMPL] Saving SMPL motion data...")

            # Prepare output directory
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Ensure .npz extension
            if not output_path.suffix == '.npz':
                output_path = output_path.with_suffix('.npz')

            # Extract global parameters (these are the ones used for retargeting)
            global_params = smpl_params.get("global", {})

            # Convert to numpy
            np_params = {}
            for key, value in global_params.items():
                if isinstance(value, torch.Tensor):
                    np_params[key] = value.cpu().numpy()
                else:
                    np_params[key] = np.array(value)

            # Save to NPZ
            np.savez(output_path, **np_params)

            # Get info
            num_frames = np_params.get("body_pose", np.array([])).shape[0] if "body_pose" in np_params else 0
            file_size_kb = output_path.stat().st_size / 1024

            info = (
                f"SaveSMPL Complete\n"
                f"Output: {output_path}\n"
                f"Frames: {num_frames}\n"
                f"File size: {file_size_kb:.1f} KB\n"
                f"Parameters: {', '.join(np_params.keys())}\n"
            )

            Log.info(f"[SaveSMPL] Saved {num_frames} frames to {output_path}")
            return (str(output_path.absolute()), info)

        except Exception as e:
            error_msg = f"SaveSMPL failed: {str(e)}"
            Log.error(error_msg)
            import traceback
            traceback.print_exc()
            return ("", error_msg)


NODE_CLASS_MAPPINGS = {
    "SaveSMPL": SaveSMPL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveSMPL": "Save SMPL Motion",
}
