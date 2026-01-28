"""
SaveMHR Node - Save MHR motion data to disk for reuse
"""

from pathlib import Path
from typing import Dict, Tuple
import torch
import numpy as np

import sys
VENDOR_PATH = Path(__file__).parent / "vendor"
sys.path.insert(0, str(VENDOR_PATH))

from hmr4d.utils.pylogger import Log


class SaveMHR:
    """
    Save MHR motion parameters to .npz file for caching and reuse.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mhr_params": ("MHR_PARAMS",),
                "output_path": ("STRING", {
                    "default": "output/mhr_motion.npz",
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_path", "info")
    FUNCTION = "save_mhr"
    OUTPUT_NODE = True
    CATEGORY = "MotionCapture/SAM3D"

    def save_mhr(
        self,
        mhr_params: Dict,
        output_path: str,
    ) -> Tuple[str, str]:
        """
        Save MHR parameters to NPZ file.

        Args:
            mhr_params: MHR parameters from SAM3DVideoInference
            output_path: Path to save NPZ file

        Returns:
            Tuple of (file_path, info_string)
        """
        try:
            Log.info("[SaveMHR] Saving MHR motion data...")

            # Prepare output directory
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Ensure .npz extension
            if not output_path.suffix == '.npz':
                output_path = output_path.with_suffix('.npz')

            # Convert all parameters to numpy
            np_params = {}
            for key, value in mhr_params.items():
                if isinstance(value, torch.Tensor):
                    np_params[key] = value.cpu().numpy()
                elif isinstance(value, (int, float, str)):
                    np_params[key] = np.array(value)
                elif value is not None:
                    np_params[key] = np.array(value)

            # Save to NPZ
            np.savez(output_path, **np_params)

            # Get info
            num_frames = mhr_params.get("num_frames", 0)
            keypoints_shape = "N/A"
            if "keypoints_3d" in mhr_params:
                kp = mhr_params["keypoints_3d"]
                if isinstance(kp, torch.Tensor):
                    keypoints_shape = str(tuple(kp.shape))
                else:
                    keypoints_shape = str(np.array(kp).shape)

            file_size_kb = output_path.stat().st_size / 1024

            info = (
                f"SaveMHR Complete\n"
                f"Output: {output_path}\n"
                f"Frames: {num_frames}\n"
                f"Keypoints shape: {keypoints_shape}\n"
                f"File size: {file_size_kb:.1f} KB\n"
                f"Parameters: {', '.join(np_params.keys())}\n"
            )

            Log.info(f"[SaveMHR] Saved {num_frames} frames to {output_path}")
            return (str(output_path.absolute()), info)

        except Exception as e:
            error_msg = f"SaveMHR failed: {str(e)}"
            Log.error(error_msg)
            import traceback
            traceback.print_exc()
            return ("", error_msg)


NODE_CLASS_MAPPINGS = {
    "SaveMHR": SaveMHR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveMHR": "Save MHR Motion",
}
