"""
SMPLRetargetToSMPL Node - Apply SMPL motion to a rigged SMPL skeleton FBX

This node takes SMPL motion data and applies it to a rigged FBX with SMPL skeleton
(typically from UniRig with SMPL template), producing an animated FBX.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from hmr4d.utils.pylogger import Log


class SMPLRetargetToSMPL:
    """
    Apply SMPL motion data to a rigged FBX with SMPL skeleton.

    This is different from SMPLToFBX which uses BVH+Rokoko for arbitrary rigs.
    This node directly applies SMPL rotations to an SMPL-skeleton FBX,
    preserving the exact motion without intermediate conversion.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "smpl_params": ("SMPL_PARAMS",),
                "rigged_fbx_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "output_filename": ("STRING", {
                    "default": "animated_character",
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("fbx_path", "frame_count")
    FUNCTION = "retarget"
    OUTPUT_NODE = True
    CATEGORY = "MotionCapture/Retarget"

    def retarget(
        self,
        smpl_params: Dict,
        rigged_fbx_path: str,
        output_filename: str,
    ) -> Tuple[str, int]:
        """
        Apply SMPL motion to a rigged SMPL skeleton FBX.

        Args:
            smpl_params: SMPL parameters from LoadSMPL or GVHMRInference
            rigged_fbx_path: Path to input rigged FBX with SMPL skeleton
            output_filename: Name for output animated FBX (without extension)

        Returns:
            Tuple of (output_fbx_path, frame_count)
        """
        try:
            Log.info("[SMPLRetargetToSMPL] Starting SMPL-to-SMPL retargeting...")

            # Validate input FBX
            rigged_fbx_path = Path(rigged_fbx_path)
            if not rigged_fbx_path.exists():
                raise FileNotFoundError(f"Input FBX not found: {rigged_fbx_path}")

            # Find Blender executable
            blender_exe = self._find_blender()
            if not blender_exe:
                raise RuntimeError(
                    "Blender not found. Please ensure Blender is installed in lib/blender/"
                )

            Log.info(f"[SMPLRetargetToSMPL] Using Blender: {blender_exe}")

            # Prepare output path
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            if not output_filename.endswith('.fbx'):
                output_filename = f"{output_filename}.fbx"
            output_path = output_dir / output_filename

            # Extract SMPL parameters to temporary NPZ
            temp_dir = Path(tempfile.gettempdir())
            motion_npz_path = temp_dir / "smpl_motion_temp.npz"
            frame_count = self._save_smpl_params(smpl_params, motion_npz_path)

            Log.info(f"[SMPLRetargetToSMPL] Saved motion data: {motion_npz_path} ({frame_count} frames)")

            # Get path to Blender script
            script_path = Path(__file__).parent.parent / "lib" / "blender_animate_smpl.py"
            if not script_path.exists():
                raise FileNotFoundError(f"Blender script not found: {script_path}")

            # Run Blender
            cmd = [
                str(blender_exe),
                "--background",
                "--python", str(script_path),
                "--",
                str(rigged_fbx_path.absolute()),
                str(motion_npz_path),
                str(output_path.absolute()),
            ]

            Log.info(f"[SMPLRetargetToSMPL] Running Blender animation...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for long animations
            )

            if result.returncode != 0:
                Log.error(f"[SMPLRetargetToSMPL] Blender error:\n{result.stderr}")
                raise RuntimeError(f"Blender animation failed: {result.stderr}")

            Log.info(f"[SMPLRetargetToSMPL] Blender output:\n{result.stdout}")

            if not output_path.exists():
                raise RuntimeError(f"Output FBX not created: {output_path}")

            Log.info(f"[SMPLRetargetToSMPL] Animation complete! Output: {output_path}")
            return (str(output_path.absolute()), frame_count)

        except Exception as e:
            error_msg = f"SMPLRetargetToSMPL failed: {str(e)}"
            Log.error(error_msg)
            import traceback
            traceback.print_exc()
            return ("", 0)

    def _find_blender(self) -> Path:
        """Find Blender executable."""
        # Check local installation first
        local_blender = Path(__file__).parent.parent / "lib" / "blender"

        if local_blender.exists():
            import platform

            system = platform.system().lower()
            if system == "windows":
                pattern = "**/blender.exe"
            elif system == "darwin":
                pattern = "**/MacOS/blender"
            else:
                pattern = "**/blender"

            executables = list(local_blender.glob(pattern))
            if executables:
                return executables[0]

        # Check system PATH
        import shutil
        system_blender = shutil.which("blender")
        if system_blender:
            return Path(system_blender)

        return None

    def _save_smpl_params(self, smpl_params: Dict, output_path: Path) -> int:
        """
        Save SMPL parameters to NPZ file for Blender.

        Returns:
            Number of frames in the motion
        """
        # Extract global parameters
        global_params = smpl_params.get("global", smpl_params)

        # Convert to numpy
        np_params = {}
        for key, value in global_params.items():
            if isinstance(value, torch.Tensor):
                np_params[key] = value.cpu().numpy()
            elif isinstance(value, np.ndarray):
                np_params[key] = value
            else:
                np_params[key] = np.array(value)

        # Determine frame count
        frame_count = 0
        if 'body_pose' in np_params:
            frame_count = np_params['body_pose'].shape[0]
        elif 'global_orient' in np_params:
            frame_count = np_params['global_orient'].shape[0]

        np.savez(output_path, **np_params)
        Log.info(f"[SMPLRetargetToSMPL] Saved SMPL params: {list(np_params.keys())}")

        return frame_count


NODE_CLASS_MAPPINGS = {
    "SMPLRetargetToSMPL": SMPLRetargetToSMPL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SMPLRetargetToSMPL": "SMPL to SMPL Retargeting",
}
