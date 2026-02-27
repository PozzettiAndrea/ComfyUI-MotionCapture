"""
SaveSMPL Node - Save SMPL motion data to disk for reuse
"""

from pathlib import Path
from typing import Dict, Tuple
import torch
import numpy as np

from comfy_api.latest import io

from .motion_utils.pylogger import Log


class SaveSMPL(io.ComfyNode):
    """
    Save SMPL motion parameters to .npz file for caching and reuse.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveSMPL",
            display_name="Save SMPL Motion",
            category="MotionCapture/SMPL",
            is_output_node=True,
            inputs=[
                io.Custom("SMPL_PARAMS").Input("smpl_params"),
                io.String.Input("output_path", default="output/motion.npz", multiline=False),
            ],
            outputs=[
                io.String.Output(display_name="file_path"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        smpl_params: Dict,
        output_path: str,
    ) -> io.NodeOutput:
        """
        Save SMPL parameters to NPZ file.

        Args:
            smpl_params: SMPL parameters from GVHMRInference
            output_path: Path to save NPZ file

        Returns:
            NodeOutput with (file_path, info_string)
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
            return io.NodeOutput(str(output_path.absolute()), info)

        except Exception as e:
            error_msg = f"SaveSMPL failed: {str(e)}"
            Log.error(error_msg, exc_info=True)
            return io.NodeOutput("", error_msg)


NODE_CLASS_MAPPINGS = {
    "SaveSMPL": SaveSMPL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveSMPL": "Save SMPL Motion",
}
