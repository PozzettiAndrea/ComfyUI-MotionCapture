
import logging
import os
from pathlib import Path
from typing import Tuple, Dict, List

from comfy_api.latest import io

log = logging.getLogger("motioncapture")

class LoadBVHFromFolder(io.ComfyNode):
    """
    Load a BVH file from a specific folder (input or output) using a dropdown menu.
    """

    @classmethod
    def define_schema(cls):
        # Scan for files
        input_dir = Path("input")
        output_dir = Path("output")

        files = []

        if input_dir.exists():
            files.extend([f"input/{f.name}" for f in input_dir.glob("*.bvh")])

        if output_dir.exists():
            # sort by modification time (newest first) to easily find latest tests
            out_files = sorted(list(output_dir.glob("*.bvh")), key=os.path.getmtime, reverse=True)
            files.extend([f"output/{f.name}" for f in out_files])

        if not files:
            files = ["None"]

        return io.Schema(
            node_id="LoadBVHFromFolder",
            display_name="Load BVH (Dropdown)",
            category="MotionCapture/BVH",
            inputs=[
                io.Combo.Input("bvh_file", options=files),
            ],
            outputs=[
                io.Custom("BVH_DATA").Output(display_name="bvh_data"),
            ],
        )

    @classmethod
    def execute(cls, bvh_file: str) -> io.NodeOutput:
        if bvh_file == "None":
            return io.NodeOutput({})

        # Handle paths relative to ComfyUI root
        file_path = Path(bvh_file)

        if not file_path.exists():
            raise ValueError(f"BVH file not found: {file_path}")

        with open(file_path, 'r') as f:
            content = f.read()

        # Parse basic info from content to populate bvh_data
        # Simple parsing to get frame count and FPS
        num_frames = 0
        frame_time = 0.033333

        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith("Frames:"):
                try:
                    num_frames = int(line.split(":")[1].strip())
                except Exception as e:
                    log.debug("Failed to parse BVH frame count: %s", e)
            elif line.startswith("Frame Time:"):
                try:
                    frame_time = float(line.split(":")[1].strip())
                except Exception as e:
                    log.debug("Failed to parse BVH frame time: %s", e)

        fps = int(round(1.0 / frame_time)) if frame_time > 0 else 30

        bvh_data = {
            "file_path": str(file_path.absolute()),
            "num_frames": num_frames,
            "fps": fps,
            "content": content # Store raw content if needed
        }

        return io.NodeOutput(bvh_data)

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan") # Always update to allow file refresh logic if needed

NODE_CLASS_MAPPINGS = {
    "LoadBVHFromFolder": LoadBVHFromFolder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadBVHFromFolder": "Load BVH (Dropdown)",
}
