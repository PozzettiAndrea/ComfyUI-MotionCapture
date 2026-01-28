
import os
from pathlib import Path
from typing import Tuple, Dict, List

class LoadBVHFromFolder:
    """
    Load a BVH file from a specific folder (input or output) using a dropdown menu.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
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

        return {
            "required": {
                "bvh_file": (files,),
            },
        }

    RETURN_TYPES = ("BVH_DATA",)
    RETURN_NAMES = ("bvh_data",)
    FUNCTION = "load_bvh"
    CATEGORY = "MotionCapture/BVH"

    def load_bvh(self, bvh_file: str) -> Tuple[Dict]:
        if bvh_file == "None":
            return ({},)
            
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
                except: pass
            elif line.startswith("Frame Time:"):
                try:
                    frame_time = float(line.split(":")[1].strip())
                except: pass
                
        fps = int(round(1.0 / frame_time)) if frame_time > 0 else 30
        
        bvh_data = {
            "file_path": str(file_path.absolute()),
            "num_frames": num_frames,
            "fps": fps,
            "content": content # Store raw content if needed
        }
        
        return (bvh_data,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan") # Always update to allow file refresh logic if needed

NODE_CLASS_MAPPINGS = {
    "LoadBVHFromFolder": LoadBVHFromFolder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadBVHFromFolder": "Load BVH (Dropdown)",
}
