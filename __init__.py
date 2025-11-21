"""
ComfyUI-MotionCapture Custom Node Package

A ComfyUI node package for GVHMR-based motion capture from video.
Extracts 3D human motion and SMPL parameters from video with SAM3 segmentation.
"""

import sys
from pathlib import Path

# Add the custom nodes directory to Python path
node_path = Path(__file__).parent / "nodes"
vendor_path = Path(__file__).parent / "vendor"

sys.path.insert(0, str(node_path))
sys.path.insert(0, str(vendor_path))

# Import nodes
from .nodes.loader_node import LoadGVHMRModels
from .nodes.inference_node import GVHMRInference

# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "LoadGVHMRModels": LoadGVHMRModels,
    "GVHMRInference": GVHMRInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadGVHMRModels": "Load GVHMR Models",
    "GVHMRInference": "GVHMR Inference",
}

# Module info
__version__ = "0.1.0"
__author__ = "ComfyUI-MotionCapture"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"\n{'='*60}")
print(f"ComfyUI-MotionCapture v{__version__} loaded successfully!")
print(f"Nodes available:")
print(f"  - LoadGVHMRModels: Load GVHMR model pipeline")
print(f"  - GVHMRInference: Run motion capture inference")
print(f"{'='*60}\n")
