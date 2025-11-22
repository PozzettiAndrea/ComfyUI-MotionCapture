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
from .nodes.viewer_node import SMPLViewer
from .nodes.save_smpl_node import SaveSMPL
from .nodes.load_smpl_node import LoadSMPL
from .nodes.retarget_node import SMPLToFBX
from .nodes.fbx_loader_node import LoadFBXCharacter
from .nodes.fbx_preview_node import FBXPreview

# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "LoadGVHMRModels": LoadGVHMRModels,
    "GVHMRInference": GVHMRInference,
    "SMPLViewer": SMPLViewer,
    "SaveSMPL": SaveSMPL,
    "LoadSMPL": LoadSMPL,
    "SMPLToFBX": SMPLToFBX,
    "LoadFBXCharacter": LoadFBXCharacter,
    "FBXPreview": FBXPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadGVHMRModels": "Load GVHMR Models",
    "GVHMRInference": "GVHMR Inference",
    "SMPLViewer": "SMPL 3D Viewer",
    "SaveSMPL": "Save SMPL Motion",
    "LoadSMPL": "Load SMPL Motion",
    "SMPLToFBX": "SMPL to FBX Retargeting",
    "LoadFBXCharacter": "Load FBX Character",
    "FBXPreview": "FBX 3D Preview",
}

# Module info
__version__ = "0.1.0"
__author__ = "ComfyUI-MotionCapture"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print(f"\n{'='*60}")
print(f"ComfyUI-MotionCapture v{__version__} loaded successfully!")
print(f"Nodes available:")
print(f"  - LoadGVHMRModels: Load GVHMR model pipeline")
print(f"  - GVHMRInference: Run motion capture inference")
print(f"  - SMPLViewer: Interactive 3D viewer for SMPL meshes")
print(f"  - SaveSMPL: Save SMPL motion data to disk")
print(f"  - LoadSMPL: Load SMPL motion data from disk")
print(f"  - SMPLToFBX: Retarget SMPL motion to FBX characters")
print(f"  - LoadFBXCharacter: Load FBX files with folder browser")
print(f"  - FBXPreview: Interactive 3D FBX viewer")
print(f"{'='*60}\n")

# Web extensions path for ComfyUI
WEB_DIRECTORY = "./web"

# Register API endpoints for dynamic file loading
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get('/motioncapture/fbx_files')
    async def get_fbx_files(request):
        """API endpoint to fetch FBX file list dynamically."""
        source = request.query.get('source_folder', 'output')

        try:
            if source == "input":
                files = LoadFBXCharacter.get_fbx_files_from_input()
            else:
                files = LoadFBXCharacter.get_fbx_files_from_output()

            return web.json_response(files)
        except Exception as e:
            print(f"[MotionCapture API] Error getting FBX files: {e}")
            return web.json_response([])

    print("[MotionCapture] API endpoint registered: /motioncapture/fbx_files")

except Exception as e:
    print(f"[MotionCapture] Warning: Could not register API endpoints: {e}")
    print("[MotionCapture] FBX file browsing will not work without PromptServer")
