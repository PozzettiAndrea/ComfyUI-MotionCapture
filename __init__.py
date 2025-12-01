"""
ComfyUI-MotionCapture Custom Node Package

A ComfyUI node package for motion capture from video.
Supports both GVHMR (SMPL output) and SAM 3D Body (MHR output) backends.
Extracts 3D human motion from video with SAM3 segmentation.
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
from .nodes.fbx_animation_viewer_node import FBXAnimationViewer
from .nodes.smpl_to_bvh_node import SMPLtoBVH
from .nodes.bvh_viewer_node import BVHViewer
from .nodes.bvh_retarget_node import BVHtoFBX
from .nodes.compare_smpl_bvh_node import CompareSMPLtoBVH
from .nodes.bvh_loader_node import LoadBVHFromFolder
from .nodes.smpl_retarget_node import SMPLRetargetToSMPL

# SAM 3D Body nodes (MHR 70-keypoint skeleton)
from .nodes.sam3d_loader_node import LoadSAM3DBodyModels
from .nodes.sam3d_inference_node import SAM3DVideoInference
from .nodes.mhr_viewer_node import MHRViewer
from .nodes.save_mhr_node import SaveMHR

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
    "FBXAnimationViewer": FBXAnimationViewer,
    "SMPLtoBVH": SMPLtoBVH,
    "BVHViewer": BVHViewer,
    "BVHtoFBX": BVHtoFBX,
    "CompareSMPLtoBVH": CompareSMPLtoBVH,
    "LoadBVHFromFolder": LoadBVHFromFolder,
    "SMPLRetargetToSMPL": SMPLRetargetToSMPL,
    # SAM 3D Body nodes
    "LoadSAM3DBodyModels": LoadSAM3DBodyModels,
    "SAM3DVideoInference": SAM3DVideoInference,
    "MHRViewer": MHRViewer,
    "SaveMHR": SaveMHR,
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
    "FBXAnimationViewer": "FBX Animation Viewer",
    "SMPLtoBVH": "SMPL to BVH Converter",
    "BVHViewer": "BVH Animation Viewer",
    "BVHtoFBX": "BVH to FBX Retargeter",
    "CompareSMPLtoBVH": "Compare SMPL vs BVH",
    "LoadBVHFromFolder": "Load BVH (Dropdown)",
    "SMPLRetargetToSMPL": "SMPL to SMPL Retargeting",
    # SAM 3D Body nodes
    "LoadSAM3DBodyModels": "Load SAM 3D Body Models",
    "SAM3DVideoInference": "SAM3D Video Inference",
    "MHRViewer": "MHR Skeleton Viewer",
    "SaveMHR": "Save MHR Motion",
}

# Module info
__version__ = "0.1.0"
__author__ = "ComfyUI-MotionCapture"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print(f"\n{'='*60}")
print(f"ComfyUI-MotionCapture v{__version__} loaded successfully!")
print(f"")
print(f"GVHMR Nodes (SMPL 24-joint output):")
print(f"  - LoadGVHMRModels: Load GVHMR model pipeline")
print(f"  - GVHMRInference: Run GVHMR motion capture inference")
print(f"")
print(f"SAM 3D Body Nodes (MHR 70-keypoint output with hands):")
print(f"  - LoadSAM3DBodyModels: Load SAM 3D Body model")
print(f"  - SAM3DVideoInference: Run SAM3D inference with temporal smoothing")
print(f"  - MHRViewer: Interactive 3D skeleton viewer for MHR output")
print(f"  - SaveMHR: Save MHR motion data to disk")
print(f"")
print(f"Common Nodes:")
print(f"  - SMPLViewer: Interactive 3D viewer for SMPL meshes")
print(f"  - SaveSMPL: Save SMPL motion data to disk")
print(f"  - LoadSMPL: Load SMPL motion data from disk")
print(f"  - SMPLToFBX: Retarget SMPL motion to FBX characters")
print(f"  - LoadFBXCharacter: Load FBX files with folder browser")
print(f"  - FBXPreview: Interactive 3D FBX viewer")
print(f"  - FBXAnimationViewer: Animated FBX playback viewer")
print(f"  - SMPLtoBVH: Convert SMPL motion to BVH format")
print(f"  - BVHViewer: Interactive 3D viewer for BVH animations")
print(f"  - BVHtoFBX: Retarget BVH motion to FBX/VRM characters")
print(f"  - CompareSMPLtoBVH: Side-by-side comparison of SMPL and BVH animations")
print(f"  - SMPLRetargetToSMPL: Apply SMPL motion to SMPL-rigged FBX")
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

    @PromptServer.instance.routes.get('/motioncapture/npz_files')
    async def get_npz_files(request):
        """API endpoint to fetch NPZ file list dynamically."""
        source = request.query.get('source_folder', 'output')

        try:
            if source == "input":
                files = LoadSMPL.get_npz_files_from_input()
            else:
                files = LoadSMPL.get_npz_files_from_output()

            return web.json_response(files)
        except Exception as e:
            print(f"[MotionCapture API] Error getting NPZ files: {e}")
            return web.json_response([])

    print("[MotionCapture] API endpoint registered: /motioncapture/npz_files")

    @PromptServer.instance.routes.get('/motioncapture/smpl_mesh')
    async def get_smpl_mesh_file(request):
        """API endpoint to fetch SMPL mesh binary file."""
        filename = request.query.get('filename', None)
        if not filename:
            raise web.HTTPBadRequest(reason="Missing filename parameter")

        filepath = Path("output") / filename
        if not filepath.is_file():
            raise web.HTTPNotFound(reason=f"File not found: {filename}")

        return web.FileResponse(filepath)

    print("[MotionCapture] API endpoint registered: /motioncapture/smpl_mesh")

except Exception as e:
    print(f"[MotionCapture] Warning: Could not register API endpoints: {e}")
    print("[MotionCapture] FBX file browsing will not work without PromptServer")
