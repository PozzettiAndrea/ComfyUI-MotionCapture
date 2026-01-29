# GPU nodes - CUDA/vendor dependencies
import sys
from pathlib import Path

# Add vendor path for GVHMR, DPVO, SAM3D
VENDOR_PATH = Path(__file__).parent / "vendor"
sys.path.insert(0, str(VENDOR_PATH))

# Import nodes
from .loader_node import LoadGVHMRModels
from .dpvo_loader_node import LoadDPVOModel
from .inference_node import GVHMRInference
from .viewer_node import SMPLViewer
from .save_smpl_node import SaveSMPL
from .load_smpl_node import LoadSMPL
from .fbx_loader_node import LoadFBXCharacter
from .fbx_preview_node import FBXPreview
from .fbx_animation_viewer_node import FBXAnimationViewer
from .smpl_to_bvh_node import SMPLtoBVH
from .bvh_viewer_node import BVHViewer
from .compare_smpl_bvh_node import CompareSMPLtoBVH
from .bvh_loader_node import LoadBVHFromFolder
from .mixamo_loader_node import LoadMixamoCharacter
from .compare_skeletons_node import CompareSkeletons
from .sam3d_loader_node import LoadSAM3DBodyModels
from .sam3d_inference_node import SAM3DVideoInference
from .mhr_viewer_node import MHRViewer
from .save_mhr_node import SaveMHR

NODE_CLASS_MAPPINGS = {
    "LoadGVHMRModels": LoadGVHMRModels,
    "LoadDPVOModel": LoadDPVOModel,
    "GVHMRInference": GVHMRInference,
    "SMPLViewer": SMPLViewer,
    "SaveSMPL": SaveSMPL,
    "LoadSMPL": LoadSMPL,
    "LoadFBXCharacter": LoadFBXCharacter,
    "FBXPreview": FBXPreview,
    "FBXAnimationViewer": FBXAnimationViewer,
    "SMPLtoBVH": SMPLtoBVH,
    "BVHViewer": BVHViewer,
    "CompareSMPLtoBVH": CompareSMPLtoBVH,
    "LoadBVHFromFolder": LoadBVHFromFolder,
    "LoadMixamoCharacter": LoadMixamoCharacter,
    "CompareSkeletons": CompareSkeletons,
    "LoadSAM3DBodyModels": LoadSAM3DBodyModels,
    "SAM3DVideoInference": SAM3DVideoInference,
    "MHRViewer": MHRViewer,
    "SaveMHR": SaveMHR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadGVHMRModels": "Load GVHMR Models",
    "LoadDPVOModel": "Load DPVO Model",
    "GVHMRInference": "GVHMR Inference",
    "SMPLViewer": "SMPL 3D Viewer",
    "SaveSMPL": "Save SMPL Motion",
    "LoadSMPL": "Load SMPL Motion",
    "LoadFBXCharacter": "Load FBX Character",
    "FBXPreview": "FBX 3D Preview",
    "FBXAnimationViewer": "FBX Animation Viewer",
    "SMPLtoBVH": "SMPL to BVH Converter",
    "BVHViewer": "BVH Animation Viewer",
    "CompareSMPLtoBVH": "Compare SMPL vs BVH",
    "LoadBVHFromFolder": "Load BVH (Dropdown)",
    "LoadMixamoCharacter": "Load Mixamo Character",
    "CompareSkeletons": "Compare Skeletons",
    "LoadSAM3DBodyModels": "Load SAM 3D Body Models",
    "SAM3DVideoInference": "SAM3D Video Inference",
    "MHRViewer": "MHR Skeleton Viewer",
    "SaveMHR": "Save MHR Motion",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
