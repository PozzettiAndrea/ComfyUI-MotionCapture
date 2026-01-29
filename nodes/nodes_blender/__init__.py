# Blender nodes - bpy dependencies

from .retarget_node import SMPLToFBX
from .bvh_retarget_node import BVHtoFBX
from .smpl_retarget_node import SMPLRetargetToSMPL
from .smpl_to_mixamo_node import SMPLToMixamo
from .rest_pose_node import ExtractRestPose

NODE_CLASS_MAPPINGS = {
    "SMPLToFBX": SMPLToFBX,
    "BVHtoFBX": BVHtoFBX,
    "SMPLRetargetToSMPL": SMPLRetargetToSMPL,
    "SMPLToMixamo": SMPLToMixamo,
    "ExtractRestPose": ExtractRestPose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SMPLToFBX": "SMPL to FBX Retargeting",
    "BVHtoFBX": "BVH to FBX Retargeter",
    "SMPLRetargetToSMPL": "SMPL to SMPL Retargeting",
    "SMPLToMixamo": "SMPL to Mixamo",
    "ExtractRestPose": "Extract Rest Pose",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
