"""
Bone mapping configurations for retargeting SMPL motion to different rig types.
"""

# SMPL to VRoid bone mapping (22 joints → 22 bones, perfect 1:1 match)
SMPL_TO_VROID_MAPPING = {
    # Root & Spine
    "hips": "J_Bip_C_Hips",
    "spine": "J_Bip_C_Spine",
    "spine1": "J_Bip_C_Chest",
    "spine2": "J_Bip_C_UpperChest",

    # Head & Neck
    "neck": "J_Bip_C_Neck",
    "head": "J_Bip_C_Head",

    # Left Arm
    "leftShoulder": "J_Bip_L_Shoulder",
    "leftArm": "J_Bip_L_UpperArm",
    "leftForeArm": "J_Bip_L_LowerArm",
    "leftHand": "J_Bip_L_Hand",

    # Right Arm
    "rightShoulder": "J_Bip_R_Shoulder",
    "rightArm": "J_Bip_R_UpperArm",
    "rightForeArm": "J_Bip_R_LowerArm",
    "rightHand": "J_Bip_R_Hand",

    # Left Leg
    "leftUpLeg": "J_Bip_L_UpperLeg",
    "leftLeg": "J_Bip_L_LowerLeg",
    "leftFoot": "J_Bip_L_Foot",
    "leftToeBase": "J_Bip_L_ToeBase",

    # Right Leg
    "rightUpLeg": "J_Bip_R_UpperLeg",
    "rightLeg": "J_Bip_R_LowerLeg",
    "rightFoot": "J_Bip_R_Foot",
    "rightToeBase": "J_Bip_R_ToeBase",
}

# SMPL joint order (for reference)
SMPL_JOINTS = [
    "hips",          # 0
    "leftUpLeg",     # 1
    "rightUpLeg",    # 2
    "spine",         # 3
    "leftLeg",       # 4
    "rightLeg",      # 5
    "spine1",        # 6
    "leftFoot",      # 7
    "rightFoot",     # 8
    "spine2",        # 9
    "leftToeBase",   # 10
    "rightToeBase",  # 11
    "neck",          # 12
    "leftShoulder",  # 13
    "rightShoulder", # 14
    "head",          # 15
    "leftArm",       # 16
    "rightArm",      # 17
    "leftForeArm",   # 18
    "rightForeArm",  # 19
    "leftHand",      # 20
    "rightHand",     # 21
]

def get_bone_mapping(rig_type: str):
    """
    Get bone mapping for a specific rig type.

    Args:
        rig_type: Type of rig ("vroid", "mixamo", "rigify", etc.)

    Returns:
        Dictionary mapping SMPL joint names to target rig bone names
    """
    if rig_type == "vroid":
        return SMPL_TO_VROID_MAPPING
    elif rig_type == "mixamo":
        # TODO: Implement Mixamo mapping
        raise NotImplementedError("Mixamo mapping not yet implemented")
    elif rig_type == "rigify":
        # TODO: Implement Rigify mapping
        raise NotImplementedError("Rigify mapping not yet implemented")
    elif rig_type == "ue5_mannequin":
        # TODO: Implement UE5 Mannequin mapping
        raise NotImplementedError("UE5 Mannequin mapping not yet implemented")
    else:
        raise ValueError(f"Unknown rig type: {rig_type}")
