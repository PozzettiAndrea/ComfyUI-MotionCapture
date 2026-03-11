"""
BVHtoFBX Node - Retarget BVH motion to rigged FBX/VRM characters using bpy

Blender operations run in an isolated environment with the bpy package.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import folder_paths

from .motion_utils.pylogger import Log

log = logging.getLogger("motioncapture")


# ===============================================================================
# BONE MAPPING
# ===============================================================================

# SMPL (Source) -> VRM/Mixamo (Target) bone mapping
BONE_MAP = {
    'Pelvis': 'Hips',
    'L_Hip': 'LeftUpperLeg',
    'R_Hip': 'RightUpperLeg',
    'Spine1': 'Spine',
    'L_Knee': 'LeftLowerLeg',
    'R_Knee': 'RightLowerLeg',
    'Spine2': 'Chest',
    'L_Ankle': 'LeftFoot',
    'R_Ankle': 'RightFoot',
    'Spine3': 'UpperChest',
    'L_Foot': 'LeftToes',
    'R_Foot': 'RightToes',
    'Neck': 'Neck',
    'L_Collar': 'LeftShoulder',
    'R_Collar': 'RightShoulder',
    'Head': 'Head',
    'L_Shoulder': 'LeftUpperArm',
    'R_Shoulder': 'RightUpperArm',
    'L_Elbow': 'LeftLowerArm',
    'R_Elbow': 'RightLowerArm',
    'L_Wrist': 'LeftHand',
    'R_Wrist': 'RightHand',
    'L_Hand': 'LeftHand',
    'R_Hand': 'RightHand'
}

# VRoid-specific bone naming (J_Bip_...)
VROID_BONE_MAP = {
    'Hips': 'J_Bip_C_Hips',
    'Spine': 'J_Bip_C_Spine',
    'Chest': 'J_Bip_C_Chest',
    'UpperChest': 'J_Bip_C_UpperChest',
    'Neck': 'J_Bip_C_Neck',
    'Head': 'J_Bip_C_Head',
    'LeftShoulder': 'J_Bip_L_Shoulder',
    'LeftUpperArm': 'J_Bip_L_UpperArm',
    'LeftLowerArm': 'J_Bip_L_LowerArm',
    'LeftHand': 'J_Bip_L_Hand',
    'RightShoulder': 'J_Bip_R_Shoulder',
    'RightUpperArm': 'J_Bip_R_UpperArm',
    'RightLowerArm': 'J_Bip_R_LowerArm',
    'RightHand': 'J_Bip_R_Hand',
    'LeftUpperLeg': 'J_Bip_L_UpperLeg',
    'LeftLowerLeg': 'J_Bip_L_LowerLeg',
    'LeftFoot': 'J_Bip_L_Foot',
    'LeftToes': 'J_Bip_L_ToeBase',
    'RightUpperLeg': 'J_Bip_R_UpperLeg',
    'RightLowerLeg': 'J_Bip_R_LowerLeg',
    'RightFoot': 'J_Bip_R_Foot',
    'RightToes': 'J_Bip_R_ToeBase',
}


# ===============================================================================
# ISOLATED BLENDER WORKER
# ===============================================================================

class BVHtoFBXWorker:
    """
    Isolated worker for BVH to FBX retargeting using bpy.
    Runs in the mocap isolated environment with bpy package.
    """

    FUNCTION = "retarget"

    def retarget(
        self,
        bvh_file: str,
        character_path: str,
        output_path: str,
        character_type: str,
        output_format: str,
    ) -> Tuple[str, str]:
        """
        Retarget BVH motion to FBX/VRM character using bpy.

        Args:
            bvh_file: Path to input BVH file
            character_path: Path to input character FBX/VRM
            output_path: Path to output file
            character_type: 'vrm' or 'fbx'
            output_format: 'vrm' or 'fbx'

        Returns:
            Tuple of (output_path, info_string)
        """
        import bpy
        import sys
        from mathutils import Matrix, Quaternion, Vector
        from math import radians, degrees

        log.info("=" * 60)
        log.info("BVH to FBX Retargeting")
        log.info("=" * 60)
        log.info("BVH: %s", bvh_file)
        log.info("Character: %s", character_path)
        log.info("Output: %s", output_path)

        # Clear scene
        bpy.ops.wm.read_homefile(use_empty=True)
        log.info("Cleared scene")

        # Import character
        if character_type == "vrm":
            log.info("Importing VRM character...")
            try:
                bpy.ops.import_scene.vrm(filepath=character_path)
                log.info("VRM import successful")
            except AttributeError:
                try:
                    bpy.ops.import_model.vrm(filepath=character_path)
                    log.info("VRM import successful (legacy command)")
                except Exception:
                    raise RuntimeError("VRM addon not found. Please install VRM Addon for Blender.")
        else:
            log.info("Importing FBX character...")
            bpy.ops.import_scene.fbx(filepath=character_path)
            log.info("FBX import successful")

        # Find character armature
        char_armature = None
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE':
                char_armature = obj
                break

        if not char_armature:
            raise RuntimeError("No armature found in character file")

        log.info("Found character armature: %s", char_armature.name)
        log.debug("Character Armature Bones: %s", [b.name for b in char_armature.data.bones])

        # Ensure we are in Object Mode
        if bpy.context.object:
            bpy.ops.object.mode_set(mode='OBJECT')

        # Load BVH
        log.info("Loading BVH animation: %s", bvh_file)
        bpy.ops.import_anim.bvh(filepath=bvh_file, global_scale=1.0)

        # Find BVH armature
        bvh_armature = None
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE' and obj != char_armature:
                bvh_armature = obj
                break

        if not bvh_armature:
            raise RuntimeError("BVH armature not found after import")

        log.info("Found BVH armature: %s", bvh_armature.name)
        log.debug("BVH Armature Bones: %s", [b.name for b in bvh_armature.data.bones])

        # Auto-detect bone naming convention
        bone_names = char_armature.pose.bones.keys()
        is_vroid = any("J_Bip_C_Hips" in b for b in bone_names)

        bone_map = BONE_MAP.copy()
        if is_vroid:
            log.info("Detected VRoid bone naming convention")
            new_map = {}
            for smpl, vrm in bone_map.items():
                if vrm in VROID_BONE_MAP:
                    new_map[smpl] = VROID_BONE_MAP[vrm]
                else:
                    new_map[smpl] = vrm
            bone_map = new_map

        # Set up bone mapping
        log.info("Setting up bone mapping...")
        bpy.context.view_layer.objects.active = char_armature
        bpy.ops.object.mode_set(mode='POSE')

        valid_mappings = []
        for smpl_bone, vrm_bone in bone_map.items():
            if vrm_bone not in char_armature.pose.bones:
                log.warning("Target bone '%s' not found", vrm_bone)
                continue
            if smpl_bone not in bvh_armature.pose.bones:
                log.warning("Source bone '%s' not found", smpl_bone)
                continue
            valid_mappings.append((smpl_bone, vrm_bone))
            log.debug("Mapping: '%s' -> '%s'", smpl_bone, vrm_bone)

        log.info("Total valid bone mappings: %d", len(valid_mappings))

        if len(valid_mappings) == 0:
            raise RuntimeError("No valid bone mappings found")

        # Calculate scale ratio between skeletons
        def get_skeleton_height(armature, hips_name, head_name):
            if hips_name in armature.data.bones and head_name in armature.data.bones:
                hips = armature.data.bones[hips_name]
                head = armature.data.bones[head_name]
                return (head.head_local - hips.head_local).length
            return 1.0

        bvh_height = get_skeleton_height(bvh_armature, 'Pelvis', 'Head')
        if is_vroid:
            target_height = get_skeleton_height(char_armature, 'J_Bip_C_Hips', 'J_Bip_C_Head')
        else:
            target_height = get_skeleton_height(char_armature, 'Hips', 'Head')

        scale_ratio = target_height / bvh_height if bvh_height > 0.01 else 1.0
        log.info("Scale ratio: %.3f", scale_ratio)

        # Scale BVH armature
        bvh_armature.scale = (scale_ratio, scale_ratio, scale_ratio)
        bpy.context.view_layer.update()

        # Apply constraints
        log.info("Applying animation via constraints...")
        constraints_applied = 0

        for smpl_bone, vrm_bone in valid_mappings:
            p_bone = char_armature.pose.bones[vrm_bone]

            # LOCAL space rotation
            const = p_bone.constraints.new('COPY_ROTATION')
            const.target = bvh_armature
            const.subtarget = smpl_bone
            const.mix_mode = 'REPLACE'
            const.owner_space = 'LOCAL'
            const.target_space = 'LOCAL'
            constraints_applied += 1

            # WORLD space location for root
            if smpl_bone == 'Pelvis':
                loc_const = p_bone.constraints.new('COPY_LOCATION')
                loc_const.target = bvh_armature
                loc_const.subtarget = smpl_bone
                loc_const.owner_space = 'WORLD'
                loc_const.target_space = 'WORLD'

        log.info("Applied %d constraints", constraints_applied)

        # Bake animation
        log.info("Baking animation...")
        action = bvh_armature.animation_data.action
        frame_start = int(action.frame_range[0])
        frame_end = int(action.frame_range[1])
        log.info("Animation frames: %d to %d", frame_start, frame_end)

        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.nla.bake(
            frame_start=frame_start,
            frame_end=frame_end,
            only_selected=True,
            visual_keying=True,
            clear_constraints=True,
            use_current_action=False,
            bake_types={'POSE'}
        )
        log.info("Baking complete")

        # Calculate foot height compensation
        bvh_l_foot = 'L_Ankle'
        bvh_r_foot = 'R_Ankle'
        if is_vroid:
            target_l_foot = 'J_Bip_L_Foot'
            target_r_foot = 'J_Bip_R_Foot'
        else:
            target_l_foot = 'LeftFoot'
            target_r_foot = 'RightFoot'

        bvh_min_foot_z = float('inf')
        target_min_foot_z = float('inf')

        sample_frames = list(range(frame_start, frame_end + 1, max(1, (frame_end - frame_start) // 20)))
        for frame in sample_frames:
            bpy.context.scene.frame_set(frame)

            if bvh_l_foot in bvh_armature.pose.bones:
                bvh_l_z = (bvh_armature.matrix_world @ bvh_armature.pose.bones[bvh_l_foot].matrix).translation.z
                bvh_min_foot_z = min(bvh_min_foot_z, bvh_l_z)
            if bvh_r_foot in bvh_armature.pose.bones:
                bvh_r_z = (bvh_armature.matrix_world @ bvh_armature.pose.bones[bvh_r_foot].matrix).translation.z
                bvh_min_foot_z = min(bvh_min_foot_z, bvh_r_z)

            if target_l_foot in char_armature.pose.bones:
                target_l_z = (char_armature.matrix_world @ char_armature.pose.bones[target_l_foot].matrix).translation.z
                target_min_foot_z = min(target_min_foot_z, target_l_z)
            if target_r_foot in char_armature.pose.bones:
                target_r_z = (char_armature.matrix_world @ char_armature.pose.bones[target_r_foot].matrix).translation.z
                target_min_foot_z = min(target_min_foot_z, target_r_z)

        foot_height_offset = target_min_foot_z - bvh_min_foot_z
        if abs(foot_height_offset) > 0.05:
            log.info("Applying height compensation: %.3fm", foot_height_offset)
            char_armature.location.z -= foot_height_offset
            bpy.context.view_layer.update()

        # Delete BVH armature
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.data.objects.remove(bvh_armature, do_unlink=True)

        # Export
        bpy.ops.object.select_all(action='DESELECT')
        char_armature.select_set(True)
        bpy.context.view_layer.objects.active = char_armature

        # Select meshes
        mesh_count = 0
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                if obj.parent == char_armature:
                    obj.select_set(True)
                    mesh_count += 1
                else:
                    for mod in obj.modifiers:
                        if mod.type == 'ARMATURE' and mod.object == char_armature:
                            obj.select_set(True)
                            mesh_count += 1
                            break

        log.info("Selected %d meshes for export", mesh_count)

        if output_format == "vrm":
            log.info("Exporting as VRM...")
            try:
                bpy.ops.export_scene.vrm(filepath=output_path, export_fbx_hdr_emb=False)
            except AttributeError:
                output_path = output_path.replace(".vrm", ".fbx")
                bpy.ops.export_scene.fbx(filepath=output_path, use_selection=True, bake_anim=True, add_leaf_bones=False)
        else:
            log.info("Exporting as FBX...")
            bpy.ops.export_scene.fbx(
                filepath=output_path,
                use_selection=True,
                bake_anim=True,
                add_leaf_bones=False,
                path_mode='COPY',
                embed_textures=True,
            )

        num_frames = frame_end - frame_start + 1
        info = (
            f"BVH Retargeting Complete\n"
            f"Output: {output_path}\n"
            f"Frames: {num_frames}\n"
            f"Scale ratio: {scale_ratio:.3f}"
        )

        log.info("Output saved to: %s", output_path)
        log.info("Retargeting complete!")

        return (output_path, info)


# ===============================================================================
# COMFYUI NODE
# ===============================================================================

class BVHtoFBX:
    """
    Retarget BVH motion data to a rigged FBX/VRM character using bpy.
    Runs in an isolated environment with the bpy package.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bvh_data": ("BVH_DATA",),
                "character_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "output_path": ("STRING", {
                    "default": "retargeted.fbx",
                    "multiline": False,
                }),
            },
            "optional": {
                "character_type": (["auto", "vrm", "fbx"],),
                "output_format": (["fbx", "vrm"],),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "info")
    FUNCTION = "retarget"
    OUTPUT_NODE = True
    CATEGORY = "MotionCapture/BVH"

    def retarget(
        self,
        bvh_data: Dict,
        character_path: str,
        output_path: str,
        character_type: str = "auto",
        output_format: str = "fbx",
    ) -> Tuple[str, str]:
        try:
            log.info("Starting BVH retargeting...")

            # Validate inputs
            if not character_path:
                raise ValueError("Character path is empty. Please select a VRM or FBX file.")

            character_path = Path(character_path)
            if not character_path.exists():
                raise FileNotFoundError(f"Character file not found: {character_path}")

            bvh_file = bvh_data.get("file_path", "")
            if not bvh_file or not Path(bvh_file).exists():
                raise FileNotFoundError(f"BVH file not found: {bvh_file}")

            # Auto-detect character type
            if character_type == "auto":
                character_type = "vrm" if character_path.suffix.lower() == ".vrm" else "fbx"

            Log.info(f"[BVHtoFBX] Character type: {character_type}")

            # Prepare output directory (use ComfyUI output folder)
            output_path = Path(output_path)
            if not output_path.is_absolute():
                output_dir = Path(folder_paths.get_output_directory())
                output_path = output_dir / output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Ensure correct extension
            if output_format == "vrm" and output_path.suffix.lower() != ".vrm":
                output_path = output_path.with_suffix(".vrm")
            elif output_format == "fbx" and output_path.suffix.lower() != ".fbx":
                output_path = output_path.with_suffix(".fbx")

            # Run retargeting in isolated environment
            worker = BVHtoFBXWorker()
            result_path, info = worker.retarget(
                bvh_file=str(Path(bvh_file).absolute()),
                character_path=str(character_path.absolute()),
                output_path=str(output_path.absolute()),
                character_type=character_type,
                output_format=output_format,
            )

            if not Path(result_path).exists():
                raise RuntimeError(f"Output file not created: {result_path}")

            # Add BVH info
            num_frames = bvh_data.get("num_frames", 0)
            fps = bvh_data.get("fps", 30)
            full_info = (
                f"BVH Retargeting Complete\n"
                f"Character: {character_path.name}\n"
                f"BVH: {Path(bvh_file).name}\n"
                f"Output: {output_path.name}\n"
                f"Frames: {num_frames}\n"
                f"FPS: {fps}\n"
                f"Format: {output_format.upper()}\n"
            )

            log.info("Retargeting complete!")
            return (str(output_path.absolute()), full_info)

        except Exception as e:
            error_msg = f"BVHtoFBX Failed:\n{str(e)}"
            log.error(error_msg)
            return ("", error_msg)


NODE_CLASS_MAPPINGS = {
    "BVHtoFBX": BVHtoFBX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BVHtoFBX": "BVH to FBX Retargeter",
}
