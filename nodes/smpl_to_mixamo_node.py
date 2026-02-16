"""
SMPLToMixamo Node - Retarget SMPL motion to Mixamo-rigged FBX characters

Blender operations run in an isolated environment with the bpy package.
This node is specifically optimized for Mixamo characters with the mixamorig: prefix.
"""

import tempfile
import os
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

log = logging.getLogger("motioncapture")

from .smpl_bvh_utils import smpl_to_bvh

# SMPL to Mixamo bone mapping
SMPL_TO_MIXAMO = {
    'Pelvis': 'mixamorig:Hips',
    'L_Hip': 'mixamorig:LeftUpLeg',
    'R_Hip': 'mixamorig:RightUpLeg',
    'Spine1': 'mixamorig:Spine',
    'L_Knee': 'mixamorig:LeftLeg',
    'R_Knee': 'mixamorig:RightLeg',
    'Spine2': 'mixamorig:Spine1',
    'L_Ankle': 'mixamorig:LeftFoot',
    'R_Ankle': 'mixamorig:RightFoot',
    'Spine3': 'mixamorig:Spine2',
    'L_Foot': 'mixamorig:LeftToeBase',
    'R_Foot': 'mixamorig:RightToeBase',
    'Neck': 'mixamorig:Neck',
    'L_Collar': 'mixamorig:LeftShoulder',
    'R_Collar': 'mixamorig:RightShoulder',
    'Head': 'mixamorig:Head',
    'L_Shoulder': 'mixamorig:LeftArm',
    'R_Shoulder': 'mixamorig:RightArm',
    'L_Elbow': 'mixamorig:LeftForeArm',
    'R_Elbow': 'mixamorig:RightForeArm',
    'L_Wrist': 'mixamorig:LeftHand',
    'R_Wrist': 'mixamorig:RightHand',
}



# ===============================================================================
# ISOLATED BLENDER WORKER
# ===============================================================================

class SMPLToMixamoWorker:
    """
    Isolated worker for SMPL to Mixamo retargeting using bpy.
    Runs in the mocap isolated environment with bpy package.
    """

    FUNCTION = "retarget"

    def retarget(
        self,
        smpl_data_path: str,
        mixamo_fbx: str,
        output_fbx: str,
        fps: int,
    ) -> Tuple[str, int, str]:
        """
        Retarget SMPL motion to Mixamo character using bpy.

        Args:
            smpl_data_path: Path to npz file with SMPL params
            mixamo_fbx: Path to input Mixamo FBX file
            output_fbx: Path to output FBX file
            fps: Frame rate

        Returns:
            Tuple of (output_path, frame_count, info_string)
        """
        import bpy
        import mathutils

        log.info("=" * 60)
        log.info("SMPL to Mixamo Retargeting")
        log.info("=" * 60)

        # Clear scene
        bpy.ops.wm.read_homefile(use_empty=True)
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # Load SMPL data
        log.info("Loading SMPL data from: %s", smpl_data_path)
        smpl_data = np.load(smpl_data_path)
        smpl_params = {key: smpl_data[key] for key in smpl_data.files}
        log.info("Loaded: %s", list(smpl_params.keys()))

        # Convert SMPL to BVH
        bvh_path = os.path.join(tempfile.gettempdir(), "smpl_mixamo_temp.bvh")
        log.info("Converting SMPL to BVH: %s", bvh_path)
        smpl_to_bvh(smpl_params, bvh_path, fps=fps)

        # Import BVH
        log.info("Importing BVH...")
        bpy.ops.import_anim.bvh(filepath=bvh_path)
        bvh_armature = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE'][0]

        # Store horizontal root motion from BVH (X, Y only)
        log.info("Storing BVH horizontal root motion...")
        bpy.context.view_layer.objects.active = bvh_armature
        bpy.ops.object.mode_set(mode='POSE')
        pelvis = bvh_armature.pose.bones.get("Pelvis")

        original_xy = []
        num_frames = int(bpy.context.scene.frame_end)
        for frame in range(1, num_frames + 1):
            bpy.context.scene.frame_set(frame)
            world_pos = bvh_armature.matrix_world @ pelvis.head
            original_xy.append((world_pos.x, world_pos.y))

        bpy.ops.object.mode_set(mode='OBJECT')

        # Import target Mixamo FBX
        log.info("Importing Mixamo FBX: %s", mixamo_fbx)
        bpy.ops.import_scene.fbx(filepath=mixamo_fbx, automatic_bone_orientation=True)
        armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']
        target_armature = [a for a in armatures if a != bvh_armature][0]

        # Verify it's a Mixamo rig
        mixamo_hips = target_armature.pose.bones.get("mixamorig:Hips")
        if not mixamo_hips:
            log.warning("No 'mixamorig:Hips' found - this may not be a Mixamo rig")

        # Delete BVH armature
        bpy.data.objects.remove(bvh_armature, do_unlink=True)

        # Apply horizontal root motion to Mixamo Hips
        log.info("Applying horizontal root motion...")
        bpy.context.view_layer.objects.active = target_armature
        bpy.ops.object.mode_set(mode='POSE')

        hips = target_armature.pose.bones.get("mixamorig:Hips")
        if not hips:
            # Fallback to other common names
            for name in ["Hips", "pelvis", "Pelvis", "hip", "Root"]:
                hips = target_armature.pose.bones.get(name)
                if hips:
                    break

        if hips and len(original_xy) > 0:
            bpy.context.scene.frame_set(1)
            ref_hips_world = target_armature.matrix_world @ hips.head
            ref_xy = original_xy[0]

            for frame in range(1, min(num_frames + 1, len(original_xy) + 1)):
                bpy.context.scene.frame_set(frame)
                current_world = target_armature.matrix_world @ hips.head

                target_x = original_xy[frame - 1][0] - ref_xy[0]
                target_y = original_xy[frame - 1][1] - ref_xy[1]
                current_x = current_world.x - ref_hips_world.x
                current_y = current_world.y - ref_hips_world.y

                delta_world = mathutils.Vector((target_x - current_x, target_y - current_y, 0))
                bone_matrix_inv = hips.bone.matrix_local.inverted()
                delta_local = bone_matrix_inv.to_3x3() @ delta_world

                hips.location = hips.location + delta_local
                hips.keyframe_insert(data_path="location", frame=frame)

        bpy.ops.object.mode_set(mode='OBJECT')

        # Export FBX
        log.info("Exporting to: %s", output_fbx)
        bpy.ops.object.select_all(action='DESELECT')
        target_armature.select_set(True)
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj.parent == target_armature:
                obj.select_set(True)

        bpy.context.view_layer.objects.active = target_armature

        bpy.ops.export_scene.fbx(
            filepath=output_fbx,
            use_selection=True,
            object_types={'ARMATURE', 'MESH'},
            bake_anim=True,
            bake_anim_use_all_bones=True,
            bake_anim_use_nla_strips=False,
            bake_anim_use_all_actions=False,
            bake_anim_step=1.0,
            bake_anim_simplify_factor=0.0,
            add_leaf_bones=False,
        )

        info = (
            f"SMPL to Mixamo Complete\n"
            f"Output: {output_fbx}\n"
            f"Frames: {num_frames}"
        )

        log.info("=" * 60)
        log.info("RETARGETING COMPLETE!")
        log.info("Output: %s", output_fbx)
        log.info("Frames: %d", num_frames)
        log.info("=" * 60)

        return (output_fbx, num_frames, info)


# ===============================================================================
# COMFYUI NODE
# ===============================================================================

class SMPLToMixamo:
    """
    Retarget SMPL motion capture data to a Mixamo-rigged FBX character.

    This node is specifically designed for Mixamo characters (with mixamorig: prefix).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "smpl_npz_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "mixamo_fbx_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "output_filename": ("STRING", {
                    "default": "mixamo_animated",
                    "multiline": False,
                }),
            },
            "optional": {
                "fps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 120,
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("fbx_path", "frame_count", "info")
    FUNCTION = "retarget"
    OUTPUT_NODE = True
    CATEGORY = "MotionCapture/Mixamo"

    def retarget(
        self,
        smpl_npz_path: str,
        mixamo_fbx_path: str,
        output_filename: str,
        fps: int = 30,
    ) -> Tuple[str, int, str]:
        """
        Retarget SMPL motion to Mixamo character.

        Args:
            smpl_npz_path: Path to npz file containing SMPL parameters
            mixamo_fbx_path: Path to input Mixamo FBX file
            output_filename: Name for output FBX (without extension)
            fps: Frame rate for animation

        Returns:
            Tuple of (output_fbx_path, frame_count, info_string)
        """
        try:
            log.info("Starting SMPL to Mixamo retargeting...")

            # Validate inputs
            smpl_npz_path = Path(smpl_npz_path)
            if not smpl_npz_path.exists():
                raise FileNotFoundError(f"SMPL npz file not found: {smpl_npz_path}")

            mixamo_fbx_path = Path(mixamo_fbx_path)
            if not mixamo_fbx_path.exists():
                raise FileNotFoundError(f"Mixamo FBX not found: {mixamo_fbx_path}")

            # Prepare output directory
            output_dir = Path("output")
            output_dir.mkdir(parents=True, exist_ok=True)

            if not output_filename.endswith('.fbx'):
                output_filename = f"{output_filename}.fbx"
            output_path = output_dir / output_filename

            log.info("Using SMPL data: %s", smpl_npz_path)

            # Create worker and run retargeting in isolated environment
            worker = SMPLToMixamoWorker()
            result_path, result_frames, info = worker.retarget(
                smpl_data_path=str(smpl_npz_path.absolute()),
                mixamo_fbx=str(mixamo_fbx_path.absolute()),
                output_fbx=str(output_path.absolute()),
                fps=fps,
            )

            if not Path(result_path).exists():
                raise RuntimeError(f"Output FBX not created: {result_path}")

            full_info = (
                f"SMPLToMixamo Complete\n"
                f"SMPL: {smpl_npz_path.name}\n"
                f"Mixamo: {mixamo_fbx_path.name}\n"
                f"Output: {output_path.name}\n"
                f"Frames: {result_frames}\n"
                f"FPS: {fps}\n"
            )

            log.info("Retargeting complete! Output: %s", output_path)
            return (str(output_path.absolute()), result_frames, full_info)

        except Exception as e:
            error_msg = f"SMPLToMixamo failed: {str(e)}"
            log.error(error_msg, exc_info=True)
            return ("", 0, error_msg)

NODE_CLASS_MAPPINGS = {
    "SMPLToMixamo": SMPLToMixamo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SMPLToMixamo": "SMPL to Mixamo",
}
