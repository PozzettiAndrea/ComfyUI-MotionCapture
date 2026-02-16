"""
SMPLToFBX Node - Retargets SMPL motion to rigged FBX characters using bpy

Blender operations run in an isolated environment with the bpy package.
"""

import logging
import tempfile
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import numpy as np

log = logging.getLogger("motioncapture")

from .smpl_bvh_utils import smpl_to_bvh


# ===============================================================================
# ISOLATED BLENDER WORKER
# ===============================================================================


class SMPLToFBXWorker:
    """
    Isolated worker for Blender retargeting operations.
    Runs in the mocap isolated environment with bpy package.
    """

    FUNCTION = "retarget"

    def retarget(
        self,
        smpl_data_path: str,
        fbx_input: str,
        fbx_output: str,
        rig_type: str,
        fps: int,
    ) -> Tuple[str, str]:
        """
        Retarget SMPL motion to FBX character using bpy.

        Args:
            smpl_data_path: Path to npz file with SMPL params
            fbx_input: Path to input FBX file
            fbx_output: Path to output FBX file
            rig_type: Rig type (auto, vroid, mixamo, etc.)
            fps: Frame rate

        Returns:
            Tuple of (output_path, info_string)
        """
        import bpy
        import mathutils

        log.info("=" * 60)
        log.info("SMPL to FBX Retargeting")
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
        bvh_path = os.path.join(tempfile.gettempdir(), "smpl_temp.bvh")
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

        # Import target FBX
        log.info("Importing target FBX: %s", fbx_input)
        bpy.ops.import_scene.fbx(filepath=fbx_input, automatic_bone_orientation=True)
        armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']
        target_armature = [a for a in armatures if a != bvh_armature][0]

        # Delete BVH armature
        bpy.data.objects.remove(bvh_armature, do_unlink=True)

        # Apply horizontal root motion only (X, Y - no vertical adjustment)
        log.info("Applying horizontal root motion...")
        bpy.context.view_layer.objects.active = target_armature
        bpy.ops.object.mode_set(mode='POSE')

        # Find hips bone
        hips = target_armature.pose.bones.get("mixamorig:Hips")
        if not hips:
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
        log.info("Exporting to: %s", fbx_output)
        bpy.ops.object.select_all(action='DESELECT')
        target_armature.select_set(True)
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj.parent == target_armature:
                obj.select_set(True)

        bpy.context.view_layer.objects.active = target_armature

        bpy.ops.export_scene.fbx(
            filepath=fbx_output,
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
            f"Retargeting Complete\n"
            f"Output: {fbx_output}\n"
            f"Frames: {num_frames}"
        )

        log.info("=" * 60)
        log.info("RETARGETING COMPLETE!")
        log.info("Output: %s", fbx_output)
        log.info("Frames: %d", num_frames)
        log.info("=" * 60)

        return (fbx_output, info)


# ===============================================================================
# COMFYUI NODE
# ===============================================================================

class SMPLToFBX:
    """
    Retarget SMPL motion capture data to a rigged FBX character using bpy.
    Runs in an isolated environment with the bpy package.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "smpl_params": ("SMPL_PARAMS",),
                "fbx_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "output_path": ("STRING", {
                    "default": "output/retargeted.fbx",
                    "multiline": False,
                }),
            },
            "optional": {
                "rig_type": (["auto", "vroid", "mixamo", "rigify", "ue5_mannequin"],),
                "fps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 120,
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_fbx_path", "info")
    FUNCTION = "retarget"
    OUTPUT_NODE = True
    CATEGORY = "MotionCapture"

    def retarget(
        self,
        smpl_params: Dict,
        fbx_path: str,
        output_path: str,
        rig_type: str = "auto",
        fps: int = 30,
    ) -> Tuple[str, str]:
        """
        Retarget SMPL motion to FBX character.

        Args:
            smpl_params: SMPL parameters from GVHMRInference
            fbx_path: Path to input rigged FBX file
            output_path: Path to save retargeted FBX
            rig_type: Type of rig (auto-detect or specific)
            fps: Frame rate for animation

        Returns:
            Tuple of (output_fbx_path, info_string)
        """
        try:
            log.info("Starting FBX retargeting...")

            # Validate inputs
            fbx_path = Path(fbx_path)
            if not fbx_path.exists():
                raise FileNotFoundError(f"Input FBX not found: {fbx_path}")

            # Prepare output directory
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Extract SMPL parameters to temporary file
            temp_dir = Path(tempfile.gettempdir()) / "mocap_retarget"
            temp_dir.mkdir(exist_ok=True)
            smpl_data_path = temp_dir / "smpl_params.npz"
            self._save_smpl_params(smpl_params, smpl_data_path)

            log.info("Saved SMPL data to: %s", smpl_data_path)

            # Create worker and run retargeting in isolated environment
            worker = SMPLToFBXWorker()
            result_path, info = worker.retarget(
                smpl_data_path=str(smpl_data_path),
                fbx_input=str(fbx_path.absolute()),
                fbx_output=str(output_path.absolute()),
                rig_type=rig_type,
                fps=fps,
            )

            if not Path(result_path).exists():
                raise RuntimeError(f"Output FBX not created: {result_path}")

            # Add frame count to info
            num_frames = smpl_params["global"]["body_pose"].shape[1] if "global" in smpl_params else 0
            full_info = (
                f"SMPLToFBX Retargeting Complete\n"
                f"Input FBX: {fbx_path}\n"
                f"Output FBX: {output_path}\n"
                f"Frames: {num_frames}\n"
                f"FPS: {fps}\n"
                f"Rig type: {rig_type}\n"
            )

            log.info("Retargeting complete!")
            return (str(output_path.absolute()), full_info)

        except Exception as e:
            error_msg = f"SMPLToFBX failed: {str(e)}"
            log.error(error_msg, exc_info=True)
            return ("", error_msg)

    def _save_smpl_params(self, smpl_params: Dict, output_path: Path):
        """Save SMPL parameters to npz file for the isolated worker."""
        global_params = smpl_params.get("global", {})

        np_params = {}
        for key, value in global_params.items():
            if isinstance(value, torch.Tensor):
                np_params[key] = value.cpu().numpy()
            else:
                np_params[key] = np.array(value)

        np.savez(output_path, **np_params)
        log.info("Saved SMPL params: %s", list(np_params.keys()))


NODE_CLASS_MAPPINGS = {
    "SMPLToFBX": SMPLToFBX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SMPLToFBX": "SMPL to FBX Retargeting",
}
