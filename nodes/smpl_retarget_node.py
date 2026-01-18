"""
SMPLRetargetToSMPL Node - Apply SMPL motion to a rigged SMPL skeleton FBX

Uses the @isolated decorator to run Blender operations in an isolated environment
with the bpy package.

This node takes SMPL motion data and applies it to a rigged FBX with SMPL skeleton
(typically from UniRig with SMPL template), producing an animated FBX.
"""

import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from comfy_env import isolated

from hmr4d.utils.pylogger import Log


# ═══════════════════════════════════════════════════════════════════════════════
# SMPL SKELETON CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SMPL_JOINT_NAMES = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee',
    'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
    'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist'
]


# ═══════════════════════════════════════════════════════════════════════════════
# ISOLATED BLENDER WORKER
# ═══════════════════════════════════════════════════════════════════════════════

@isolated(env="mocap", import_paths=[".", "..", "../.."])
class SMPLRetargetWorker:
    """
    Isolated worker for SMPL-to-SMPL retargeting using bpy.
    Runs in the mocap isolated environment with bpy package.
    """

    def retarget(
        self,
        input_fbx: str,
        motion_npz: str,
        output_fbx: str,
    ) -> Tuple[str, int]:
        """
        Apply SMPL motion data to a rigged FBX and export animated FBX.

        Args:
            input_fbx: Path to input rigged FBX (with SMPL skeleton)
            motion_npz: Path to motion data NPZ file
            output_fbx: Path to output animated FBX

        Returns:
            Tuple of (output_path, frame_count)
        """
        import bpy
        import numpy as np
        from mathutils import Matrix, Vector

        print("=" * 60)
        print("SMPL Animation Application")
        print("=" * 60)
        print(f"Input FBX: {input_fbx}")
        print(f"Motion NPZ: {motion_npz}")
        print(f"Output FBX: {output_fbx}")

        def rodrigues_to_matrix(rotvec):
            """Convert axis-angle (Rodrigues) to rotation matrix."""
            angle = np.linalg.norm(rotvec)
            if angle < 1e-8:
                return Matrix.Identity(3)
            axis = Vector(rotvec / angle)
            return Matrix.Rotation(angle, 3, axis)

        # Load motion data
        motion = np.load(motion_npz)
        num_frames = motion['body_pose'].shape[0]
        print(f"Number of frames: {num_frames}")

        body_pose = motion['body_pose'][:num_frames]
        global_orient = motion['global_orient'][:num_frames]
        transl = motion['transl'][:num_frames]

        # Clear scene and import FBX
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        bpy.ops.import_scene.fbx(filepath=input_fbx)

        # Find armature and mesh
        armature = None
        mesh_obj = None
        for obj in bpy.context.scene.objects:
            if obj.type == 'ARMATURE':
                armature = obj
            elif obj.type == 'MESH':
                mesh_obj = obj

        if armature is None:
            raise RuntimeError("No armature found in FBX")
        if mesh_obj is None:
            raise RuntimeError("No mesh found in FBX")

        print(f"Armature: {armature.name}")
        print(f"Mesh: {mesh_obj.name}")

        # Get bone rest matrices in edit mode
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='EDIT')

        bone_rest_matrices = {}
        for name in SMPL_JOINT_NAMES:
            bone = armature.data.edit_bones.get(name)
            if bone:
                bone_rest_matrices[name] = bone.matrix.to_3x3().copy()

        bpy.ops.object.mode_set(mode='POSE')

        # SMPL bone local coordinate system:
        #   X: along bone (from joint to child)
        #   Y: perpendicular
        #   Z: perpendicular
        #
        # Blender bone local coordinate system:
        #   X: perpendicular (roll)
        #   Y: along bone
        #   Z: perpendicular
        #
        # Basis transformation: SMPL local -> Blender local
        smpl_to_blender_local = Matrix([
            [0, 0, 1],  # Blender X = SMPL Z
            [1, 0, 0],  # Blender Y = SMPL X
            [0, 1, 0],  # Blender Z = SMPL Y
        ]).to_3x3()
        smpl_to_blender_local_inv = smpl_to_blender_local.inverted()

        # World coordinate conversion (SMPL world -> rig world)
        # UniRig SMPL export: right = -X (L_shoulder at +X, R at -X)
        # SMPL standard: right = +X
        smpl_to_rig_world = Matrix([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ]).to_3x3()

        # Set rotation mode for all bones
        for bone in armature.pose.bones:
            bone.rotation_mode = 'QUATERNION'

        # Set frame range
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = num_frames

        print(f"Applying {num_frames} frames...")

        # Apply animation frame by frame
        for frame in range(num_frames):
            bpy.context.scene.frame_set(frame + 1)

            # Apply global orientation to Pelvis
            pelvis = armature.pose.bones.get('Pelvis')
            if pelvis:
                R_smpl_world = rodrigues_to_matrix(global_orient[frame])

                # Convert SMPL world rotation to rig world
                R_rig_world = smpl_to_rig_world @ R_smpl_world @ smpl_to_rig_world.inverted()

                # Convert world rotation to bone local
                rest_mat = bone_rest_matrices['Pelvis']
                R_local = rest_mat.inverted() @ R_rig_world @ rest_mat

                pelvis.rotation_quaternion = R_local.to_quaternion()
                pelvis.keyframe_insert(data_path='rotation_quaternion', frame=frame + 1)

                # Apply translation (with coordinate flip)
                t = transl[frame]
                pelvis.location = Vector((-t[0], t[1], -t[2]))
                pelvis.keyframe_insert(data_path='location', frame=frame + 1)

            # Apply body pose to other joints
            for joint_idx in range(21):
                joint_name = SMPL_JOINT_NAMES[joint_idx + 1]
                bone = armature.pose.bones.get(joint_name)
                if bone:
                    rotvec = body_pose[frame, joint_idx * 3:(joint_idx + 1) * 3]

                    # SMPL rotation is in SMPL bone-local coordinates
                    R_smpl_local = rodrigues_to_matrix(rotvec)

                    # Convert from SMPL local coords to Blender local coords
                    R_blender_local = smpl_to_blender_local @ R_smpl_local @ smpl_to_blender_local_inv

                    bone.rotation_quaternion = R_blender_local.to_quaternion()
                    bone.keyframe_insert(data_path='rotation_quaternion', frame=frame + 1)

            if frame % 100 == 0:
                print(f"  Frame {frame + 1}/{num_frames}")

        print("Animation applied!")

        # Export animated FBX
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        armature.select_set(True)
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = armature

        bpy.ops.export_scene.fbx(
            filepath=output_fbx,
            use_selection=True,
            object_types={'ARMATURE', 'MESH'},
            add_leaf_bones=False,
            bake_anim=True,
            bake_anim_use_all_actions=False,
            bake_anim_use_nla_strips=False,
        )

        print(f"Exported: {output_fbx}")
        return (output_fbx, num_frames)


# ═══════════════════════════════════════════════════════════════════════════════
# COMFYUI NODE
# ═══════════════════════════════════════════════════════════════════════════════

class SMPLRetargetToSMPL:
    """
    Apply SMPL motion data to a rigged FBX with SMPL skeleton.
    Runs in an isolated environment with the bpy package.

    This is different from SMPLToFBX which uses BVH+Rokoko for arbitrary rigs.
    This node directly applies SMPL rotations to an SMPL-skeleton FBX,
    preserving the exact motion without intermediate conversion.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "smpl_params": ("SMPL_PARAMS",),
                "rigged_fbx_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "output_filename": ("STRING", {
                    "default": "animated_character",
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("fbx_path", "frame_count")
    FUNCTION = "retarget"
    OUTPUT_NODE = True
    CATEGORY = "MotionCapture/Retarget"

    def retarget(
        self,
        smpl_params: Dict,
        rigged_fbx_path: str,
        output_filename: str,
    ) -> Tuple[str, int]:
        """
        Apply SMPL motion to a rigged SMPL skeleton FBX.

        Args:
            smpl_params: SMPL parameters from LoadSMPL or GVHMRInference
            rigged_fbx_path: Path to input rigged FBX with SMPL skeleton
            output_filename: Name for output animated FBX (without extension)

        Returns:
            Tuple of (output_fbx_path, frame_count)
        """
        try:
            Log.info("[SMPLRetargetToSMPL] Starting SMPL-to-SMPL retargeting...")

            # Validate input FBX
            rigged_fbx_path = Path(rigged_fbx_path)
            if not rigged_fbx_path.exists():
                raise FileNotFoundError(f"Input FBX not found: {rigged_fbx_path}")

            # Prepare output path
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            if not output_filename.endswith('.fbx'):
                output_filename = f"{output_filename}.fbx"
            output_path = output_dir / output_filename

            # Extract SMPL parameters to temporary NPZ
            temp_dir = Path(tempfile.gettempdir())
            motion_npz_path = temp_dir / "smpl_motion_temp.npz"
            frame_count = self._save_smpl_params(smpl_params, motion_npz_path)

            Log.info(f"[SMPLRetargetToSMPL] Saved motion data: {motion_npz_path} ({frame_count} frames)")

            # Run retargeting in isolated environment
            worker = SMPLRetargetWorker()
            result_path, result_frames = worker.retarget(
                input_fbx=str(rigged_fbx_path.absolute()),
                motion_npz=str(motion_npz_path),
                output_fbx=str(output_path.absolute()),
            )

            if not Path(result_path).exists():
                raise RuntimeError(f"Output FBX not created: {result_path}")

            Log.info(f"[SMPLRetargetToSMPL] Animation complete! Output: {output_path}")
            return (str(output_path.absolute()), result_frames)

        except Exception as e:
            error_msg = f"SMPLRetargetToSMPL failed: {str(e)}"
            Log.error(error_msg)
            import traceback
            traceback.print_exc()
            return ("", 0)

    def _save_smpl_params(self, smpl_params: Dict, output_path: Path) -> int:
        """
        Save SMPL parameters to NPZ file for the isolated worker.

        Returns:
            Number of frames in the motion
        """
        # Extract global parameters
        global_params = smpl_params.get("global", smpl_params)

        # Convert to numpy
        np_params = {}
        for key, value in global_params.items():
            if isinstance(value, torch.Tensor):
                np_params[key] = value.cpu().numpy()
            elif isinstance(value, np.ndarray):
                np_params[key] = value
            else:
                np_params[key] = np.array(value)

        # Determine frame count
        frame_count = 0
        if 'body_pose' in np_params:
            frame_count = np_params['body_pose'].shape[0]
        elif 'global_orient' in np_params:
            frame_count = np_params['global_orient'].shape[0]

        np.savez(output_path, **np_params)
        Log.info(f"[SMPLRetargetToSMPL] Saved SMPL params: {list(np_params.keys())}")

        return frame_count


NODE_CLASS_MAPPINGS = {
    "SMPLRetargetToSMPL": SMPLRetargetToSMPL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SMPLRetargetToSMPL": "SMPL to SMPL Retargeting",
}
