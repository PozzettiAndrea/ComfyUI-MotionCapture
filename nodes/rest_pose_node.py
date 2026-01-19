"""
MotionCapture Extract Rest Pose Node
Extract skeleton rest pose from FBX or SMPL parameters, output as FBX.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import tempfile

try:
    import folder_paths
except ImportError:
    folder_paths = None

# SMPL canonical skeleton (24 joints)
SMPL_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]

SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
]

# T-pose canonical positions (Y-up, meters, ~1.7m height)
SMPL_REST_POSITIONS = np.array([
    [0.0, 0.0, 0.0],        # 0 Pelvis
    [0.09, -0.065, 0.0],    # 1 L_Hip
    [-0.09, -0.065, 0.0],   # 2 R_Hip
    [0.0, 0.09, 0.0],       # 3 Spine1
    [0.09, -0.49, 0.0],     # 4 L_Knee
    [-0.09, -0.49, 0.0],    # 5 R_Knee
    [0.0, 0.20, 0.0],       # 6 Spine2
    [0.09, -0.87, 0.0],     # 7 L_Ankle
    [-0.09, -0.87, 0.0],    # 8 R_Ankle
    [0.0, 0.32, 0.0],       # 9 Spine3
    [0.09, -0.92, 0.12],    # 10 L_Foot
    [-0.09, -0.92, 0.12],   # 11 R_Foot
    [0.0, 0.46, 0.0],       # 12 Neck
    [0.06, 0.40, 0.0],      # 13 L_Collar
    [-0.06, 0.40, 0.0],     # 14 R_Collar
    [0.0, 0.57, 0.0],       # 15 Head
    [0.18, 0.40, 0.0],      # 16 L_Shoulder
    [-0.18, 0.40, 0.0],     # 17 R_Shoulder
    [0.45, 0.40, 0.0],      # 18 L_Elbow
    [-0.45, 0.40, 0.0],     # 19 R_Elbow
    [0.70, 0.40, 0.0],      # 20 L_Wrist
    [-0.70, 0.40, 0.0],     # 21 R_Wrist
    [0.78, 0.40, 0.0],      # 22 L_Hand
    [-0.78, 0.40, 0.0],     # 23 R_Hand
], dtype=np.float32)


class ExtractRestPose:
    """
    Extract skeleton rest pose from FBX file or SMPL parameters.

    Outputs an FBX file path containing the skeleton in T-pose,
    compatible with CompareSkeletons for side-by-side comparison.

    For FBX source: Imports FBX, strips all animation data, exports T-pose
    For SMPL source: Creates armature from canonical SMPL joint positions
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_type": (["fbx", "smpl"], {
                    "default": "fbx",
                    "tooltip": "Source type: FBX file or SMPL parameters"
                }),
                "output_name": ("STRING", {
                    "default": "rest_pose",
                    "tooltip": "Output filename (without extension)"
                }),
            },
            "optional": {
                "fbx_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to input FBX file (when source_type=fbx)"
                }),
                "smpl_params": ("SMPL_PARAMS", {
                    "tooltip": "SMPL parameters dict (when source_type=smpl)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("fbx_path", "info")
    FUNCTION = "extract_rest_pose"
    CATEGORY = "MotionCapture/Skeleton"

    def extract_rest_pose(
        self,
        source_type: str,
        output_name: str,
        fbx_path: str = "",
        smpl_params: Optional[Dict] = None,
    ) -> Tuple[str, str]:
        """Extract rest pose skeleton and save as FBX."""
        from comfy_env import isolated

        print(f"[ExtractRestPose] Source type: {source_type}")

        # Setup output path
        if folder_paths:
            output_dir = folder_paths.get_output_directory()
        else:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)

        if not output_name.endswith('.fbx'):
            output_name_fbx = f"{output_name}.fbx"
        else:
            output_name_fbx = output_name
        output_path = os.path.join(output_dir, output_name_fbx)

        if source_type == "fbx":
            # Validate FBX path
            if not fbx_path:
                raise ValueError("fbx_path is required when source_type=fbx")

            # Handle relative paths
            if not os.path.isabs(fbx_path):
                if folder_paths:
                    input_dir = folder_paths.get_input_directory()
                    if os.path.exists(os.path.join(input_dir, fbx_path)):
                        fbx_path = os.path.join(input_dir, fbx_path)
                    elif os.path.exists(os.path.join(output_dir, fbx_path)):
                        fbx_path = os.path.join(output_dir, fbx_path)

            if not os.path.exists(fbx_path):
                raise FileNotFoundError(f"FBX file not found: {fbx_path}")

            print(f"[ExtractRestPose] Input FBX: {fbx_path}")

            # Extract rest pose from FBX using isolated worker
            worker = RestPoseWorker()
            bone_count = worker.extract_from_fbx(fbx_path, output_path)
            source_info = f"FBX: {os.path.basename(fbx_path)}"

        else:  # smpl
            print(f"[ExtractRestPose] Creating SMPL rest pose skeleton")

            # Use canonical SMPL positions
            joint_positions = SMPL_REST_POSITIONS.copy()

            # Create SMPL skeleton using isolated worker
            worker = RestPoseWorker()
            bone_count = worker.create_smpl_skeleton(
                joint_positions=joint_positions,
                joint_names=SMPL_JOINT_NAMES,
                parent_indices=SMPL_PARENTS,
                output_path=output_path,
            )
            source_info = "SMPL canonical T-pose"

        print(f"[ExtractRestPose] Output: {output_path}")
        print(f"[ExtractRestPose] Bones: {bone_count}")

        info = (
            f"Source: {source_info}\n"
            f"Bones: {bone_count}\n"
            f"Output: {output_name_fbx}"
        )

        return (output_path, info)


# Isolated worker for bpy operations
try:
    from comfy_env import isolated

    @isolated(env="mocap", import_paths=[".", "..", "../.."])
    class RestPoseWorker:
        """Blender worker for rest pose extraction."""

        FUNCTION = "extract_from_fbx"

        def extract_from_fbx(self, input_fbx: str, output_fbx: str) -> int:
            """Extract rest pose from FBX file."""
            import bpy
            from mathutils import Quaternion

            print(f"[RestPoseWorker] Extracting rest pose from: {input_fbx}")

            # Clean scene
            bpy.ops.wm.read_homefile(use_empty=True)
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete()

            # Import FBX
            bpy.ops.import_scene.fbx(filepath=input_fbx)

            # Find armature
            armature = None
            for obj in bpy.context.scene.objects:
                if obj.type == 'ARMATURE':
                    armature = obj
                    break

            if not armature:
                raise RuntimeError("No armature found in FBX file")

            bone_count = len(armature.data.bones)
            print(f"[RestPoseWorker] Found armature with {bone_count} bones")

            # Clear animation data
            if armature.animation_data:
                armature.animation_data_clear()

            # Reset bones to rest pose
            bpy.context.view_layer.objects.active = armature
            bpy.ops.object.mode_set(mode='POSE')

            for bone in armature.pose.bones:
                bone.rotation_mode = 'QUATERNION'
                bone.rotation_quaternion = Quaternion((1, 0, 0, 0))
                bone.location = (0, 0, 0)
                bone.scale = (1, 1, 1)

            bpy.ops.object.mode_set(mode='OBJECT')

            # Export FBX
            os.makedirs(os.path.dirname(output_fbx), exist_ok=True)
            bpy.ops.export_scene.fbx(
                filepath=output_fbx,
                use_selection=False,
                bake_anim=False,
                add_leaf_bones=True,
            )

            return bone_count

        def create_smpl_skeleton(
            self,
            joint_positions: np.ndarray,
            joint_names: list,
            parent_indices: list,
            output_path: str,
        ) -> int:
            """Create SMPL skeleton FBX from joint positions."""
            import bpy
            from mathutils import Vector

            print(f"[RestPoseWorker] Creating SMPL skeleton with {len(joint_names)} joints")

            # Clean scene
            bpy.ops.wm.read_homefile(use_empty=True)
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete()

            # Create armature
            armature_data = bpy.data.armatures.new("SMPL_Armature")
            armature_obj = bpy.data.objects.new("SMPL_Skeleton", armature_data)
            bpy.context.scene.collection.objects.link(armature_obj)
            bpy.context.view_layer.objects.active = armature_obj

            # Enter edit mode
            bpy.ops.object.mode_set(mode='EDIT')
            edit_bones = armature_data.edit_bones
            bones_by_name = {}

            for i, (name, pos) in enumerate(zip(joint_names, joint_positions)):
                bone = edit_bones.new(name)
                head = Vector(pos.tolist())

                # Find children for tail direction
                children_indices = [j for j, p in enumerate(parent_indices) if p == i]

                if children_indices:
                    child_positions = joint_positions[children_indices]
                    avg_child = np.mean(child_positions, axis=0)
                    tail = Vector(avg_child.tolist())
                    if (tail - head).length < 0.01:
                        tail = head + Vector((0, 0.05, 0))
                else:
                    parent_idx = parent_indices[i]
                    if parent_idx >= 0:
                        parent_pos = Vector(joint_positions[parent_idx].tolist())
                        direction = head - parent_pos
                        if direction.length > 0.001:
                            direction.normalize()
                            tail = head + direction * 0.05
                        else:
                            tail = head + Vector((0, 0.05, 0))
                    else:
                        tail = head + Vector((0, 0.1, 0))

                bone.head = head
                bone.tail = tail
                bones_by_name[name] = bone

            # Set parents
            for i, (name, parent_idx) in enumerate(zip(joint_names, parent_indices)):
                if parent_idx >= 0:
                    parent_name = joint_names[parent_idx]
                    bones_by_name[name].parent = bones_by_name[parent_name]

            bpy.ops.object.mode_set(mode='OBJECT')

            # Export
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            bpy.ops.export_scene.fbx(
                filepath=output_path,
                use_selection=False,
                bake_anim=False,
                add_leaf_bones=True,
            )

            return len(joint_names)

except ImportError:
    # Fallback if comfy_env not available
    class RestPoseWorker:
        def extract_from_fbx(self, input_fbx: str, output_fbx: str) -> int:
            raise RuntimeError("comfy_env not available - cannot run bpy operations")

        def create_smpl_skeleton(self, *args, **kwargs) -> int:
            raise RuntimeError("comfy_env not available - cannot run bpy operations")
