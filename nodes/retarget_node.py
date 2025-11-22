"""
SMPLToFBX Node - Retargets SMPL motion to rigged FBX characters
"""

from pathlib import Path
from typing import Dict, Tuple
import torch
import numpy as np

from hmr4d.utils.pylogger import Log


class SMPLToFBX:
    """
    Retarget SMPL motion capture data to a rigged FBX character using Blender.
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
            Log.info("[SMPLToFBX] Starting FBX retargeting...")

            # Validate inputs
            fbx_path = Path(fbx_path)
            if not fbx_path.exists():
                raise FileNotFoundError(f"Input FBX not found: {fbx_path}")

            # Get Blender executable
            blender_exe = self._find_blender()
            if not blender_exe:
                raise RuntimeError(
                    "Blender not found. Please run: python install.py --install-blender"
                )

            Log.info(f"[SMPLToFBX] Using Blender: {blender_exe}")

            # Prepare output directory
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Extract SMPL parameters to temporary files
            temp_dir = Path(__file__).parent.parent / "temp"
            temp_dir.mkdir(exist_ok=True)

            smpl_data_path = temp_dir / "smpl_params.npz"
            self._save_smpl_params(smpl_params, smpl_data_path)

            Log.info(f"[SMPLToFBX] Saved SMPL data to: {smpl_data_path}")

            # Create Blender retargeting script
            blender_script = self._create_blender_script(
                fbx_input=str(fbx_path.absolute()),
                fbx_output=str(output_path.absolute()),
                smpl_data=str(smpl_data_path.absolute()),
                rig_type=rig_type,
                fps=fps,
            )

            script_path = temp_dir / "retarget_script.py"
            with open(script_path, 'w') as f:
                f.write(blender_script)

            Log.info(f"[SMPLToFBX] Created Blender script: {script_path}")

            # Run Blender in background mode
            import subprocess

            cmd = [
                str(blender_exe),
                "--background",
                "--python", str(script_path),
            ]

            Log.info(f"[SMPLToFBX] Running Blender retargeting...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                Log.error(f"[SMPLToFBX] Blender error:\n{result.stderr}")
                raise RuntimeError(f"Blender retargeting failed: {result.stderr}")

            Log.info(f"[SMPLToFBX] Blender output:\n{result.stdout}")

            if not output_path.exists():
                raise RuntimeError(f"Output FBX not created: {output_path}")

            # Create info string
            num_frames = smpl_params["global"]["body_pose"].shape[1] if "global" in smpl_params else 0
            info = (
                f"SMPLToFBX Retargeting Complete\n"
                f"Input FBX: {fbx_path}\n"
                f"Output FBX: {output_path}\n"
                f"Frames: {num_frames}\n"
                f"FPS: {fps}\n"
                f"Rig type: {rig_type}\n"
            )

            Log.info("[SMPLToFBX] Retargeting complete!")
            return (str(output_path.absolute()), info)

        except Exception as e:
            error_msg = f"SMPLToFBX failed: {str(e)}"
            Log.error(error_msg)
            import traceback
            traceback.print_exc()
            return ("", error_msg)

    def _find_blender(self) -> Path:
        """Find Blender executable."""
        # Check local installation first
        local_blender = Path(__file__).parent.parent / "lib" / "blender"

        if local_blender.exists():
            import platform

            system = platform.system().lower()
            if system == "windows":
                pattern = "**/blender.exe"
            elif system == "darwin":
                pattern = "**/MacOS/blender"
            else:
                pattern = "**/blender"

            executables = list(local_blender.glob(pattern))
            if executables:
                return executables[0]

        # Check system PATH
        import shutil
        system_blender = shutil.which("blender")
        if system_blender:
            return Path(system_blender)

        return None

    def _save_smpl_params(self, smpl_params: Dict, output_path: Path):
        """Save SMPL parameters to npz file for Blender."""
        # Extract global parameters
        global_params = smpl_params.get("global", {})

        # Convert to numpy and save
        np_params = {}
        for key, value in global_params.items():
            if isinstance(value, torch.Tensor):
                np_params[key] = value.cpu().numpy()
            else:
                np_params[key] = np.array(value)

        np.savez(output_path, **np_params)
        Log.info(f"[SMPLToFBX] Saved SMPL params: {list(np_params.keys())}")

    def _create_blender_script(
        self,
        fbx_input: str,
        fbx_output: str,
        smpl_data: str,
        rig_type: str,
        fps: int,
    ) -> str:
        """Create Python script for Blender to execute retargeting."""

        script = f'''
import bpy
import numpy as np
import mathutils
from pathlib import Path

def clear_scene():
    """Clear default scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def load_smpl_params(smpl_path):
    """Load SMPL parameters from npz file."""
    data = np.load(smpl_path)
    return {{key: data[key] for key in data.files}}

def import_fbx(fbx_path):
    """Import FBX character."""
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    # Get the armature
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']
    if not armatures:
        raise RuntimeError("No armature found in FBX file")
    return armatures[0]

def validate_vroid_rig(armature):
    """Validate that armature has VRoid bone naming."""
    print(f"Validating VRoid rig for armature: {{armature.name}}")

    # Required VRoid bones for basic retargeting
    required_bones = [
        'J_Bip_C_Hips',
        'J_Bip_C_Spine',
        'J_Bip_C_Chest',
        'J_Bip_C_UpperChest',
        'J_Bip_L_Shoulder',
        'J_Bip_R_Shoulder',
    ]

    # Check which required bones exist
    found_bones = []
    missing_bones = []

    for bone_name in required_bones:
        if bone_name in armature.data.bones:
            found_bones.append(bone_name)
        else:
            missing_bones.append(bone_name)

    print(f"Found {{len(found_bones)}}/{{len(required_bones)}} required VRoid bones")

    if len(found_bones) < 4:
        print("\\n" + "="*60)
        print("ERROR: This does not appear to be a VRoid rig!")
        print("="*60)
        print(f"Only {{len(found_bones)}}/{{len(required_bones)}} required bones found.")
        print(f"\\nMissing bones: {{', '.join(missing_bones)}}")
        print(f"\\nAvailable bones in FBX:")
        for bone in list(armature.data.bones.keys())[:20]:
            print(f"  - {{bone}}")
        if len(armature.data.bones) > 20:
            print(f"  ... and {{len(armature.data.bones) - 20}} more")
        print("\\nPlease use a VRoid character FBX for retargeting.")
        print("="*60)
        raise ValueError(f"Invalid rig: Only {{len(found_bones)}}/{{len(required_bones)}} required VRoid bones found")

    print(f"✓ VRoid rig validation passed! Found bones: {{', '.join(found_bones)}}")
    return True

def create_smpl_armature(smpl_params):
    """Create SMPL armature from parameters."""
    # For now, create a simple skeleton
    # TODO: Full SMPL skeleton implementation
    bpy.ops.object.armature_add()
    smpl_armature = bpy.context.active_object
    smpl_armature.name = "SMPL_Armature"
    return smpl_armature

def apply_smpl_animation(armature, smpl_params, fps):
    """Apply SMPL animation to armature."""
    # Get parameters
    body_pose = smpl_params.get('body_pose')  # Shape: (B, L, 63) or (B, L, 21, 3)
    global_orient = smpl_params.get('global_orient')  # Shape: (B, L, 3)
    transl = smpl_params.get('transl')  # Shape: (B, L, 3)

    if body_pose is None:
        print("No body_pose found in SMPL params")
        return

    # Set scene FPS
    bpy.context.scene.render.fps = fps

    # Get number of frames
    num_frames = body_pose.shape[1] if len(body_pose.shape) > 2 else body_pose.shape[0]
    print(f"Applying animation: {{num_frames}} frames at {{fps}} FPS")

    # Set animation range
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = num_frames

    # Apply root translation if available
    if transl is not None and len(transl.shape) >= 2:
        for frame_idx in range(num_frames):
            bpy.context.scene.frame_set(frame_idx + 1)
            # Assuming transl shape is (B, L, 3), take first batch
            trans = transl[0, frame_idx] if len(transl.shape) == 3 else transl[frame_idx]
            armature.location = mathutils.Vector(trans.tolist())
            armature.keyframe_insert(data_path="location", frame=frame_idx + 1)

    print(f"Animation applied successfully")

def retarget_smpl_to_armature(target_armature, smpl_params, fps, rig_type):
    """
    Retarget SMPL motion to target armature using proper coordinate system conversion.

    This function handles:
    1. Y-up (SMPL) to Z-up (Blender) coordinate conversion
    2. Bone rest pose transformation
    3. Both rotations AND translations
    """

    # SMPL joint order (22 joints)
    SMPL_JOINTS = [
        "hips", "leftUpLeg", "rightUpLeg", "spine", "leftLeg", "rightLeg",
        "spine1", "leftFoot", "rightFoot", "spine2", "leftToeBase", "rightToeBase",
        "neck", "leftShoulder", "rightShoulder", "head", "leftArm", "rightArm",
        "leftForeArm", "rightForeArm", "leftHand", "rightHand"
    ]

    # VRoid bone mapping
    SMPL_TO_VROID_MAPPING = {{
        "hips": "J_Bip_C_Hips",
        "spine": "J_Bip_C_Spine",
        "spine1": "J_Bip_C_Chest",
        "spine2": "J_Bip_C_UpperChest",
        "neck": "J_Bip_C_Neck",
        "head": "J_Bip_C_Head",
        "leftShoulder": "J_Bip_L_Shoulder",
        "leftArm": "J_Bip_L_UpperArm",
        "leftForeArm": "J_Bip_L_LowerArm",
        "leftHand": "J_Bip_L_Hand",
        "rightShoulder": "J_Bip_R_Shoulder",
        "rightArm": "J_Bip_R_UpperArm",
        "rightForeArm": "J_Bip_R_LowerArm",
        "rightHand": "J_Bip_R_Hand",
        "leftUpLeg": "J_Bip_L_UpperLeg",
        "leftLeg": "J_Bip_L_LowerLeg",
        "leftFoot": "J_Bip_L_Foot",
        "leftToeBase": "J_Bip_L_ToeBase",
        "rightUpLeg": "J_Bip_R_UpperLeg",
        "rightLeg": "J_Bip_R_LowerLeg",
        "rightFoot": "J_Bip_R_Foot",
        "rightToeBase": "J_Bip_R_ToeBase",
    }}

    # ========== COORDINATE SYSTEM CONVERSION ==========
    # SMPL uses Z-up with Y=forward: (X=right, Y=forward, Z=up)
    # Blender uses Z-up with Y=back: (X=right, Y=back, Z=up)
    # Conversion matrix: X→X, Y→-Y, Z→Z
    print("\\n" + "="*60)
    print("COORDINATE SYSTEM SETUP")
    print("="*60)
    print("SMPL coordinate system: (X=right, Y=forward, Z=up)")
    print("Blender coordinate system: (X=right, Y=back, Z=up)")
    print("Conversion: X→X, Y→-Y, Z→Z (flip Y axis)")

    SMPL_TO_BLENDER = mathutils.Matrix((
        (1,  0,  0),   # X stays X (right)
        (0, -1,  0),   # Y becomes -Y (forward becomes back)
        (0,  0,  1)    # Z stays Z (up)
    ))
    print(f"SMPL-to-Blender conversion matrix:\\n{{SMPL_TO_BLENDER}}")

    # ========== T-POSE TO A-POSE OFFSET ==========
    # SMPL uses T-pose (arms horizontal), VRoid uses A-pose (arms ~45° down)
    # Apply corrective rotation for affected bones
    import math
    ARM_POSE_OFFSETS = {{
        "J_Bip_L_Shoulder": math.radians(45),   # 45° offset
        "J_Bip_L_UpperArm": math.radians(45),
        "J_Bip_R_Shoulder": math.radians(45),
        "J_Bip_R_UpperArm": math.radians(45),
    }}
    print(f"\\nT-pose to A-pose offsets defined for {{len(ARM_POSE_OFFSETS)}} bones")

    # Get bone mapping based on rig type
    if rig_type == "vroid":
        bone_mapping = SMPL_TO_VROID_MAPPING
    else:
        print(f"Warning: Rig type '{{rig_type}}' not implemented, using VRoid mapping")
        bone_mapping = SMPL_TO_VROID_MAPPING

    # Get SMPL parameters
    body_pose = smpl_params.get('body_pose')  # Shape: (B, L, 63) or (L, 63)
    global_orient = smpl_params.get('global_orient')  # Shape: (B, L, 3) or (L, 3)
    transl = smpl_params.get('transl')  # Shape: (B, L, 3) or (L, 3)

    if body_pose is None:
        print("ERROR: No body_pose found in SMPL params")
        return

    print(f"body_pose shape: {{body_pose.shape}}")
    print(f"global_orient shape: {{global_orient.shape if global_orient is not None else 'None'}}")
    print(f"transl shape: {{transl.shape if transl is not None else 'None'}}")

    # Handle batch dimension - take first batch if present
    if len(body_pose.shape) == 3:  # (B, L, 63)
        body_pose = body_pose[0]  # (L, 63)
    if global_orient is not None and len(global_orient.shape) == 3:  # (B, L, 3)
        global_orient = global_orient[0]  # (L, 3)
    if transl is not None and len(transl.shape) == 3:  # (B, L, 3)
        transl = transl[0]  # (L, 3)

    # Reshape body_pose from (L, 63) to (L, 21, 3)
    num_frames = body_pose.shape[0]
    body_pose = body_pose.reshape(num_frames, 21, 3)

    print(f"Retargeting {{num_frames}} frames to {{target_armature.name}}")

    # Debug: Print armature scale and bone count
    print(f"\\nTarget Armature Info:")
    print(f"  Name: {{target_armature.name}}")
    print(f"  Scale: {{target_armature.scale}}")
    print(f"  Location: {{target_armature.location}}")
    print(f"  Bone count: {{len(target_armature.data.bones)}}")

    # Set scene FPS and frame range
    bpy.context.scene.render.fps = fps
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = num_frames

    # Set armature as active and enter pose mode
    bpy.context.view_layer.objects.active = target_armature
    bpy.ops.object.mode_set(mode='POSE')

    # ========== EXTRACT BONE REST POSE MATRICES ==========
    # CRITICAL: We need bone rest orientations to transform rotations properly
    print("\\n" + "="*60)
    print("EXTRACTING BONE REST POSE MATRICES")
    print("="*60)
    bone_rest_matrices = {{}}
    for bone_name in target_armature.pose.bones.keys():
        pose_bone = target_armature.pose.bones[bone_name]
        # matrix_local is the bone's rest pose matrix in armature space
        # We need the 3x3 rotation part for coordinate transformations
        rest_matrix_3x3 = pose_bone.bone.matrix_local.to_3x3()
        bone_rest_matrices[bone_name] = rest_matrix_3x3
        # Debug: Print first few bones
        if len(bone_rest_matrices) <= 3:
            print(f"  {{bone_name}}: rest matrix extracted (shape: 3x3)")
    print(f"✓ Extracted rest matrices for {{len(bone_rest_matrices)}} bones")

    # ========== APPLY ROOT TRANSLATION (WITH COORDINATE CONVERSION) ==========
    if transl is not None:
        print("\\n" + "="*60)
        print("APPLYING ROOT TRANSLATION (SMPL → Blender)")
        print("="*60)
        # Debug: Show first translation
        trans_smpl_0 = transl[0]
        trans_blender_0 = mathutils.Vector((
            trans_smpl_0[0],      # X stays X (right)
            -trans_smpl_0[1],     # -Y (forward becomes back)
            trans_smpl_0[2]       # Z stays Z (up)
        ))
        print(f"Frame 1 SMPL translation:    {{trans_smpl_0}} (X=right, Y=forward, Z=up)")
        print(f"Frame 1 Blender translation: {{trans_blender_0}} (X=right, Y=back, Z=up)")

        for frame_idx in range(num_frames):
            bpy.context.scene.frame_set(frame_idx + 1)
            trans_smpl = transl[frame_idx]
            # Convert SMPL (Y=forward) to Blender (Y=back)
            trans_blender = mathutils.Vector((
                trans_smpl[0],      # X stays X (right)
                -trans_smpl[1],     # -Y (forward becomes back)
                trans_smpl[2]       # Z stays Z (up)
            ))
            target_armature.location = trans_blender
            target_armature.keyframe_insert(data_path="location", frame=frame_idx + 1)
        print(f"✓ Applied translation for {{num_frames}} frames")

    # ========== APPLY ROTATIONS TO EACH BONE ==========
    print("\\n" + "="*60)
    print("BONE MAPPING AND ROTATION RETARGETING")
    print("="*60)
    bones_mapped = 0
    bones_skipped = 0
    mapping_log = []

    for joint_idx, smpl_joint_name in enumerate(SMPL_JOINTS):
        # Skip hips - its rotation is handled by global_orient
        if smpl_joint_name == "hips":
            continue

        # Get target bone name from mapping
        target_bone_name = bone_mapping.get(smpl_joint_name)

        if target_bone_name is None:
            mapping_log.append(f"✗ {{smpl_joint_name:20s}} → [NO MAPPING]")
            bones_skipped += 1
            continue

        # Check if bone exists in armature
        if target_bone_name not in target_armature.pose.bones:
            mapping_log.append(f"✗ {{smpl_joint_name:20s}} → {{target_bone_name}} [NOT FOUND]")
            bones_skipped += 1
            continue

        pose_bone = target_armature.pose.bones[target_bone_name]
        rest_matrix = bone_rest_matrices[target_bone_name]
        mapping_log.append(f"✓ {{smpl_joint_name:20s}} → {{target_bone_name}}")

        # Set rotation mode to QUATERNION (more stable than AXIS_ANGLE)
        pose_bone.rotation_mode = 'QUATERNION'

        # Debug: Sample first frame rotation values
        body_pose_idx = joint_idx - 1
        sample_axis_angle = body_pose[0, body_pose_idx]
        sample_angle = np.linalg.norm(sample_axis_angle)

        # Debug first bone transformation in detail
        if bones_mapped == 0:
            print(f"\\n--- DEBUG: First Bone Transformation ({{smpl_joint_name}} → {{target_bone_name}}) ---")
            print(f"  SMPL axis-angle (frame 1): {{sample_axis_angle}}")
            print(f"  Angle magnitude: {{sample_angle:.4f}} radians ({{np.degrees(sample_angle):.2f}} degrees)")
            if sample_angle > 1e-8:
                sample_axis = sample_axis_angle / sample_angle
                print(f"  Rotation axis: {{sample_axis}}")

        # Apply rotation for each frame
        for frame_idx in range(num_frames):
            bpy.context.scene.frame_set(frame_idx + 1)

            # Get axis-angle rotation from body_pose (3D vector)
            # Note: body_pose excludes hips (joint 0), so subtract 1 from index
            axis_angle = body_pose[frame_idx, body_pose_idx]  # (3,)
            angle = np.linalg.norm(axis_angle)

            if angle > 1e-8:
                # ===== PROPER ROTATION TRANSFORMATION =====
                # Step 1: Convert axis-angle to rotation matrix
                axis = axis_angle / angle
                R_smpl = mathutils.Matrix.Rotation(
                    angle, 3, mathutils.Vector(axis.tolist())
                )

                # Step 2: Apply Y-up to Z-up coordinate conversion
                # R_converted = Y_TO_Z @ R_smpl @ Y_TO_Z.T
                R_converted = SMPL_TO_BLENDER @ R_smpl @ SMPL_TO_BLENDER.transposed()

                # Step 3: Apply T-pose to A-pose offset for arm bones
                if target_bone_name in ARM_POSE_OFFSETS:
                    offset_angle = ARM_POSE_OFFSETS[target_bone_name]
                    # Test X-axis rotation first (typical for shoulder rotation)
                    offset_axis = mathutils.Vector((1, 0, 0))
                    R_offset = mathutils.Matrix.Rotation(offset_angle, 3, offset_axis)
                    # Apply as: R_offset @ R_converted @ R_offset^-1
                    R_converted = R_offset @ R_converted @ R_offset.inverted()

                    # Debug first arm bone
                    if bones_mapped == 0 and frame_idx == 0:
                        print(f"  Applied {{math.degrees(offset_angle):.1f}}° offset on X-axis for {{target_bone_name}}")

                # Step 4: Transform to bone's local space using rest_matrix
                # This transforms rotation from world space to bone's local space
                R_local = rest_matrix.inverted() @ R_converted @ rest_matrix

                # Step 5: Convert to quaternion
                quat = R_local.to_quaternion()
                pose_bone.rotation_quaternion = quat

                # Debug first bone, first frame
                if bones_mapped == 0 and frame_idx == 0:
                    print(f"  Final quaternion (with rest_matrix): {{quat}}")
            else:
                # Identity rotation
                pose_bone.rotation_quaternion = mathutils.Quaternion((1, 0, 0, 0))

            pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx + 1)

        bones_mapped += 1

    # Print mapping results
    print("\\nBone Mapping Results:")
    for log_line in mapping_log:
        print(f"  {{log_line}}")

    # ========== APPLY GLOBAL ORIENTATION TO ROOT BONE (HIPS) ==========
    if global_orient is not None:
        print("\\n" + "="*60)
        print("APPLYING GLOBAL ORIENTATION (ROOT/HIPS)")
        print("="*60)
        root_bone_name = bone_mapping.get("hips")
        if root_bone_name and root_bone_name in target_armature.pose.bones:
            root_bone = target_armature.pose.bones[root_bone_name]
            root_rest_matrix = bone_rest_matrices[root_bone_name]

            # Use QUATERNION mode for consistency
            root_bone.rotation_mode = 'QUATERNION'

            # Debug first frame
            sample_global_orient = global_orient[0]
            sample_global_angle = np.linalg.norm(sample_global_orient)
            print(f"Frame 1 global_orient: {{sample_global_orient}}")
            print(f"Angle magnitude: {{sample_global_angle:.4f}} radians ({{np.degrees(sample_global_angle):.2f}} degrees)")

            for frame_idx in range(num_frames):
                bpy.context.scene.frame_set(frame_idx + 1)

                # Get axis-angle rotation
                axis_angle = global_orient[frame_idx]
                angle = np.linalg.norm(axis_angle)

                if angle > 1e-8:
                    # Same transformation as other bones
                    axis = axis_angle / angle
                    R_smpl = mathutils.Matrix.Rotation(
                        angle, 3, mathutils.Vector(axis.tolist())
                    )
                    R_converted = SMPL_TO_BLENDER @ R_smpl @ SMPL_TO_BLENDER.transposed()
                    R_local = root_rest_matrix.inverted() @ R_converted @ root_rest_matrix
                    quat = R_local.to_quaternion()
                    root_bone.rotation_quaternion = quat

                    # Debug first frame
                    if frame_idx == 0:
                        print(f"  Root quaternion (with rest_matrix, frame 1): {{quat}}")
                else:
                    root_bone.rotation_quaternion = mathutils.Quaternion((1, 0, 0, 0))

                root_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx + 1)

            print(f"✓ Applied global orientation for {{num_frames}} frames")

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    print("\\n" + "="*60)
    print("RETARGETING SUMMARY")
    print("="*60)
    print(f"✓ Total frames processed: {{num_frames}}")
    print(f"✓ Bones successfully mapped: {{bones_mapped}}")
    print(f"✗ Bones skipped: {{bones_skipped}}")
    print(f"✓ Root translation: {{'Applied' if transl is not None else 'Not available'}}")
    print(f"✓ Global orientation: {{'Applied' if global_orient is not None else 'Not available'}}")
    print("="*60)

def export_fbx(output_path):
    """Export scene as FBX with armature, mesh, and textures."""
    # Deselect all first
    bpy.ops.object.select_all(action='DESELECT')

    # Select armature and all mesh objects
    for obj in bpy.data.objects:
        if obj.type in ['MESH', 'ARMATURE']:
            obj.select_set(True)
            print(f"(bpy.data.{{obj.type.lower()}}s['{{obj.name}}'], '{{obj.mode}}')")

    # Export with mesh and textures
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=True,  # Export selected objects only
        object_types={{'ARMATURE', 'MESH'}},  # Include armature and meshes
        bake_anim=True,  # Bake animation
        add_leaf_bones=False,
        mesh_smooth_type='FACE',  # Preserve mesh smoothing
        path_mode='COPY',  # Copy textures to output directory
        embed_textures=True,  # Embed textures in FBX
    )

def main():
    print("="*60)
    print("SMPL to FBX Retargeting")
    print("="*60)

    # Clear scene
    print("Clearing scene...")
    clear_scene()

    # Load SMPL data
    print(f"Loading SMPL data from: {smpl_data!r}")
    smpl_params = load_smpl_params("{smpl_data}")
    print(f"SMPL params loaded: {{list(smpl_params.keys())}}")

    # Import target FBX
    print(f"Importing FBX from: {fbx_input!r}")
    target_armature = import_fbx("{fbx_input}")
    print(f"Imported armature: {{target_armature.name}}")

    # Validate VRoid rig
    validate_vroid_rig(target_armature)

    # Retarget SMPL motion to target armature
    print(f"Retargeting SMPL motion to {{target_armature.name}} (rig_type: {rig_type})...")
    retarget_smpl_to_armature(target_armature, smpl_params, {fps}, "{rig_type}")

    # Export animated FBX
    print(f"Exporting to: {fbx_output!r}")
    export_fbx("{fbx_output}")

    print("="*60)
    print("Retargeting complete!")
    print("="*60)

if __name__ == "__main__":
    main()
'''
        return script


NODE_CLASS_MAPPINGS = {
    "SMPLToFBX": SMPLToFBX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SMPLToFBX": "SMPL to FBX Retargeting",
}
