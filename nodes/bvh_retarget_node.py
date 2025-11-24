"""
BVHtoFBX Node - Retarget BVH motion to rigged FBX/VRM characters using Blender
"""

from pathlib import Path
from typing import Dict, Tuple
import subprocess
import tempfile
import shutil
import os

from hmr4d.utils.pylogger import Log


class BVHtoFBX:
    """
    Retarget BVH motion data to a rigged FBX/VRM character using Blender's BVH Retargeter addon.
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
                    "default": "output/retargeted.fbx",
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
            Log.info("[BVHtoFBX] Starting BVH retargeting...")

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
                if character_path.suffix.lower() == ".vrm":
                    character_type = "vrm"
                else:
                    character_type = "fbx"

            Log.info(f"[BVHtoFBX] Character type: {character_type}")

            # Get Blender executable
            blender_exe = self._find_blender()
            if not blender_exe:
                raise RuntimeError(
                    "Blender not found. Please install Blender and ensure it's in your PATH."
                )

            Log.info(f"[BVHtoFBX] Using Blender: {blender_exe}")

            # Prepare output directory
            output_path = Path(output_path)
            
            # Security/Convention: If relative path doesn't start with output/, force it to output/
            if not output_path.is_absolute() and not str(output_path).startswith("output/") and not str(output_path).startswith("temp/"):
                output_path = Path("output") / output_path
            
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Ensure output has correct extension
            if output_format == "vrm" and output_path.suffix.lower() != ".vrm":
                output_path = output_path.with_suffix(".vrm")
            elif output_format == "fbx" and output_path.suffix.lower() != ".fbx":
                output_path = output_path.with_suffix(".fbx")

            # Create Blender retargeting script
            blender_script = self._create_blender_script(
                character_input=str(character_path.absolute()),
                bvh_input=str(Path(bvh_file).absolute()),
                output_file=str(output_path.absolute()),
                character_type=character_type,
                output_format=output_format,
            )

            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                script_path = Path(f.name)
                f.write(blender_script)

            Log.info(f"[BVHtoFBX] Created Blender script: {script_path}")

            try:
                # Run Blender in background mode
                cmd = [
                    str(blender_exe),
                    "--background",
                    "--python", str(script_path),
                ]

                Log.info(f"[BVHtoFBX] Running Blender command: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )

                if result.returncode != 0:
                    error_details = f"Blender Error (Code {result.returncode}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                    Log.error(f"[BVHtoFBX] {error_details}")
                    raise RuntimeError(error_details)

                Log.info(f"[BVHtoFBX] Blender output:\n{result.stdout}")

            finally:
                # Clean up temporary script
                script_path.unlink(missing_ok=True)

            if not output_path.exists():
                raise RuntimeError(f"Output file not created at: {output_path}\nCheck Blender logs above.")

            # Create info string
            num_frames = bvh_data.get("num_frames", 0)
            fps = bvh_data.get("fps", 30)

            info = (
                f"BVH Retargeting Complete\n"
                f"Character: {character_path.name}\n"
                f"BVH: {Path(bvh_file).name}\n"
                f"Output: {output_path.name}\n"
                f"Frames: {num_frames}\n"
                f"FPS: {fps}\n"
                f"Format: {output_format.upper()}\n"
            )

            Log.info("[BVHtoFBX] Retargeting complete!")
            return (str(output_path.absolute()), info)

        except Exception as e:
            error_msg = f"BVHtoFBX Failed:\n{str(e)}"
            Log.error(error_msg)
            return ("", error_msg)

    def _find_blender(self) -> Path:
        """Find Blender executable."""
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

        import shutil as sh
        system_blender = sh.which("blender")
        if system_blender:
            return Path(system_blender)

        return None

    def _create_blender_script(
        self,
        character_input: str,
        bvh_input: str,
        output_file: str,
        character_type: str,
        output_format: str,
    ) -> str:
        """
        Create Blender Python script for BVH retargeting.
        Includes built-in SMPL to VRM bone mapping.
        """
        # No manual backslash replacement needed if we trust Path.absolute() and python string handling
        # character_input = character_input.replace("\", "/") 

        # Bone Mapping: SMPL (Source) -> VRM/Mixamo (Target)
        bone_map = {
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

        # Use a raw string for the template to minimize escape issues
        script_template = r'''
import bpy
import sys
import traceback

print("[BVHtoFBX] Starting Blender retargeting script")

# =============================================================================
# RETARGETING DOCUMENTATION
# =============================================================================
#
# PROBLEM:
#   Retarget motion from SMPL BVH (from HMR4D/motion capture) to VRoid/VRM skeleton.
#   The two skeletons have very different bone orientations:
#   - SMPL Pelvis Y-axis points UP (+Z world)
#   - VRoid Hips Y-axis points DOWN (-Z world)
#   - Difference is ~158 degrees for pelvis, varies for other bones
#
# APPROACHES TRIED:
#
# 1. WORLD space COPY_ROTATION:
#    - Forces target bones to match source bone world orientations
#    - RESULT: Bone positions look correct BUT skeleton hierarchy breaks
#    - The spine ends up BELOW the hips because world rotation doesn't
#      account for different bone rest poses
#    - Mesh deforms incorrectly ("gremlin" distortion)
#
# 2. LOCAL space COPY_ROTATION:
#    - Copies local rotations directly from source to target
#    - RESULT: Skeleton hierarchy preserved (spine above hips)
#    - But bone orientations may not match exactly because rest poses differ
#    - This is the current approach - hierarchy is more important than exact matching
#
# 3. LOCAL space + 180° BVH armature rotation:
#    - Tried rotating BVH armature 180° to align forward directions
#    - RESULT: Didn't help - armature rotation doesn't affect bone local rotations
#    - Made things worse (character more tilted)
#
# 4. FBX export scale issues:
#    - apply_unit_scale=True converts meters to centimeters (100x larger)
#    - global_scale=0.01 makes character 2cm tall (too small)
#    - Default export settings seem to work best
#
# CURRENT STATUS:
#   - Using LOCAL space for rotations (preserves skeleton hierarchy)
#   - Using WORLD space for root location (preserves movement)
#   - Skeleton structure is correct but orientations may not match BVH exactly
#   - FBX scale may appear wrong in some viewers (Three.js expects different units)
#
# KNOWN ISSUES:
#   - Character orientation may not match BVH exactly
#   - FBX may appear giant/tiny in some viewers due to unit interpretation
#   - Mesh deformation quality depends on how well LOCAL rotations transfer
#
# =============================================================================

# Bone Mapping Dictionary - Will be replaced by Python
BONE_MAP = REPLACE_BONE_MAP

try:
    # Clear scene
    bpy.ops.wm.read_homefile(use_empty=True)
    print("[BVHtoFBX] Cleared scene")

    # Import character
    character_path = "REPLACE_CHARACTER_INPUT"
    character_type = "REPLACE_CHARACTER_TYPE"

    if character_type == "vrm":
        print("[BVHtoFBX] Importing VRM character...")
        try:
            bpy.ops.import_scene.vrm(filepath=character_path)
            print("[BVHtoFBX] VRM import successful")
        except AttributeError:
            try:
                bpy.ops.import_model.vrm(filepath=character_path)
                print("[BVHtoFBX] VRM import successful (legacy command)")
            except:
                print("[BVHtoFBX] ERROR: VRM addon not found. Please install VRM Addon for Blender.")
                sys.exit(1)
    else:
        print("[BVHtoFBX] Importing FBX character...")
        bpy.ops.import_scene.fbx(filepath=character_path)
        print("[BVHtoFBX] FBX import successful")

    # Find character armature
    char_armature = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            char_armature = obj
            break

    if not char_armature:
        print("[BVHtoFBX] ERROR: No armature found in character file")
        sys.exit(1)

    print(f"[BVHtoFBX] Found character armature: {char_armature.name}")
    print(f"[BVHtoFBX] Armature scale: {char_armature.scale[:]}")
    print(f"[BVHtoFBX] Armature location: {char_armature.location[:]}")

    # List all objects in scene
    print(f"[BVHtoFBX] All objects in scene:")
    for obj in bpy.data.objects:
        print(f"[BVHtoFBX]   - {obj.name} (type: {obj.type}, parent: {obj.parent.name if obj.parent else 'None'})")

    # Print character bone names for debugging
    print(f"[BVHtoFBX] Character Armature Bones: {[b.name for b in char_armature.data.bones]}")

    # Ensure we are in Object Mode
    if bpy.context.object:
        bpy.ops.object.mode_set(mode='OBJECT')

    # Load BVH (Source Motion)
    bvh_path = "REPLACE_BVH_INPUT"
    print(f"[BVHtoFBX] Loading BVH animation: {bvh_path}")
    # Import BVH with global_scale=1.0 (SMPL BVH output is already in meters)
    # Default settings (axis_forward='-Z', axis_up='Y') convert to Blender's Z-up coordinate system
    # This matches VRoid which also uses Z-up
    bpy.ops.import_anim.bvh(filepath=bvh_path, global_scale=1.0)

    # Find BVH armature
    bvh_armature = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE' and obj != char_armature:
            bvh_armature = obj
            break

    if not bvh_armature:
        print("[BVHtoFBX] ERROR: BVH armature not found after import")
        sys.exit(1)

    print(f"[BVHtoFBX] Found BVH armature: {bvh_armature.name}")
    print(f"[BVHtoFBX] BVH Armature Bones: {[b.name for b in bvh_armature.data.bones]}")

    # --- RETARGETING LOGIC ---
    print("[BVHtoFBX] Starting retargeting...")

    import mathutils
    from mathutils import Matrix, Quaternion, Vector
    from math import radians, degrees, atan2, acos, sqrt

    # Auto-detect bone naming convention (Standard VRM vs VRoid FBX)
    bone_names = char_armature.pose.bones.keys()
    is_vroid = any("J_Bip_C_Hips" in b for b in bone_names)

    if is_vroid:
        print("[BVHtoFBX] Detected VRoid bone naming convention (J_Bip_...)")
        vroid_map = {
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
        # Update BONE_MAP with VRoid names if present
        new_map = {}
        for smpl, vrm in BONE_MAP.items():
            if vrm in vroid_map:
                new_map[smpl] = vroid_map[vrm]
            else:
                new_map[smpl] = vrm
        BONE_MAP = new_map

    # ===========================================
    # STEP 1: Skip T-pose normalization
    # ===========================================
    # We don't modify the rest pose because:
    # - armature_apply() breaks mesh skinning (bind matrices don't update)
    # - LOCAL space constraints work with existing rest poses
    # - The skeleton hierarchy is preserved this way
    print("[BVHtoFBX] Step 1: Skipping T-pose normalization (preserving original rest pose)")

    # ===========================================
    # STEP 2: Set up bone mapping
    # ===========================================
    print("[BVHtoFBX] Step 2: Setting up bone mapping...")

    bpy.context.view_layer.objects.active = char_armature
    bpy.ops.object.mode_set(mode='POSE')

    # Build valid bone mapping
    valid_mappings = []
    for smpl_bone, vrm_bone in BONE_MAP.items():
        if vrm_bone not in char_armature.pose.bones:
            print(f"[BVHtoFBX] WARNING: Target bone '{vrm_bone}' not found. Skipping.")
            continue
        if smpl_bone not in bvh_armature.pose.bones:
            print(f"[BVHtoFBX] WARNING: Source bone '{smpl_bone}' not found. Skipping.")
            continue
        valid_mappings.append((smpl_bone, vrm_bone))
        print(f"[BVHtoFBX] Mapping: '{smpl_bone}' -> '{vrm_bone}'")

    print(f"[BVHtoFBX] Total valid bone mappings: {len(valid_mappings)}")

    if len(valid_mappings) == 0:
        print("[BVHtoFBX] ERROR: No valid bone mappings found.")
        sys.exit(1)

    # ===========================================
    # STEP 3: Calculate scale ratio between skeletons
    # ===========================================
    print("[BVHtoFBX] Step 3: Calculating skeleton scale ratio...")

    # Calculate height of each skeleton (from hips to head)
    def get_skeleton_height(armature, hips_name, head_name):
        if hips_name in armature.data.bones and head_name in armature.data.bones:
            hips = armature.data.bones[hips_name]
            head = armature.data.bones[head_name]
            return (head.head_local - hips.head_local).length
        return 1.0

    # Get BVH skeleton height
    bvh_height = get_skeleton_height(bvh_armature, 'Pelvis', 'Head')

    # Get target skeleton height
    if is_vroid:
        target_height = get_skeleton_height(char_armature, 'J_Bip_C_Hips', 'J_Bip_C_Head')
    else:
        target_height = get_skeleton_height(char_armature, 'Hips', 'Head')

    # Calculate scale ratio (target / source)
    if bvh_height > 0.01:
        scale_ratio = target_height / bvh_height
    else:
        scale_ratio = 1.0

    print(f"[BVHtoFBX] BVH skeleton height: {bvh_height:.3f}m")
    print(f"[BVHtoFBX] Target skeleton height: {target_height:.3f}m")
    print(f"[BVHtoFBX] Scale ratio: {scale_ratio:.3f}")

    # Scale the BVH armature to match target skeleton size
    # This allows COPY_LOCATION constraint to work without manual scaling
    bvh_armature.scale = (scale_ratio, scale_ratio, scale_ratio)
    bpy.context.view_layer.update()
    print(f"[BVHtoFBX] Scaled BVH armature by {scale_ratio:.3f}")

    # ===========================================
    # STEP 4: Apply retargeting with constraints
    # ===========================================
    print("[BVHtoFBX] Step 4: Applying animation via constraints...")

    # CONSTRAINT SPACE CHOICE (see documentation at top of file):
    # - ROTATION: LOCAL space - preserves skeleton hierarchy
    #   (WORLD space was tried but breaks hierarchy - spine ends up below hips)
    # - LOCATION: WORLD space for root only - preserves world movement
    constraints_applied = 0

    for smpl_bone, vrm_bone in valid_mappings:
        p_bone = char_armature.pose.bones[vrm_bone]

        # LOCAL space rotation - preserves parent-child bone relationships
        const = p_bone.constraints.new('COPY_ROTATION')
        const.target = bvh_armature
        const.subtarget = smpl_bone
        const.mix_mode = 'REPLACE'
        const.owner_space = 'LOCAL'
        const.target_space = 'LOCAL'

        constraints_applied += 1

        # WORLD space location for root bone only
        if smpl_bone == 'Pelvis':
            loc_const = p_bone.constraints.new('COPY_LOCATION')
            loc_const.target = bvh_armature
            loc_const.subtarget = smpl_bone
            loc_const.owner_space = 'WORLD'
            loc_const.target_space = 'WORLD'
            print(f"[BVHtoFBX] Added COPY_LOCATION constraint for root bone: {smpl_bone} -> {vrm_bone}")

    print(f"[BVHtoFBX] Applied {constraints_applied} constraints")

    # ===========================================
    # STEP 5: Bake animation (constraints handle everything)
    # ===========================================
    print("[BVHtoFBX] Step 5: Baking animation...")

    action = bvh_armature.animation_data.action
    frame_start = int(action.frame_range[0])
    frame_end = int(action.frame_range[1])
    print(f"[BVHtoFBX] Animation frames: {frame_start} to {frame_end}")

    # Select all pose bones for baking
    bpy.ops.pose.select_all(action='SELECT')

    # Bake with visual keying - converts constraint results to keyframes
    # After baking, constraints are removed and animation is stored as local rotations
    bpy.ops.nla.bake(
        frame_start=frame_start,
        frame_end=frame_end,
        only_selected=True,
        visual_keying=True,
        clear_constraints=True,
        use_current_action=False,
        bake_types={'POSE'}
    )

    print("[BVHtoFBX] Baking complete")

    # ===========================================
    # STEP 6: Compensate for foot height offset
    # ===========================================
    # Different skeleton proportions cause feet to be at different heights.
    # We adjust by moving the armature to align minimum foot heights.
    print("[BVHtoFBX] Step 6: Calculating foot height compensation...")

    # Find foot bones
    bvh_l_foot = 'L_Ankle'
    bvh_r_foot = 'R_Ankle'
    if is_vroid:
        target_l_foot = 'J_Bip_L_Foot'
        target_r_foot = 'J_Bip_R_Foot'
        target_root = 'J_Bip_C_Hips'
    else:
        target_l_foot = 'LeftFoot'
        target_r_foot = 'RightFoot'
        target_root = 'Hips'

    # Sample foot heights at multiple frames to find ground level
    bvh_min_foot_z = float('inf')
    target_min_foot_z = float('inf')

    sample_frames = list(range(frame_start, frame_end + 1, max(1, (frame_end - frame_start) // 20)))
    for frame in sample_frames:
        bpy.context.scene.frame_set(frame)

        # BVH foot heights
        if bvh_l_foot in bvh_armature.pose.bones:
            bvh_l_z = (bvh_armature.matrix_world @ bvh_armature.pose.bones[bvh_l_foot].matrix).translation.z
            bvh_min_foot_z = min(bvh_min_foot_z, bvh_l_z)
        if bvh_r_foot in bvh_armature.pose.bones:
            bvh_r_z = (bvh_armature.matrix_world @ bvh_armature.pose.bones[bvh_r_foot].matrix).translation.z
            bvh_min_foot_z = min(bvh_min_foot_z, bvh_r_z)

        # Target foot heights
        if target_l_foot in char_armature.pose.bones:
            target_l_z = (char_armature.matrix_world @ char_armature.pose.bones[target_l_foot].matrix).translation.z
            target_min_foot_z = min(target_min_foot_z, target_l_z)
        if target_r_foot in char_armature.pose.bones:
            target_r_z = (char_armature.matrix_world @ char_armature.pose.bones[target_r_foot].matrix).translation.z
            target_min_foot_z = min(target_min_foot_z, target_r_z)

    foot_height_offset = target_min_foot_z - bvh_min_foot_z
    print(f"[BVHtoFBX] BVH min foot Z: {bvh_min_foot_z:.3f}m")
    print(f"[BVHtoFBX] Target min foot Z: {target_min_foot_z:.3f}m")
    print(f"[BVHtoFBX] Foot height offset: {foot_height_offset:.3f}m")

    # Move armature to compensate for height difference
    if abs(foot_height_offset) > 0.05:
        print(f"[BVHtoFBX] Applying height compensation: lowering armature by {foot_height_offset:.3f}m")
        char_armature.location.z -= foot_height_offset
        bpy.context.view_layer.update()
        print("[BVHtoFBX] Height compensation applied")

    # Delete BVH armature
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.data.objects.remove(bvh_armature, do_unlink=True)

    # Export
    output_path = "REPLACE_OUTPUT_FILE"
    output_format = "REPLACE_OUTPUT_FORMAT"

    bpy.ops.object.select_all(action='DESELECT')
    char_armature.select_set(True)
    bpy.context.view_layer.objects.active = char_armature

    # Find all meshes that use this armature (via modifier or parenting)
    mesh_count = 0
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            # Check if parented to armature
            if obj.parent == char_armature:
                obj.select_set(True)
                mesh_count += 1
                print(f"[BVHtoFBX] Selected mesh (child): {obj.name}")
            else:
                # Check if has armature modifier pointing to our armature
                for mod in obj.modifiers:
                    if mod.type == 'ARMATURE' and mod.object == char_armature:
                        obj.select_set(True)
                        mesh_count += 1
                        print(f"[BVHtoFBX] Selected mesh (modifier): {obj.name}")
                        break

    print(f"[BVHtoFBX] Total meshes selected for export: {mesh_count}")

    if output_format == "vrm":
        print("[BVHtoFBX] Exporting as VRM...")
        try:
            bpy.ops.export_scene.vrm(filepath=output_path, export_fbx_hdr_emb=False) 
            print("[BVHtoFBX] VRM export successful")
        except AttributeError:
            print("[BVHtoFBX] ERROR: VRM export failed. Falling back to FBX.")
            output_path = output_path.replace(".vrm", ".fbx")
            bpy.ops.export_scene.fbx(filepath=output_path, use_selection=True, bake_anim=True, add_leaf_bones=False)
            print("[BVHtoFBX] FBX export successful (fallback)")
    else:
        print("[BVHtoFBX] Exporting as FBX...")
        bpy.ops.export_scene.fbx(
            filepath=output_path,
            use_selection=True,
            bake_anim=True,
            add_leaf_bones=False,
            # Texture settings - embed textures in FBX for portability
            path_mode='COPY',
            embed_textures=True,
        )

    print(f"[BVHtoFBX] Output saved to: {output_path}")
    print("[BVHtoFBX] Retargeting complete!")

except Exception as e:
    print(f"[BVHtoFBX] ERROR: {str(e)}")
    traceback.print_exc()
    sys.exit(1)
'''
        
        # Inject variables into script using simple replace to avoid f-string syntax errors
        script = script_template.replace("REPLACE_BONE_MAP", str(bone_map))
        script = script.replace("REPLACE_CHARACTER_INPUT", character_input)
        script = script.replace("REPLACE_CHARACTER_TYPE", character_type)
        script = script.replace("REPLACE_BVH_INPUT", bvh_input)
        script = script.replace("REPLACE_OUTPUT_FILE", output_file)
        script = script.replace("REPLACE_OUTPUT_FORMAT", output_format)

        return script


NODE_CLASS_MAPPINGS = {
    "BVHtoFBX": BVHtoFBX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BVHtoFBX": "BVH to FBX Retargeter",
}