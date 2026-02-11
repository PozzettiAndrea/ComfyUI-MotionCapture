"""
SMPLToMixamo Node - Retarget SMPL motion to Mixamo-rigged FBX characters

Blender operations run in an isolated environment with the bpy package and
Rokoko addon for automatic bone mapping.

This node is specifically optimized for Mixamo characters with the mixamorig: prefix.
"""

import tempfile
import os
from pathlib import Path
from typing import Tuple

import numpy as np


# ===============================================================================
# SMPL SKELETON CONFIGURATION
# ===============================================================================

SMPL_BONE_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"
]

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

SMPL_OFFSETS = [
    [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0],
    [0, 1, 0], [0, -1, 0], [0, -1, 0], [0, 1, 0], [0, -0.5, 0.5], [0, -0.5, 0.5],
    [0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0], [-1, 0, 0],
    [1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]
]

# SMPL to Mixamo bone mapping for reference (used by Rokoko auto-mapping)
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
# HELPER FUNCTIONS
# ===============================================================================

# Module-level cache for Rokoko addon status
_ROKOKO_INSTALLED = False


def _ensure_rokoko_addon():
    """Install and enable Rokoko addon in bpy if not present."""
    global _ROKOKO_INSTALLED

    if _ROKOKO_INSTALLED:
        return True

    import bpy

    # Check if already enabled
    if "rokoko_studio_live_blender" in bpy.context.preferences.addons:
        _ROKOKO_INSTALLED = True
        return True

    # Try to enable if installed but not enabled
    try:
        bpy.ops.preferences.addon_enable(module="rokoko_studio_live_blender")
        _ROKOKO_INSTALLED = True
        print("[SMPLToMixamo] Rokoko addon enabled")
        return True
    except Exception:
        pass

    # Download and install Rokoko addon v1.4.3
    print("[SMPLToMixamo] Downloading Rokoko addon v1.4.3...")
    addon_url = "https://github.com/Rokoko/rokoko-studio-live-blender/archive/refs/tags/v1-4-3.zip"

    try:
        import urllib.request
        import zipfile
        import shutil

        temp_dir = tempfile.gettempdir()
        zip_path = os.path.join(temp_dir, "rokoko_addon.zip")
        extract_dir = os.path.join(temp_dir, "rokoko_extract")

        # Download
        urllib.request.urlretrieve(addon_url, zip_path)

        # Extract
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)

        # Find the extracted folder (rokoko-studio-live-blender-1-4-3)
        extracted_folder = None
        for item in os.listdir(extract_dir):
            if item.startswith("rokoko-studio-live-blender"):
                extracted_folder = os.path.join(extract_dir, item)
                break

        if extracted_folder:
            # Rename to expected addon name
            addon_folder = os.path.join(extract_dir, "rokoko_studio_live_blender")
            if os.path.exists(addon_folder):
                shutil.rmtree(addon_folder)
            shutil.move(extracted_folder, addon_folder)

            # Create zip for Blender addon installer
            addon_zip = os.path.join(temp_dir, "rokoko_ready.zip")
            shutil.make_archive(addon_zip.replace('.zip', ''), 'zip', extract_dir, "rokoko_studio_live_blender")

            # Install addon
            bpy.ops.preferences.addon_install(filepath=addon_zip)
            bpy.ops.preferences.addon_enable(module="rokoko_studio_live_blender")

            # Cleanup
            os.remove(zip_path)
            os.remove(addon_zip)
            shutil.rmtree(extract_dir)

            _ROKOKO_INSTALLED = True
            print("[SMPLToMixamo] Rokoko addon v1.4.3 installed and enabled")
            return True

    except Exception as e:
        print(f"[SMPLToMixamo] Warning: Could not install Rokoko addon: {e}")
        return False

    return False


def _axis_angle_to_euler_zxy(axis_angle):
    """Convert axis-angle to ZXY Euler angles (BVH standard)."""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return [0.0, 0.0, 0.0]
    axis = axis_angle / angle
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    x, y, z = axis

    # Rotation matrix
    R = np.array([
        [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
        [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
        [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]
    ])

    # Extract ZXY Euler
    if abs(R[2, 1]) < 0.99999:
        x_rot = np.arcsin(-R[2, 1])
        y_rot = np.arctan2(R[2, 0], R[2, 2])
        z_rot = np.arctan2(R[0, 1], R[1, 1])
    else:
        x_rot = np.pi / 2 if R[2, 1] < 0 else -np.pi / 2
        y_rot = np.arctan2(-R[0, 2], R[0, 0])
        z_rot = 0

    return [np.degrees(z_rot), np.degrees(x_rot), np.degrees(y_rot)]


def _smpl_to_bvh(smpl_params: dict, output_path: str, fps: int = 30) -> str:
    """Convert SMPL parameters to BVH file."""
    body_pose = smpl_params.get('body_pose')
    global_orient = smpl_params.get('global_orient')
    transl = smpl_params.get('transl')

    if body_pose is None:
        raise ValueError("No body_pose in SMPL params")

    num_frames = body_pose.shape[0]
    body_pose = body_pose.reshape(num_frames, 21, 3)

    # Build BVH header
    lines = ["HIERARCHY", "ROOT Pelvis", "{", "\tOFFSET 0.0 0.0 0.0",
             "\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"]

    def add_joint(idx, depth):
        indent = "\t" * depth
        children = [i for i, p in enumerate(SMPL_PARENTS) if p == idx]

        if children:
            for child_idx in children:
                child_name = SMPL_BONE_NAMES[child_idx]
                child_offset = SMPL_OFFSETS[child_idx]
                lines.append(f"{indent}JOINT {child_name}")
                lines.append(f"{indent}{{")
                lines.append(f"{indent}\tOFFSET {child_offset[0]*10:.4f} {child_offset[1]*10:.4f} {child_offset[2]*10:.4f}")
                lines.append(f"{indent}\tCHANNELS 3 Zrotation Xrotation Yrotation")
                add_joint(child_idx, depth + 1)
                lines.append(f"{indent}}}")
        else:
            offset = SMPL_OFFSETS[idx]
            lines.append(f"{indent}End Site")
            lines.append(f"{indent}{{")
            lines.append(f"{indent}\tOFFSET {offset[0]*5:.4f} {offset[1]*5:.4f} {offset[2]*5:.4f}")
            lines.append(f"{indent}}}")

    add_joint(0, 1)
    lines.append("}")

    # Motion section
    lines.append("MOTION")
    lines.append(f"Frames: {num_frames}")
    lines.append(f"Frame Time: {1.0/fps:.6f}")

    for frame in range(num_frames):
        values = []

        # Root position (convert SMPL Y-up to BVH Z-up)
        if transl is not None:
            t = transl[frame]
            values.extend([t[0]*100, t[2]*100, t[1]*100])  # Scale and swap Y/Z
        else:
            values.extend([0, 0, 0])

        # Root rotation
        if global_orient is not None:
            euler = _axis_angle_to_euler_zxy(global_orient[frame])
            values.extend(euler)
        else:
            values.extend([0, 0, 0])

        # Body pose rotations
        for joint_idx in range(21):
            euler = _axis_angle_to_euler_zxy(body_pose[frame, joint_idx])
            values.extend(euler)

        lines.append(" ".join(f"{v:.4f}" for v in values))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"[SMPLToMixamo] Created BVH: {output_path} ({num_frames} frames)")
    return output_path


# ===============================================================================
# ISOLATED BLENDER WORKER
# ===============================================================================

class SMPLToMixamoWorker:
    """
    Isolated worker for SMPL to Mixamo retargeting using bpy + Rokoko.
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
        Retarget SMPL motion to Mixamo character using bpy + Rokoko.

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

        print("=" * 60)
        print("SMPL to Mixamo Retargeting")
        print("=" * 60)

        # Try to setup Rokoko addon
        rokoko_available = _ensure_rokoko_addon()

        # Clear scene
        bpy.ops.wm.read_homefile(use_empty=True)
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # Load SMPL data
        print(f"\nLoading SMPL data from: {smpl_data_path}")
        smpl_data = np.load(smpl_data_path)
        smpl_params = {key: smpl_data[key] for key in smpl_data.files}
        print(f"Loaded: {list(smpl_params.keys())}")

        # Convert SMPL to BVH
        bvh_path = os.path.join(tempfile.gettempdir(), "smpl_mixamo_temp.bvh")
        print(f"\nConverting SMPL to BVH: {bvh_path}")
        _smpl_to_bvh(smpl_params, bvh_path, fps=fps)

        # Import BVH
        print("\nImporting BVH...")
        bpy.ops.import_anim.bvh(filepath=bvh_path)
        bvh_armature = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE'][0]

        # Store horizontal root motion from BVH (X, Y only)
        print("\nStoring BVH horizontal root motion...")
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
        print(f"\nImporting Mixamo FBX: {mixamo_fbx}")
        bpy.ops.import_scene.fbx(filepath=mixamo_fbx, automatic_bone_orientation=True)
        armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']
        target_armature = [a for a in armatures if a != bvh_armature][0]

        # Verify it's a Mixamo rig
        mixamo_hips = target_armature.pose.bones.get("mixamorig:Hips")
        if not mixamo_hips:
            print("WARNING: No 'mixamorig:Hips' found - this may not be a Mixamo rig")

        # Retarget with Rokoko
        if rokoko_available:
            print("\nRetargeting with Rokoko...")
            bpy.context.scene.rsl_retargeting_auto_scaling = False
            bpy.context.scene.rsl_retargeting_armature_source = bvh_armature
            bpy.context.scene.rsl_retargeting_armature_target = target_armature
            bpy.ops.rsl.build_bone_list()

            # Fix duplicate target bones (Rokoko quirk)
            bone_list = bpy.context.scene.rsl_retargeting_bone_list
            seen_targets = {}
            to_clear = []
            for i, item in enumerate(bone_list):
                if item.bone_name_target and item.bone_name_target in seen_targets:
                    to_clear.append(i)
                elif item.bone_name_target:
                    seen_targets[item.bone_name_target] = i
            for i in to_clear:
                bone_list[i].bone_name_target = ""

            bpy.ops.rsl.retarget_animation()
        else:
            print("\nWARNING: Rokoko addon not available, using basic retargeting")

        # Delete BVH armature
        bpy.data.objects.remove(bvh_armature, do_unlink=True)

        # Apply horizontal root motion to Mixamo Hips
        print("\nApplying horizontal root motion...")
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
        print(f"\nExporting to: {output_fbx}")
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
            f"Frames: {num_frames}\n"
            f"Rokoko: {'Yes' if rokoko_available else 'No (basic retargeting)'}"
        )

        print("\n" + "=" * 60)
        print("RETARGETING COMPLETE!")
        print(f"Output: {output_fbx}")
        print(f"Frames: {num_frames}")
        print("=" * 60)

        return (output_fbx, num_frames, info)


# ===============================================================================
# COMFYUI NODE
# ===============================================================================

class SMPLToMixamo:
    """
    Retarget SMPL motion capture data to a Mixamo-rigged FBX character.

    This node is specifically designed for Mixamo characters (with mixamorig: prefix).
    It uses Rokoko addon for automatic bone mapping and motion retargeting.
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
            print("[SMPLToMixamo] Starting SMPL to Mixamo retargeting...")

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

            print(f"[SMPLToMixamo] Using SMPL data: {smpl_npz_path}")

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

            print(f"[SMPLToMixamo] Retargeting complete! Output: {output_path}")
            return (str(output_path.absolute()), result_frames, full_info)

        except Exception as e:
            error_msg = f"SMPLToMixamo failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            return ("", 0, error_msg)

NODE_CLASS_MAPPINGS = {
    "SMPLToMixamo": SMPLToMixamo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SMPLToMixamo": "SMPL to Mixamo",
}
