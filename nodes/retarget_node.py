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
            with open(script_path, 'w', encoding='utf-8') as f:
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
        """Create Python script for Blender to execute retargeting.

        Uses BVH workflow with Rokoko + horizontal root motion.
        This approach produces good rotations/poses without mesh twisting.
        """

        # Get the addons directory path
        blender_exe = self._find_blender()
        if blender_exe:
            addons_dir = str(blender_exe.parent / "4.2" / "scripts" / "addons")
        else:
            addons_dir = ""

        script = f'''
import sys
import os

# Add addons directory to path for Rokoko
addons_dir = "{addons_dir}"
if addons_dir and addons_dir not in sys.path:
    sys.path.insert(0, addons_dir)

import bpy
import numpy as np
import mathutils

# ============================================================
# ROKOKO ADDON SETUP
# ============================================================
def setup_rokoko():
    """Load and register Rokoko addon."""
    try:
        import rokoko_studio_live_blender
        rokoko_studio_live_blender.register()
        print("Rokoko Studio Live addon loaded successfully!")
        return True
    except Exception as e:
        print(f"Warning: Could not load Rokoko addon: {{e}}")
        return False

# ============================================================
# SMPL TO BVH CONVERSION
# ============================================================
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

def axis_angle_to_euler_zxy(axis_angle):
    \"\"\"Convert axis-angle to ZXY Euler angles (BVH standard).\"\"\"
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

def smpl_to_bvh(smpl_params, output_path, fps=30):
    \"\"\"Convert SMPL parameters to BVH file.\"\"\"
    body_pose = smpl_params.get('body_pose')
    global_orient = smpl_params.get('global_orient')
    transl = smpl_params.get('transl')

    if body_pose is None:
        raise ValueError("No body_pose in SMPL params")

    num_frames = body_pose.shape[0]
    body_pose = body_pose.reshape(num_frames, 21, 3)

    # Build BVH header
    lines = ["HIERARCHY", "ROOT Pelvis", "{{", "\\tOFFSET 0.0 0.0 0.0",
             "\\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"]

    def add_joint(idx, depth):
        indent = "\\t" * depth
        name = SMPL_BONE_NAMES[idx]
        offset = SMPL_OFFSETS[idx]
        children = [i for i, p in enumerate(SMPL_PARENTS) if p == idx]

        if children:
            for child_idx in children:
                child_name = SMPL_BONE_NAMES[child_idx]
                child_offset = SMPL_OFFSETS[child_idx]
                lines.append(f"{{indent}}JOINT {{child_name}}")
                lines.append(f"{{indent}}{{{{")
                lines.append(f"{{indent}}\\tOFFSET {{child_offset[0]*10:.4f}} {{child_offset[1]*10:.4f}} {{child_offset[2]*10:.4f}}")
                lines.append(f"{{indent}}\\tCHANNELS 3 Zrotation Xrotation Yrotation")
                add_joint(child_idx, depth + 1)
                lines.append(f"{{indent}}}}}}")
        else:
            lines.append(f"{{indent}}End Site")
            lines.append(f"{{indent}}{{{{")
            lines.append(f"{{indent}}\\tOFFSET {{offset[0]*5:.4f}} {{offset[1]*5:.4f}} {{offset[2]*5:.4f}}")
            lines.append(f"{{indent}}}}}}")

    add_joint(0, 1)
    lines.append("}}")

    # Motion section
    lines.append("MOTION")
    lines.append(f"Frames: {{num_frames}}")
    lines.append(f"Frame Time: {{1.0/fps:.6f}}")

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
            euler = axis_angle_to_euler_zxy(global_orient[frame])
            values.extend(euler)
        else:
            values.extend([0, 0, 0])

        # Body pose rotations
        for joint_idx in range(21):
            euler = axis_angle_to_euler_zxy(body_pose[frame, joint_idx])
            values.extend(euler)

        lines.append(" ".join(f"{{v:.4f}}" for v in values))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\\n".join(lines))

    print(f"Created BVH: {{output_path}} ({{num_frames}} frames)")
    return output_path

# ============================================================
# MAIN FUNCTIONS
# ============================================================
def clear_scene():
    \"\"\"Clear default scene.\"\"\"
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def load_smpl_params(smpl_path):
    \"\"\"Load SMPL parameters from npz file.\"\"\"
    data = np.load(smpl_path)
    return {{key: data[key] for key in data.files}}

def fix_bone_list_duplicates():
    \"\"\"Remove duplicate target bones from the Rokoko bone list.\"\"\"
    bone_list = bpy.context.scene.rsl_retargeting_bone_list
    seen_targets = {{}}
    to_clear = []
    for i, item in enumerate(bone_list):
        if item.bone_name_target and item.bone_name_target in seen_targets:
            to_clear.append(i)
        elif item.bone_name_target:
            seen_targets[item.bone_name_target] = i
    for i in to_clear:
        bone_list[i].bone_name_target = ""

def main():
    print("="*60)
    print("SMPL to FBX Retargeting (BVH + Rokoko + Horizontal Motion)")
    print("="*60)

    # Setup Rokoko addon
    rokoko_available = setup_rokoko()

    # Clear scene
    clear_scene()

    # Load SMPL data
    smpl_path = "{smpl_data}"
    print(f"\\nLoading SMPL data from: {{smpl_path}}")
    smpl_params = load_smpl_params(smpl_path)
    print(f"Loaded: {{list(smpl_params.keys())}}")

    # Convert SMPL to BVH
    import tempfile
    bvh_path = tempfile.gettempdir() + "/smpl_temp.bvh"
    print(f"\\nConverting SMPL to BVH: {{bvh_path}}")
    smpl_to_bvh(smpl_params, bvh_path, fps={fps})

    # Import BVH
    print("\\nImporting BVH...")
    bpy.ops.import_anim.bvh(filepath=bvh_path)
    bvh_armature = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE'][0]

    # Store horizontal root motion from BVH (X, Y only)
    print("\\nStoring BVH horizontal root motion...")
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
    fbx_path = "{fbx_input}"
    print(f"\\nImporting target FBX: {{fbx_path}}")
    bpy.ops.import_scene.fbx(filepath=fbx_path, automatic_bone_orientation=True)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']
    target_armature = [a for a in armatures if a != bvh_armature][0]

    # Retarget with Rokoko (auto_scaling OFF for better rotations)
    if rokoko_available:
        print("\\nRetargeting with Rokoko...")
        bpy.context.scene.rsl_retargeting_auto_scaling = False
        bpy.context.scene.rsl_retargeting_armature_source = bvh_armature
        bpy.context.scene.rsl_retargeting_armature_target = target_armature
        bpy.ops.rsl.build_bone_list()
        fix_bone_list_duplicates()
        bpy.ops.rsl.retarget_animation()
    else:
        print("\\nWARNING: Rokoko addon not available!")

    # Delete BVH armature
    bpy.data.objects.remove(bvh_armature, do_unlink=True)

    # Apply HORIZONTAL root motion only (X, Y - no vertical adjustment)
    print("\\nApplying horizontal root motion...")
    bpy.context.view_layer.objects.active = target_armature
    bpy.ops.object.mode_set(mode='POSE')

    hips = target_armature.pose.bones.get("mixamorig:Hips")
    if not hips:
        # Try other common naming conventions
        for name in ["Hips", "pelvis", "Pelvis", "hip", "Root"]:
            hips = target_armature.pose.bones.get(name)
            if hips:
                break

    if hips and len(original_xy) > 0:
        # Get frame 1 reference
        bpy.context.scene.frame_set(1)
        ref_hips_world = target_armature.matrix_world @ hips.head
        ref_xy = original_xy[0]

        for frame in range(1, min(num_frames + 1, len(original_xy) + 1)):
            bpy.context.scene.frame_set(frame)

            current_world = target_armature.matrix_world @ hips.head

            # Target XY relative to frame 1
            target_x = original_xy[frame - 1][0] - ref_xy[0]
            target_y = original_xy[frame - 1][1] - ref_xy[1]

            # Current XY relative to frame 1
            current_x = current_world.x - ref_hips_world.x
            current_y = current_world.y - ref_hips_world.y

            # Delta needed (horizontal only)
            delta_world = mathutils.Vector((target_x - current_x, target_y - current_y, 0))

            # Convert to bone local space
            bone_matrix_inv = hips.bone.matrix_local.inverted()
            delta_local = bone_matrix_inv.to_3x3() @ delta_world

            hips.location = hips.location + delta_local
            hips.keyframe_insert(data_path="location", frame=frame)

    bpy.ops.object.mode_set(mode='OBJECT')

    # Export
    output_path = "{fbx_output}"
    print(f"\\nExporting to: {{output_path}}")

    bpy.ops.object.select_all(action='DESELECT')
    target_armature.select_set(True)
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.parent == target_armature:
            obj.select_set(True)

    bpy.context.view_layer.objects.active = target_armature

    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=True,
        object_types={{'ARMATURE', 'MESH'}},
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        add_leaf_bones=False,
    )

    print("\\n" + "="*60)
    print("RETARGETING COMPLETE!")
    print(f"Output: {{output_path}}")
    print(f"Frames: {{num_frames}}")
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
