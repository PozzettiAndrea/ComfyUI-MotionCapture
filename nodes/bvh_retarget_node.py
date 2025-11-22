"""
BVHtoFBX Node - Retarget BVH motion to rigged FBX/VRM characters using Blender
"""

from pathlib import Path
from typing import Dict, Tuple
import subprocess
import tempfile
import shutil

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
        """
        Retarget BVH motion to FBX/VRM character.

        Args:
            bvh_data: BVH data dictionary from SMPLtoBVH node
            character_path: Path to input character file (VRM or FBX)
            output_path: Path to save retargeted file
            character_type: Type of character file (auto-detect or specific)
            output_format: Output format (fbx or vrm)

        Returns:
            Tuple of (output_path, info_string)
        """
        try:
            Log.info("[BVHtoFBX] Starting BVH retargeting...")

            # Validate inputs
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
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Ensure output has correct extension
            if output_format == "vrm" and output_path.suffix.lower() != ".vrm":
                output_path = output_path.with_suffix('.vrm')
            elif output_format == "fbx" and output_path.suffix.lower() != ".fbx":
                output_path = output_path.with_suffix('.fbx')

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

                Log.info("[BVHtoFBX] Running Blender retargeting...")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )

                if result.returncode != 0:
                    Log.error(f"[BVHtoFBX] Blender error:\n{result.stderr}")
                    raise RuntimeError(f"Blender retargeting failed: {result.stderr}")

                Log.info(f"[BVHtoFBX] Blender output:\n{result.stdout}")

            finally:
                # Clean up temporary script
                script_path.unlink(missing_ok=True)

            if not output_path.exists():
                raise RuntimeError(f"Output file not created: {output_path}")

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
            error_msg = f"BVHtoFBX failed: {str(e)}"
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

        Args:
            character_input: Path to character file (VRM or FBX)
            bvh_input: Path to BVH file
            output_file: Path to output file
            character_type: Type of character ("vrm" or "fbx")
            output_format: Output format ("vrm" or "fbx")

        Returns:
            Blender Python script as string
        """
        # Escape paths for Python string
        character_input = character_input.replace("\\", "\\\\")
        bvh_input = bvh_input.replace("\\", "\\\\")
        output_file = output_file.replace("\\", "\\\\")

        script = f'''
import bpy
import sys
import traceback

print("[BVHtoFBX] Starting Blender retargeting script")

try:
    # Clear scene
    bpy.ops.wm.read_homefile(use_empty=True)
    print("[BVHtoFBX] Cleared scene")

    # Import character
    character_path = "{character_input}"
    character_type = "{character_type}"

    if character_type == "vrm":
        print("[BVHtoFBX] Importing VRM character...")
        try:
            bpy.ops.import_scene.vrm(filepath=character_path)
            print("[BVHtoFBX] VRM import successful")
        except AttributeError:
            print("[BVHtoFBX] ERROR: VRM addon not found. Please install VRM Addon for Blender.")
            print("[BVHtoFBX] Download from: https://github.com/saturday06/VRM-Addon-for-Blender")
            sys.exit(1)
    else:
        print("[BVHtoFBX] Importing FBX character...")
        bpy.ops.import_scene.fbx(filepath=character_path)
        print("[BVHtoFBX] FBX import successful")

    # Find armature
    armature = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break

    if not armature:
        print("[BVHtoFBX] ERROR: No armature found in character file")
        sys.exit(1)

    print(f"[BVHtoFBX] Found armature: {{armature.name}}")

    # Select armature
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    # Load and retarget BVH
    bvh_path = "{bvh_input}"
    print(f"[BVHtoFBX] Loading BVH animation: {{bvh_path}}")

    try:
        # Try using BVH Retargeter addon if available
        bpy.ops.mcp.load_and_retarget(filepath=bvh_path)
        print("[BVHtoFBX] BVH retargeting successful (using BVH Retargeter addon)")
    except AttributeError:
        print("[BVHtoFBX] BVH Retargeter addon not found, using standard BVH import...")

        # Fallback: Standard BVH import + manual retargeting
        # Import BVH (creates new armature)
        bpy.ops.import_anim.bvh(filepath=bvh_path)
        print("[BVHtoFBX] BVH imported")

        # Find BVH armature (newest armature)
        bvh_armature = None
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE' and obj != armature:
                bvh_armature = obj
                break

        if not bvh_armature:
            print("[BVHtoFBX] ERROR: BVH armature not found")
            sys.exit(1)

        print(f"[BVHtoFBX] Found BVH armature: {{bvh_armature.name}}")

        # Copy animation data from BVH armature to character armature
        # This is a simplified retargeting - in production, use BVH Retargeter addon
        if bvh_armature.animation_data and bvh_armature.animation_data.action:
            if not armature.animation_data:
                armature.animation_data_create()
            armature.animation_data.action = bvh_armature.animation_data.action.copy()
            print("[BVHtoFBX] Animation data copied")

        # Delete BVH armature
        bpy.data.objects.remove(bvh_armature, do_unlink=True)
        print("[BVHtoFBX] Cleaned up BVH armature")

    # Export result
    output_path = "{output_file}"
    output_format = "{output_format}"

    # Select all objects for export
    bpy.ops.object.select_all(action='SELECT')

    if output_format == "vrm":
        print("[BVHtoFBX] Exporting as VRM...")
        try:
            bpy.ops.export_scene.vrm(filepath=output_path)
            print("[BVHtoFBX] VRM export successful")
        except AttributeError:
            print("[BVHtoFBX] ERROR: VRM addon export not available, falling back to FBX")
            output_path = output_path.replace(".vrm", ".fbx")
            bpy.ops.export_scene.fbx(filepath=output_path)
            print("[BVHtoFBX] FBX export successful (fallback)")
    else:
        print("[BVHtoFBX] Exporting as FBX...")
        bpy.ops.export_scene.fbx(
            filepath=output_path,
            use_selection=True,
            bake_anim=True,
        )
        print("[BVHtoFBX] FBX export successful")

    print(f"[BVHtoFBX] Output saved to: {{output_path}}")
    print("[BVHtoFBX] Retargeting complete!")

except Exception as e:
    print(f"[BVHtoFBX] ERROR: {{str(e)}}")
    traceback.print_exc()
    sys.exit(1)
'''

        return script


NODE_CLASS_MAPPINGS = {
    "BVHtoFBX": BVHtoFBX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BVHtoFBX": "BVH to FBX Retargeter",
}
