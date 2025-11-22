"""
Blender Addon Installer - Automate installation of VRM and BVH Retargeter addons
"""

import subprocess
import sys
from pathlib import Path
import urllib.request
import zipfile
import tempfile
import shutil

from hmr4d.utils.pylogger import Log


# Addon download URLs
VRM_ADDON_URL = "https://github.com/saturday06/VRM-Addon-for-Blender/releases/download/2_20_88/VRM_Addon_for_Blender-2_20_88.zip"
BVH_RETARGETER_URL = "https://github.com/Diffeomorphic/retarget-bvh/archive/refs/heads/master.zip"


def find_blender_executable():
    """
    Find Blender executable on the system.

    Returns:
        Path to Blender executable or None if not found
    """
    # Check common locations
    common_paths = [
        # Linux
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "~/blender/blender",
        # macOS
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "~/Applications/Blender.app/Contents/MacOS/Blender",
        # Windows
        "C:/Program Files/Blender Foundation/Blender/blender.exe",
        "C:/Program Files/Blender Foundation/Blender 4.0/blender.exe",
        "C:/Program Files/Blender Foundation/Blender 3.6/blender.exe",
    ]

    for path_str in common_paths:
        path = Path(path_str).expanduser()
        if path.exists() and path.is_file():
            Log.info(f"[BlenderAddon] Found Blender at: {path}")
            return str(path)

    # Try to find via 'which' command (Unix-like systems)
    try:
        result = subprocess.run(
            ["which", "blender"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            blender_path = result.stdout.strip()
            Log.info(f"[BlenderAddon] Found Blender via 'which': {blender_path}")
            return blender_path
    except Exception:
        pass

    Log.warning("[BlenderAddon] Blender executable not found")
    return None


def get_blender_addons_path(blender_executable):
    """
    Get the Blender addons directory path.

    Args:
        blender_executable: Path to Blender executable

    Returns:
        Path to Blender addons directory
    """
    # Run Blender to get the addons path
    python_code = """
import bpy
import os
addons_path = bpy.utils.user_resource('SCRIPTS', path='addons')
print(f"ADDONS_PATH:{addons_path}")
"""

    try:
        result = subprocess.run(
            [blender_executable, "--background", "--python-expr", python_code],
            capture_output=True,
            text=True,
            check=False
        )

        for line in result.stdout.split('\n'):
            if line.startswith('ADDONS_PATH:'):
                addons_path = line.replace('ADDONS_PATH:', '').strip()
                Log.info(f"[BlenderAddon] Addons path: {addons_path}")
                return Path(addons_path)

    except Exception as e:
        Log.error(f"[BlenderAddon] Failed to get addons path: {e}")

    return None


def download_file(url, dest_path):
    """
    Download a file from URL to destination path.

    Args:
        url: URL to download from
        dest_path: Destination file path
    """
    Log.info(f"[BlenderAddon] Downloading from {url}")
    urllib.request.urlretrieve(url, dest_path)
    Log.info(f"[BlenderAddon] Downloaded to {dest_path}")


def install_addon_from_zip(blender_executable, zip_path):
    """
    Install Blender addon from ZIP file.

    Args:
        blender_executable: Path to Blender executable
        zip_path: Path to addon ZIP file

    Returns:
        True if installation succeeded, False otherwise
    """
    # Python code to install addon
    python_code = f"""
import bpy
import sys

try:
    bpy.ops.preferences.addon_install(filepath="{zip_path}")
    print("ADDON_INSTALLED:SUCCESS")
except Exception as e:
    print(f"ADDON_INSTALLED:FAILED:{{e}}")
    sys.exit(1)
"""

    temp_script = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    temp_script.write(python_code)
    temp_script.close()

    try:
        result = subprocess.run(
            [blender_executable, "--background", "--python", temp_script.name],
            capture_output=True,
            text=True,
            check=False
        )

        for line in result.stdout.split('\n'):
            if line.startswith('ADDON_INSTALLED:SUCCESS'):
                Log.info("[BlenderAddon] Addon installed successfully")
                return True
            elif line.startswith('ADDON_INSTALLED:FAILED'):
                Log.error(f"[BlenderAddon] Addon installation failed: {line}")
                return False

    except Exception as e:
        Log.error(f"[BlenderAddon] Failed to install addon: {e}")
        return False
    finally:
        Path(temp_script.name).unlink(missing_ok=True)

    return False


def extract_addon_to_addons_dir(zip_path, addons_path):
    """
    Extract addon ZIP directly to Blender addons directory.

    Args:
        zip_path: Path to addon ZIP file
        addons_path: Path to Blender addons directory

    Returns:
        True if extraction succeeded, False otherwise
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get the list of files in the ZIP
            file_list = zip_ref.namelist()

            # Find the root addon folder name
            root_folders = set()
            for file_name in file_list:
                parts = Path(file_name).parts
                if len(parts) > 0:
                    root_folders.add(parts[0])

            Log.info(f"[BlenderAddon] Root folders in ZIP: {root_folders}")

            # Extract to addons directory
            zip_ref.extractall(addons_path)
            Log.info(f"[BlenderAddon] Extracted addon to {addons_path}")

        return True

    except Exception as e:
        Log.error(f"[BlenderAddon] Failed to extract addon: {e}")
        return False


def is_addon_installed(blender_executable, addon_module_name):
    """
    Check if an addon is already installed.

    Args:
        blender_executable: Path to Blender executable
        addon_module_name: Module name of the addon (e.g., 'VRM_Addon_for_Blender')

    Returns:
        True if addon is installed, False otherwise
    """
    python_code = f"""
import bpy
import addon_utils

addon_found = False
for mod in addon_utils.modules():
    if mod.__name__ == "{addon_module_name}":
        addon_found = True
        break

if addon_found:
    print("ADDON_STATUS:INSTALLED")
else:
    print("ADDON_STATUS:NOT_INSTALLED")
"""

    try:
        result = subprocess.run(
            [blender_executable, "--background", "--python-expr", python_code],
            capture_output=True,
            text=True,
            check=False,
            timeout=30
        )

        for line in result.stdout.split('\n'):
            if line.startswith('ADDON_STATUS:INSTALLED'):
                return True
            elif line.startswith('ADDON_STATUS:NOT_INSTALLED'):
                return False

    except Exception as e:
        Log.warning(f"[BlenderAddon] Failed to check addon status: {e}")

    return False


def install_vrm_addon(blender_executable=None):
    """
    Install VRM Addon for Blender.

    Args:
        blender_executable: Path to Blender executable (optional, will auto-detect)

    Returns:
        True if installation succeeded, False otherwise
    """
    if blender_executable is None:
        blender_executable = find_blender_executable()
        if blender_executable is None:
            Log.error("[BlenderAddon] Cannot install VRM addon: Blender not found")
            return False

    # Check if already installed
    if is_addon_installed(blender_executable, "VRM_Addon_for_Blender"):
        Log.info("[BlenderAddon] VRM addon is already installed")
        return True

    Log.info("[BlenderAddon] Installing VRM Addon...")

    # Get addons directory
    addons_path = get_blender_addons_path(blender_executable)
    if addons_path is None:
        Log.error("[BlenderAddon] Cannot get Blender addons path")
        return False

    addons_path.mkdir(parents=True, exist_ok=True)

    # Download VRM addon
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = Path(temp_dir) / "vrm_addon.zip"
        download_file(VRM_ADDON_URL, zip_path)

        # Extract to addons directory
        if not extract_addon_to_addons_dir(zip_path, addons_path):
            return False

    Log.info("[BlenderAddon] VRM Addon installed successfully")
    return True


def install_bvh_retargeter_addon(blender_executable=None):
    """
    Install BVH Retargeter addon for Blender.

    Args:
        blender_executable: Path to Blender executable (optional, will auto-detect)

    Returns:
        True if installation succeeded, False otherwise
    """
    if blender_executable is None:
        blender_executable = find_blender_executable()
        if blender_executable is None:
            Log.error("[BlenderAddon] Cannot install BVH Retargeter: Blender not found")
            return False

    # Check if already installed
    if is_addon_installed(blender_executable, "retarget_bvh"):
        Log.info("[BlenderAddon] BVH Retargeter addon is already installed")
        return True

    Log.info("[BlenderAddon] Installing BVH Retargeter addon...")

    # Get addons directory
    addons_path = get_blender_addons_path(blender_executable)
    if addons_path is None:
        Log.error("[BlenderAddon] Cannot get Blender addons path")
        return False

    addons_path.mkdir(parents=True, exist_ok=True)

    # Download BVH Retargeter
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = Path(temp_dir) / "bvh_retargeter.zip"
        download_file(BVH_RETARGETER_URL, zip_path)

        # Extract and rename
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find extracted folder (likely 'retarget-bvh-master')
        extracted_folders = [d for d in Path(temp_dir).iterdir() if d.is_dir()]
        if not extracted_folders:
            Log.error("[BlenderAddon] No folder found in BVH Retargeter ZIP")
            return False

        source_folder = extracted_folders[0]
        dest_folder = addons_path / "retarget_bvh"

        # Remove old installation if exists
        if dest_folder.exists():
            shutil.rmtree(dest_folder)

        # Copy to addons directory
        shutil.copytree(source_folder, dest_folder)
        Log.info(f"[BlenderAddon] Copied BVH Retargeter to {dest_folder}")

    Log.info("[BlenderAddon] BVH Retargeter addon installed successfully")
    return True


def install_all_addons(blender_executable=None):
    """
    Install all required Blender addons.

    Args:
        blender_executable: Path to Blender executable (optional, will auto-detect)

    Returns:
        True if all installations succeeded, False otherwise
    """
    if blender_executable is None:
        blender_executable = find_blender_executable()
        if blender_executable is None:
            Log.error("[BlenderAddon] Blender not found. Please install Blender first.")
            return False

    Log.info("[BlenderAddon] Installing all required Blender addons...")

    vrm_success = install_vrm_addon(blender_executable)
    bvh_success = install_bvh_retargeter_addon(blender_executable)

    if vrm_success and bvh_success:
        Log.info("[BlenderAddon] All addons installed successfully")
        return True
    else:
        Log.warning("[BlenderAddon] Some addons failed to install")
        return False


if __name__ == "__main__":
    # Command line usage
    install_all_addons()
