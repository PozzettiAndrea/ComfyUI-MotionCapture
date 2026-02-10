"""ComfyUI-MotionCapture Prestartup Script."""

import shutil
from pathlib import Path

from comfy_env import setup_env, copy_files
from comfy_3d_viewers import copy_viewer

setup_env()

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy viewers
viewers = [
    "fbx", "fbx_compare",
    "bvh", "fbx_animation", "compare_smpl_bvh",
    "smpl", "smpl_camera",
]

for viewer in viewers:
    try:
        copy_viewer(viewer, SCRIPT_DIR / "web")
    except Exception as e:
        print(f"[MotionCapture] {e}")

# Copy file list updater utility
try:
    from comfy_3d_viewers import get_file_list_updater_path
    src = Path(get_file_list_updater_path())
    if src.exists():
        dst = SCRIPT_DIR / "web" / "js" / "file_list_updater.js"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)
except ImportError:
    pass

# Copy assets (all to input/, FBX also to input/3d/)
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input", "*")
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input" / "3d", "*.fbx")
