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
    copy_viewer(viewer, SCRIPT_DIR / "web")

# Copy assets (all to input/, FBX also to input/3d/)
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input")
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input" / "3d", "*.fbx")
