#!/usr/bin/env python3
"""
ComfyUI-MotionCapture prestartup script.

Automatically copies:
- Example assets and workflows to ComfyUI directories
- All viewer files from comfy-3d-viewers package

Runs before ComfyUI's main initialization.
"""

import shutil
from pathlib import Path


def copy_viewer_file(src_path: str, dst_path: Path, name: str) -> bool:
    """Copy a viewer file if it needs updating."""
    src = Path(src_path)
    if not src.exists():
        print(f"[MotionCapture] Warning: Source file not found: {src}")
        return False

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if not dst_path.exists() or src.stat().st_mtime > dst_path.stat().st_mtime:
        shutil.copy2(src, dst_path)
        print(f"[MotionCapture] Copied {name}")
        return True
    return False


def copy_viewers():
    """Copy all viewer files from comfy-3d-viewers package."""
    try:
        from comfy_3d_viewers import (
            # FBX viewer (rigged mesh preview)
            get_fbx_html_path,
            get_fbx_bundle_path,
            get_fbx_node_widget_path,
            # BVH viewer
            get_bvh_html_path,
            get_bvh_widget_path,
            # SMPL viewer (Canvas 2D)
            get_smpl_widget_path,
            # MHR viewer (Canvas 2D)
            get_mhr_widget_path,
            # FBX animation viewer
            get_fbx_animation_html_path,
            get_fbx_animation_widget_path,
            # Compare SMPL/BVH viewer
            get_compare_smpl_bvh_html_path,
            get_compare_smpl_bvh_widget_path,
            # Compare Skeletons viewer (side-by-side FBX)
            get_fbx_compare_html_path,
            get_fbx_compare_widget_path,
            # File list updater utility
            get_file_list_updater_path,
        )
    except ImportError:
        print("[MotionCapture] Warning: comfy-3d-viewers not installed, viewers disabled")
        print("[MotionCapture] Install with: pip install comfy-3d-viewers")
        return

    try:
        custom_node_dir = Path(__file__).parent
        web_dir = custom_node_dir / "web"
        three_dir = web_dir / "three"
        js_dir = web_dir / "js"

        # Create directories
        web_dir.mkdir(exist_ok=True)
        three_dir.mkdir(exist_ok=True)
        js_dir.mkdir(exist_ok=True)

        copied_count = 0

        # ═══════════════════════════════════════════════════════════════
        # FBX Rigged Mesh Preview (viewer_fbx.html + mesh_preview_fbx.js)
        # ═══════════════════════════════════════════════════════════════
        if copy_viewer_file(get_fbx_html_path(), web_dir / "viewer_fbx.html", "viewer_fbx.html"):
            copied_count += 1
        if copy_viewer_file(get_fbx_bundle_path(), three_dir / "viewer-bundle.js", "viewer-bundle.js"):
            copied_count += 1
        if copy_viewer_file(get_fbx_node_widget_path(), js_dir / "mesh_preview_fbx.js", "mesh_preview_fbx.js"):
            copied_count += 1

        # ═══════════════════════════════════════════════════════════════
        # BVH Viewer (viewer_bvh.html + bvh_viewer.js)
        # ═══════════════════════════════════════════════════════════════
        if copy_viewer_file(get_bvh_html_path(), web_dir / "viewer_bvh.html", "viewer_bvh.html"):
            copied_count += 1
        if copy_viewer_file(get_bvh_widget_path(), js_dir / "bvh_viewer.js", "bvh_viewer.js"):
            copied_count += 1

        # ═══════════════════════════════════════════════════════════════
        # SMPL Viewer (Canvas 2D, no HTML template needed)
        # ═══════════════════════════════════════════════════════════════
        if copy_viewer_file(get_smpl_widget_path(), js_dir / "smpl_viewer.js", "smpl_viewer.js"):
            copied_count += 1

        # ═══════════════════════════════════════════════════════════════
        # MHR Skeleton Viewer (Canvas 2D, no HTML template needed)
        # ═══════════════════════════════════════════════════════════════
        if copy_viewer_file(get_mhr_widget_path(), js_dir / "mhr_viewer.js", "mhr_viewer.js"):
            copied_count += 1

        # ═══════════════════════════════════════════════════════════════
        # FBX Animation Viewer (viewer_fbx_animation.html + fbx_animation_viewer.js)
        # ═══════════════════════════════════════════════════════════════
        if copy_viewer_file(get_fbx_animation_html_path(), web_dir / "viewer_fbx_animation.html", "viewer_fbx_animation.html"):
            copied_count += 1
        if copy_viewer_file(get_fbx_animation_widget_path(), js_dir / "fbx_animation_viewer.js", "fbx_animation_viewer.js"):
            copied_count += 1

        # ═══════════════════════════════════════════════════════════════
        # Compare SMPL/BVH Viewer (viewer_compare_smpl_bvh.html + compare_smpl_bvh.js)
        # ═══════════════════════════════════════════════════════════════
        if copy_viewer_file(get_compare_smpl_bvh_html_path(), web_dir / "viewer_compare_smpl_bvh.html", "viewer_compare_smpl_bvh.html"):
            copied_count += 1
        if copy_viewer_file(get_compare_smpl_bvh_widget_path(), js_dir / "compare_smpl_bvh.js", "compare_smpl_bvh.js"):
            copied_count += 1

        # ═══════════════════════════════════════════════════════════════
        # Compare Skeletons Viewer (viewer_fbx_compare.html + compare_skeleton_widget.js)
        # ═══════════════════════════════════════════════════════════════
        if copy_viewer_file(get_fbx_compare_html_path(), web_dir / "viewer_fbx_compare.html", "viewer_fbx_compare.html"):
            copied_count += 1
        if copy_viewer_file(get_fbx_compare_widget_path(), js_dir / "compare_skeleton_widget.js", "compare_skeleton_widget.js"):
            copied_count += 1

        # ═══════════════════════════════════════════════════════════════
        # File List Updater Utility
        # ═══════════════════════════════════════════════════════════════
        if copy_viewer_file(get_file_list_updater_path(), js_dir / "file_list_updater.js", "file_list_updater.js"):
            copied_count += 1

        if copied_count > 0:
            print(f"[MotionCapture] Updated {copied_count} viewer file(s)")
        else:
            print("[MotionCapture] Viewer files are up to date")

    except Exception as e:
        print(f"[MotionCapture] Error copying viewer files: {e}")
        import traceback
        traceback.print_exc()


def copy_assets():
    """Copy all files from assets/ to ComfyUI/input/ (FBX files go to input/3d/)"""
    try:
        custom_node_dir = Path(__file__).parent
        comfyui_dir = custom_node_dir.parent.parent
        input_dir = comfyui_dir / "input"
        input_3d_dir = input_dir / "3d"  # Subfolder for 3D assets (FBX, etc.)
        assets_src = custom_node_dir / "assets"

        if not assets_src.exists():
            print("[MotionCapture] No assets/ directory found, skipping asset copy")
            return

        input_dir.mkdir(parents=True, exist_ok=True)
        input_3d_dir.mkdir(parents=True, exist_ok=True)

        # File extensions that should go to input/3d/
        EXTENSIONS_3D = {".fbx"}

        copied_count = 0
        skipped_count = 0

        for item in assets_src.iterdir():
            if item.name.startswith('.'):
                continue
            if item.is_dir():
                continue

            # Route FBX and other 3D files to input/3d/, others to input/
            if item.suffix.lower() in EXTENSIONS_3D:
                dest = input_3d_dir / item.name
            else:
                dest = input_dir / item.name

            if dest.exists():
                skipped_count += 1
                continue

            try:
                shutil.copy2(item, dest)
                print(f"[MotionCapture] Copied asset: {item.name} -> {dest}")
                copied_count += 1
            except Exception as e:
                print(f"[MotionCapture] Failed to copy {item.name}: {e}")

        if copied_count > 0:
            print(f"[MotionCapture] Copied {copied_count} asset file(s)")
        if skipped_count > 0:
            print(f"[MotionCapture] Skipped {skipped_count} existing asset file(s)")

    except Exception as e:
        print(f"[MotionCapture] Error copying assets: {e}")


# Run on import
if __name__ == "__main__":
    print("[MotionCapture] Running prestartup script...")
    copy_viewers()
    copy_assets()
    print("[MotionCapture] Prestartup script completed")
else:
    # Also run when imported by ComfyUI
    print("[MotionCapture] Running prestartup script...")
    copy_viewers()
    copy_assets()
    print("[MotionCapture] Prestartup script completed")
