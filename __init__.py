"""
ComfyUI-MotionCapture Custom Node Package

A ComfyUI node package for motion capture from video.
Supports both GVHMR (SMPL output) and SAM 3D Body (MHR output) backends.
Extracts 3D human motion from video with SAM3 segmentation.
"""

import os
from pathlib import Path

# Module info
__version__ = "0.2.0"
__author__ = "ComfyUI-MotionCapture"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Pytest detection - skip heavy initialization during tests
force_init = os.environ.get('MOCAP_FORCE_INIT') == '1'
is_pytest = 'PYTEST_CURRENT_TEST' in os.environ
skip_init = is_pytest and not force_init

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

if skip_init:
    print("[MotionCapture] Running in pytest mode, skipping node initialization")
else:
    nodes_dir = Path(__file__).parent / "nodes"

    # ==============================================================================
    # GPU nodes (CUDA + vendor deps) - isolated environment
    # ==============================================================================
    try:
        from comfy_env import wrap_isolated_nodes

        # GPU nodes
        try:
            from .nodes.nodes_gpu import NODE_CLASS_MAPPINGS as gpu_mappings
            from .nodes.nodes_gpu import NODE_DISPLAY_NAME_MAPPINGS as gpu_display
            gpu_wrapped = wrap_isolated_nodes(gpu_mappings, nodes_dir / "nodes_gpu")
            NODE_CLASS_MAPPINGS.update(gpu_wrapped)
            NODE_DISPLAY_NAME_MAPPINGS.update(gpu_display)
            print(f"[MotionCapture] GPU nodes loaded ({len(gpu_mappings)} nodes, isolated)")
        except ImportError as e:
            print(f"[MotionCapture] GPU nodes not available: {e}")

        # Blender nodes
        try:
            from .nodes.nodes_blender import NODE_CLASS_MAPPINGS as blender_mappings
            from .nodes.nodes_blender import NODE_DISPLAY_NAME_MAPPINGS as blender_display
            blender_wrapped = wrap_isolated_nodes(blender_mappings, nodes_dir / "nodes_blender")
            NODE_CLASS_MAPPINGS.update(blender_wrapped)
            NODE_DISPLAY_NAME_MAPPINGS.update(blender_display)
            print(f"[MotionCapture] Blender nodes loaded ({len(blender_mappings)} nodes, isolated)")
        except ImportError as e:
            print(f"[MotionCapture] Blender nodes not available: {e}")

    except ImportError:
        print("[MotionCapture] comfy-env not installed, isolated nodes disabled")
        print("[MotionCapture] Install with: pip install comfy-env>=0.0.19")

    # ==============================================================================
    # Viewer nodes - NOT isolated (to avoid IPC size limits for large mesh data)
    # ==============================================================================
    try:
        from .nodes.viewer_node import NODE_CLASS_MAPPINGS as viewer_mappings
        from .nodes.viewer_node import NODE_DISPLAY_NAME_MAPPINGS as viewer_display
        NODE_CLASS_MAPPINGS.update(viewer_mappings)
        NODE_DISPLAY_NAME_MAPPINGS.update(viewer_display)
        print(f"[MotionCapture] Viewer nodes loaded ({len(viewer_mappings)} nodes, main process)")
    except ImportError as e:
        print(f"[MotionCapture] Viewer nodes not available: {e}")

    print(f"[MotionCapture] Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")

    # Register API endpoints for dynamic file loading
    try:
        from server import PromptServer
        from aiohttp import web

        # Import node classes for API endpoints (they're already loaded above)
        LoadFBXCharacter = NODE_CLASS_MAPPINGS.get("LoadFBXCharacter")
        LoadMixamoCharacter = NODE_CLASS_MAPPINGS.get("LoadMixamoCharacter")
        LoadSMPL = NODE_CLASS_MAPPINGS.get("LoadSMPL")

        if LoadFBXCharacter:
            @PromptServer.instance.routes.get('/motioncapture/fbx_files')
            async def get_fbx_files(request):
                """API endpoint to fetch FBX file list dynamically."""
                source = request.query.get('source_folder', 'output')
                try:
                    if source == "input":
                        files = LoadFBXCharacter.get_fbx_files_from_input()
                    else:
                        files = LoadFBXCharacter.get_fbx_files_from_output()
                    return web.json_response(files)
                except Exception as e:
                    print(f"[MotionCapture API] Error getting FBX files: {e}")
                    return web.json_response([])

            print("[MotionCapture] API endpoint registered: /motioncapture/fbx_files")

        if LoadMixamoCharacter:
            @PromptServer.instance.routes.get('/motioncapture/mixamo_files')
            async def get_mixamo_files(request):
                """API endpoint to fetch Mixamo FBX file list from input/3d."""
                try:
                    files = LoadMixamoCharacter.get_mixamo_files()
                    return web.json_response(files)
                except Exception as e:
                    print(f"[MotionCapture API] Error getting Mixamo files: {e}")
                    return web.json_response([])

            print("[MotionCapture] API endpoint registered: /motioncapture/mixamo_files")

        if LoadSMPL:
            @PromptServer.instance.routes.get('/motioncapture/npz_files')
            async def get_npz_files(request):
                """API endpoint to fetch NPZ file list dynamically."""
                source = request.query.get('source_folder', 'output')
                try:
                    if source == "input":
                        files = LoadSMPL.get_npz_files_from_input()
                    else:
                        files = LoadSMPL.get_npz_files_from_output()
                    return web.json_response(files)
                except Exception as e:
                    print(f"[MotionCapture API] Error getting NPZ files: {e}")
                    return web.json_response([])

            print("[MotionCapture] API endpoint registered: /motioncapture/npz_files")

        @PromptServer.instance.routes.get('/motioncapture/smpl_mesh')
        async def get_smpl_mesh_file(request):
            """API endpoint to fetch SMPL mesh binary file."""
            filename = request.query.get('filename', None)
            if not filename:
                raise web.HTTPBadRequest(reason="Missing filename parameter")

            filepath = Path("output") / filename
            if not filepath.is_file():
                raise web.HTTPNotFound(reason=f"File not found: {filename}")

            return web.FileResponse(filepath)

        print("[MotionCapture] API endpoint registered: /motioncapture/smpl_mesh")

    except Exception as e:
        print(f"[MotionCapture] Warning: Could not register API endpoints: {e}")
        print("[MotionCapture] FBX file browsing will not work without PromptServer")
