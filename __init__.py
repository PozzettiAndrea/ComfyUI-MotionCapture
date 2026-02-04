"""ComfyUI-MotionCapture: Motion capture from video for ComfyUI."""

from pathlib import Path

from comfy_env import wrap_nodes
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

wrap_nodes()

# Register API endpoints for dynamic file loading
try:
    from server import PromptServer
    from aiohttp import web

    LoadFBXCharacter = NODE_CLASS_MAPPINGS.get("LoadFBXCharacter")
    LoadMixamoCharacter = NODE_CLASS_MAPPINGS.get("LoadMixamoCharacter")
    LoadSMPL = NODE_CLASS_MAPPINGS.get("LoadSMPL")

    if LoadFBXCharacter:
        @PromptServer.instance.routes.get('/motioncapture/fbx_files')
        async def get_fbx_files(request):
            source = request.query.get('source_folder', 'output')
            try:
                if source == "input":
                    files = LoadFBXCharacter.get_fbx_files_from_input()
                else:
                    files = LoadFBXCharacter.get_fbx_files_from_output()
                return web.json_response(files)
            except Exception:
                return web.json_response([])

    if LoadMixamoCharacter:
        @PromptServer.instance.routes.get('/motioncapture/mixamo_files')
        async def get_mixamo_files(request):
            try:
                files = LoadMixamoCharacter.get_mixamo_files()
                return web.json_response(files)
            except Exception:
                return web.json_response([])

    if LoadSMPL:
        @PromptServer.instance.routes.get('/motioncapture/npz_files')
        async def get_npz_files(request):
            source = request.query.get('source_folder', 'output')
            try:
                if source == "input":
                    files = LoadSMPL.get_npz_files_from_input()
                else:
                    files = LoadSMPL.get_npz_files_from_output()
                return web.json_response(files)
            except Exception:
                return web.json_response([])

    @PromptServer.instance.routes.get('/motioncapture/smpl_mesh')
    async def get_smpl_mesh_file(request):
        filename = request.query.get('filename', None)
        if not filename:
            raise web.HTTPBadRequest(reason="Missing filename parameter")
        filepath = Path("output") / filename
        if not filepath.is_file():
            raise web.HTTPNotFound(reason=f"File not found: {filename}")
        return web.FileResponse(filepath)

except Exception:
    pass

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
