"""ComfyUI-MotionCapture: Motion capture from video for ComfyUI."""

import sys
import logging
import pathlib
import importlib.util

log = logging.getLogger("motioncapture")
log.info("loading...")

# Preload bpy's bundled libsycl and libembree (with SYCL support) before any
# other package (e.g. pymeshlab) can load a SYCL-less libembree4.so.4.
try:
    import importlib.util as _ilu
    import ctypes as _ctypes
    _bpy_spec = _ilu.find_spec("bpy")
    if _bpy_spec is not None:
        _bpy_lib_dir = pathlib.Path(_bpy_spec.origin).parent / "lib"
        for _lib in ("libsycl.so.7", "libembree4.so.4"):
            _lib_path = _bpy_lib_dir / _lib
            if _lib_path.exists():
                _ctypes.CDLL(str(_lib_path), mode=_ctypes.RTLD_GLOBAL)
                log.info("Preloaded %s", _lib)
except Exception as _e:
    log.warning("bpy library preload failed (bpy may not work): %s", _e)

_pkg_dir = pathlib.Path(__file__).parent
_nodes_dir = _pkg_dir / "nodes"

# Load MotionCapture's 'nodes' package under a unique name to avoid collision
# with ComfyUI's top-level nodes.py (both would otherwise be named 'nodes').
_mod_name = "ComfyUI_MotionCapture_nodes"
if _mod_name not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        _mod_name,
        _nodes_dir / "__init__.py",
        submodule_search_locations=[str(_nodes_dir)],
    )
    _mod = importlib.util.module_from_spec(_spec)
    _mod.__package__ = _mod_name
    sys.modules[_mod_name] = _mod
    _spec.loader.exec_module(_mod)
else:
    _mod = sys.modules[_mod_name]

NODE_CLASS_MAPPINGS = _mod.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = _mod.NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
