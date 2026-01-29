"""
LoadDPVOModel Node - Loads and initializes DPVO visual odometry model

DPVO is used for camera motion estimation in moving camera scenarios.
This node auto-downloads the model checkpoint and returns a pre-loaded model bundle.
"""

import os
import sys
import zipfile
from pathlib import Path
import torch

# Add vendor path for DPVO
VENDOR_PATH = Path(__file__).parent / "vendor"
sys.path.insert(0, str(VENDOR_PATH))

# Import logging from GVHMR
from hmr4d.utils.pylogger import Log

# Global cache - persists across node executions
_DPVO_MODEL_CACHE = {}


class LoadDPVOModel:
    """
    ComfyUI node for loading DPVO (Deep Patch Visual Odometry) model.

    DPVO provides accurate camera motion estimation for moving camera scenarios,
    which is essential for accurate 3D human motion recovery in GVHMR.

    Downloads model automatically from Dropbox if missing.
    """

    # Model download configuration
    MODEL_CONFIGS = {
        "dpvo": {
            "url": "https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip?dl=1",
            "filename": "dpvo.pth",
        }
    }

    def __init__(self):
        # Models stored in ComfyUI/models/motion_capture/dpvo/
        self.models_dir = Path(__file__).parent.parent.parent.parent / "models" / "motion_capture" / "dpvo"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "model_path_override": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional: Override default model checkpoint path (dpvo.pth)"
                }),
            }
        }

    RETURN_TYPES = ("DPVO_MODEL",)
    RETURN_NAMES = ("dpvo_model",)
    FUNCTION = "load_model"
    CATEGORY = "MotionCapture/GVHMR"

    def _create_default_config(self, config_path: Path) -> None:
        """Create default config.yaml with DPVO defaults."""
        Log.info(f"[LoadDPVOModel] Creating default config at {config_path}")
        default_config = """# DPVO default configuration
# These are the default values from dpvo/config.py

BUFFER_SIZE: 4096
CENTROID_SEL_STRAT: 'RANDOM'
PATCHES_PER_FRAME: 80
REMOVAL_WINDOW: 20
OPTIMIZATION_WINDOW: 12
PATCH_LIFETIME: 12
KEYFRAME_INDEX: 4
KEYFRAME_THRESH: 12.5
MOTION_MODEL: 'DAMPED_LINEAR'
MOTION_DAMPING: 0.5
MIXED_PRECISION: true
LOOP_CLOSURE: false
BACKEND_THRESH: 64.0
MAX_EDGE_AGE: 1000
GLOBAL_OPT_FREQ: 15
CLASSIC_LOOP_CLOSURE: false
LOOP_CLOSE_WINDOW_SIZE: 3
LOOP_RETR_THRESH: 0.04
"""
        config_path.write_text(default_config)

    def download_model(self, target_dir: Path) -> bool:
        """Download DPVO model from Dropbox if missing."""
        checkpoint_path = target_dir / "dpvo.pth"
        config_path = target_dir / "config.yaml"

        if checkpoint_path.exists() and config_path.exists():
            Log.info(f"[LoadDPVOModel] Model found at {target_dir}")
            return True

        Log.info("[LoadDPVOModel] Downloading DPVO model from Dropbox...")
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            import requests
            from tqdm import tqdm

            url = self.MODEL_CONFIGS["dpvo"]["url"]
            zip_path = target_dir / "models.zip"

            # Download with progress bar
            response = requests.get(url, stream=True, allow_redirects=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(zip_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading DPVO") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

            Log.info("[LoadDPVOModel] Extracting model files...")

            # Extract zip
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)

            # The zip contains a 'models' subdirectory - move contents up
            models_subdir = target_dir / "models"
            if models_subdir.exists():
                for item in models_subdir.iterdir():
                    dest = target_dir / item.name
                    if not dest.exists():
                        item.rename(dest)
                # Clean up empty models dir
                try:
                    models_subdir.rmdir()
                except OSError:
                    pass

            # Clean up zip file
            zip_path.unlink()

            if checkpoint_path.exists():
                Log.info(f"[LoadDPVOModel] Downloaded to {target_dir}")
                # Create default config.yaml if not present in the zip
                if not config_path.exists():
                    self._create_default_config(config_path)
                return True
            else:
                Log.error(f"[LoadDPVOModel] dpvo.pth not found after extraction")
                return False

        except ImportError:
            Log.error("[LoadDPVOModel] requests package not installed. Run: pip install requests")
            return False
        except Exception as e:
            Log.error(f"[LoadDPVOModel] Download failed: {e}")
            return False

    def load_model(self, model_path_override=""):
        """Load DPVO model and return model bundle."""

        # Determine model path
        if model_path_override and model_path_override.strip():
            checkpoint_path = Path(model_path_override)
            model_dir = checkpoint_path.parent
        else:
            model_dir = self.models_dir
            checkpoint_path = model_dir / "dpvo.pth"

        # Check cache
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_key = f"{checkpoint_path}_{device}"

        if cache_key in _DPVO_MODEL_CACHE:
            Log.info("[LoadDPVOModel] Using cached model")
            return (_DPVO_MODEL_CACHE[cache_key],)

        Log.info("[LoadDPVOModel] Initializing DPVO model...")

        # Check and download model if needed
        if not checkpoint_path.exists():
            if not self.download_model(model_dir):
                raise FileNotFoundError(
                    f"\n{'='*80}\n"
                    f"DPVO Model Not Found!\n\n"
                    f"Could not download DPVO model automatically.\n\n"
                    f"Manual download:\n"
                    f"  1. Download models.zip from:\n"
                    f"     https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip\n"
                    f"  2. Extract to: {model_dir}/\n"
                    f"  3. Ensure dpvo.pth and config.yaml exist\n\n"
                    f"Expected structure:\n"
                    f"  {model_dir}/\n"
                    f"    ├── dpvo.pth\n"
                    f"    └── config.yaml\n"
                    f"{'='*80}"
                )

        # Check config file - create default if missing
        config_path = model_dir / "config.yaml"
        if not config_path.exists():
            self._create_default_config(config_path)

        try:
            # Import DPVO components
            from dpvo.dpvo import DPVO
            from dpvo.config import cfg as dpvo_cfg

            Log.info(f"[LoadDPVOModel] Loading model from {checkpoint_path}...")

            # Load config
            dpvo_cfg.merge_from_file(str(config_path))

            # Create model bundle with config and checkpoint path
            # We don't instantiate DPVO here since it needs image dimensions (ht, wd)
            # Instead, we return the config and checkpoint path for lazy loading
            model_bundle = {
                "config": dpvo_cfg.clone(),
                "checkpoint_path": str(checkpoint_path),
                "config_path": str(config_path),
                "device": device,
                "model_dir": str(model_dir),
            }

            # Cache it
            _DPVO_MODEL_CACHE[cache_key] = model_bundle

            Log.info("[LoadDPVOModel] DPVO model loaded successfully!")
            Log.info(f"[LoadDPVOModel] Device: {device}")
            Log.info(f"[LoadDPVOModel] Checkpoint: {checkpoint_path}")

            return (model_bundle,)

        except ImportError as e:
            raise RuntimeError(
                f"Failed to import DPVO module. "
                f"Ensure lietorch and torch-scatter are installed.\n"
                f"Error: {e}"
            ) from e
        except Exception as e:
            Log.error(f"[LoadDPVOModel] Failed to load model: {e}")
            raise


# Node registration
NODE_CLASS_MAPPINGS = {
    "LoadDPVOModel": LoadDPVOModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadDPVOModel": "Load DPVO Model",
}
