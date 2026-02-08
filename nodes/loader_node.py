"""
LoadGVHMRModels Node - Downloads and verifies GVHMR model files
"""

import os
import sys
import zipfile
from pathlib import Path
import torch
import folder_paths

# Add vendor path for GVHMR
VENDOR_PATH = Path(__file__).parent / "vendor"
sys.path.insert(0, str(VENDOR_PATH))

MODELS_DIR = Path(folder_paths.models_dir) / "motion_capture"

# Import logger
from hmr4d.utils.pylogger import Log

# Check DPVO availability
DPVO_AVAILABLE = False
try:
    from dpvo.dpvo import DPVO
    from dpvo.config import cfg as dpvo_cfg
    DPVO_AVAILABLE = True
except ImportError:
    pass


class LoadGVHMRModels:
    """
    ComfyUI node for loading GVHMR models and preprocessing components.
    Downloads models automatically if missing (except SMPL body models).
    """

    # Model download configuration (HuggingFace)
    MODEL_CONFIGS = {
        "gvhmr": {
            "repo_id": "camenduru/GVHMR",
            "filename": "gvhmr/gvhmr_siga24_release.ckpt",
        },
        "vitpose": {
            "repo_id": "camenduru/GVHMR",
            "filename": "vitpose/vitpose-h-multi-coco.pth",
        },
        "hmr2": {
            "repo_id": "camenduru/GVHMR",
            "filename": "hmr2/epoch=10-step=25000.ckpt",
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "model_path_override": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional: Override default model checkpoint path"
                }),
                "cache_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep model in GPU memory between inference runs"
                }),
                "load_dpvo": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Load DPVO model for moving camera scenarios (downloads ~100MB if missing)"
                }),
            }
        }

    RETURN_TYPES = ("GVHMR_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "load_models"
    CATEGORY = "MotionCapture/GVHMR"

    def check_and_download_model(self, model_name: str, target_path: Path) -> bool:
        """Check if model exists, download from HuggingFace if missing."""
        if target_path.exists():
            Log.info(f"[LoadGVHMRModels] {model_name} found at {target_path}")
            return True

        if model_name not in self.MODEL_CONFIGS:
            Log.error(f"[LoadGVHMRModels] No download config for {model_name}")
            return False

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            Log.error("[LoadGVHMRModels] huggingface_hub not installed. Run: pip install huggingface_hub")
            return False

        config = self.MODEL_CONFIGS[model_name]
        Log.info(f"[LoadGVHMRModels] Downloading {model_name} from HuggingFace...")
        Log.info(f"[LoadGVHMRModels] Repository: {config['repo_id']}")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            downloaded_path = hf_hub_download(
                repo_id=config["repo_id"],
                filename=config["filename"],
                cache_dir=str(MODELS_DIR / "_hf_cache"),
            )
            # Copy to target location
            import shutil
            shutil.copy(downloaded_path, str(target_path))
            Log.info(f"[LoadGVHMRModels] Downloaded {model_name} to {target_path}")
            return True
        except Exception as e:
            Log.error(f"[LoadGVHMRModels] Failed to download {model_name}: {e}")
            return False

    def download_smpl_from_hf(self, model_name: str, target_path: Path) -> bool:
        """Download SMPL model from HuggingFace if missing."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            Log.error("[LoadGVHMRModels] huggingface_hub not installed. Run: pip install huggingface_hub")
            return False

        hf_files = {
            # Use NPZ versions of SMPL models (no chumpy dependency)
            "SMPL_FEMALE.npz": "4_SMPLhub/SMPL/X_model_npz/SMPL_F_model.npz",
            "SMPL_MALE.npz": "4_SMPLhub/SMPL/X_model_npz/SMPL_M_model.npz",
            "SMPL_NEUTRAL.npz": "4_SMPLhub/SMPL/X_model_npz/SMPL_N_model.npz",
            "SMPLX_FEMALE.npz": "4_SMPLhub/SMPLX/X_npz/SMPLX_FEMALE.npz",
            "SMPLX_MALE.npz": "4_SMPLhub/SMPLX/X_npz/SMPLX_MALE.npz",
            "SMPLX_NEUTRAL.npz": "4_SMPLhub/SMPLX/X_npz/SMPLX_NEUTRAL.npz",
        }

        if model_name not in hf_files:
            return False

        Log.info(f"[LoadGVHMRModels] Downloading {model_name} from HuggingFace...")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            downloaded = hf_hub_download(
                repo_id="lithiumice/models_hub",
                filename=hf_files[model_name],
                cache_dir=str(MODELS_DIR / "_hf_cache"),
            )
            import shutil
            shutil.copy(downloaded, str(target_path))
            Log.info(f"[LoadGVHMRModels] Downloaded {model_name}")
            return True
        except Exception as e:
            Log.error(f"[LoadGVHMRModels] Failed to download {model_name}: {e}")
            return False

    def check_smpl_models(self) -> bool:
        """Check if SMPL body models are available, download from HuggingFace if missing."""
        smpl_dir = MODELS_DIR / "body_models" / "smpl"
        smplx_dir = MODELS_DIR / "body_models" / "smplx"

        smpl_files = ["SMPL_FEMALE.npz", "SMPL_MALE.npz", "SMPL_NEUTRAL.npz"]
        smplx_files = ["SMPLX_FEMALE.npz", "SMPLX_MALE.npz", "SMPLX_NEUTRAL.npz"]

        # Check and download SMPL models if missing
        for filename in smpl_files:
            file_path = smpl_dir / filename
            if not file_path.exists():
                Log.info(f"[LoadGVHMRModels] {filename} not found, downloading from HuggingFace...")
                if not self.download_smpl_from_hf(filename, file_path):
                    Log.warn(f"[LoadGVHMRModels] Could not auto-download {filename}")

        # Check and download SMPL-X models if missing
        for filename in smplx_files:
            file_path = smplx_dir / filename
            if not file_path.exists():
                Log.info(f"[LoadGVHMRModels] {filename} not found, downloading from HuggingFace...")
                if not self.download_smpl_from_hf(filename, file_path):
                    Log.warn(f"[LoadGVHMRModels] Could not auto-download {filename}")

        # Final check
        smpl_exists = all((smpl_dir / f).exists() for f in smpl_files)
        smplx_exists = all((smplx_dir / f).exists() for f in smplx_files)

        if not (smpl_exists or smplx_exists):
            error_msg = (
                "\n" + "="*80 + "\n"
                "SMPL Body Models Not Found!\n\n"
                "Attempted auto-download from HuggingFace but failed.\n"
                "You can manually download SMPL models:\n\n"
                "Option 1: Run install.py script\n"
                "  cd ComfyUI/custom_nodes/ComfyUI-MotionCapture\n"
                "  python install.py\n\n"
                "Option 2: Manual download (official sources)\n"
                "  1. Visit https://smpl.is.tue.mpg.de/ and register\n"
                "  2. Visit https://smpl-x.is.tue.mpg.de/ and register\n"
                "  3. Place files in:\n"
                f"     {smpl_dir}/\n"
                f"     {smplx_dir}/\n\n"
                f"See {MODELS_DIR}/README.md for detailed instructions.\n"
                + "="*80
            )
            raise FileNotFoundError(error_msg)

        Log.info(f"[LoadGVHMRModels] SMPL body models found")
        return True

    def _create_dpvo_config(self, config_path: Path) -> None:
        """Create default config.yaml with DPVO defaults."""
        Log.info(f"[LoadGVHMRModels] Creating default DPVO config at {config_path}")
        default_config = """# DPVO default configuration
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

    def download_dpvo_model(self, target_dir: Path) -> bool:
        """Download DPVO model from Dropbox if missing."""
        checkpoint_path = target_dir / "dpvo.pth"
        config_path = target_dir / "config.yaml"

        if checkpoint_path.exists() and config_path.exists():
            Log.info(f"[LoadGVHMRModels] DPVO model found at {target_dir}")
            return True

        Log.info("[LoadGVHMRModels] Downloading DPVO model from Dropbox...")
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            import requests
            from tqdm import tqdm

            url = "https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip?dl=1"
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

            Log.info("[LoadGVHMRModels] Extracting DPVO model files...")

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
                try:
                    models_subdir.rmdir()
                except OSError:
                    pass

            # Clean up zip file
            zip_path.unlink()

            if checkpoint_path.exists():
                Log.info(f"[LoadGVHMRModels] DPVO downloaded to {target_dir}")
                if not config_path.exists():
                    self._create_dpvo_config(config_path)
                return True
            else:
                Log.error(f"[LoadGVHMRModels] dpvo.pth not found after extraction")
                return False

        except ImportError:
            Log.error("[LoadGVHMRModels] requests package not installed. Run: pip install requests")
            return False
        except Exception as e:
            Log.error(f"[LoadGVHMRModels] DPVO download failed: {e}")
            return False

    def load_dpvo_model(self, dpvo_dir: Path) -> dict:
        """Load DPVO model and return model bundle."""
        if not DPVO_AVAILABLE:
            Log.warn("[LoadGVHMRModels] DPVO package not installed, skipping DPVO loading")
            return None

        checkpoint_path = dpvo_dir / "dpvo.pth"
        config_path = dpvo_dir / "config.yaml"

        if not checkpoint_path.exists():
            if not self.download_dpvo_model(dpvo_dir):
                Log.warn("[LoadGVHMRModels] Could not download DPVO model")
                return None

        if not config_path.exists():
            self._create_dpvo_config(config_path)

        try:
            Log.info(f"[LoadGVHMRModels] Loading DPVO config from {config_path}...")
            dpvo_cfg.merge_from_file(str(config_path))

            device = "cuda" if torch.cuda.is_available() else "cpu"

            model_bundle = {
                "config": dpvo_cfg.clone(),
                "checkpoint_path": str(checkpoint_path),
                "config_path": str(config_path),
                "device": device,
                "model_dir": str(dpvo_dir),
            }

            Log.info("[LoadGVHMRModels] DPVO model loaded successfully!")
            return model_bundle

        except Exception as e:
            Log.error(f"[LoadGVHMRModels] Failed to load DPVO model: {e}")
            return None

    def load_models(self, model_path_override="", cache_model=False, load_dpvo=False):
        """Download models if needed and return config for GVHMRInference."""

        Log.info("[LoadGVHMRModels] Checking GVHMR models...")

        # Define model paths
        gvhmr_path = MODELS_DIR / "gvhmr" / "gvhmr_siga24_release.ckpt"
        vitpose_path = MODELS_DIR / "vitpose" / "vitpose-h-multi-coco.pth"
        hmr2_path = MODELS_DIR / "hmr2" / "epoch=10-step=25000.ckpt"

        # Override GVHMR path if specified
        if model_path_override and model_path_override.strip():
            gvhmr_path = Path(model_path_override)

        # Check and download models
        self.check_and_download_model("gvhmr", gvhmr_path)
        self.check_and_download_model("vitpose", vitpose_path)
        self.check_and_download_model("hmr2", hmr2_path)

        # Check SMPL models
        self.check_smpl_models()

        # Verify all models exist
        if not all([gvhmr_path.exists(), vitpose_path.exists(), hmr2_path.exists()]):
            raise FileNotFoundError(
                "Not all required models are available. "
                "Please check error messages above or run install.py script."
            )

        Log.info("[LoadGVHMRModels] All models verified!")

        # Load DPVO if requested
        dpvo_model = None
        if load_dpvo:
            dpvo_dir = MODELS_DIR / "dpvo"
            dpvo_model = self.load_dpvo_model(dpvo_dir)
            if dpvo_model:
                Log.info("[LoadGVHMRModels] DPVO model included in config")
            else:
                Log.warn("[LoadGVHMRModels] DPVO requested but could not be loaded")

        # Return config dict (models will be loaded by GVHMRInference)
        config = {
            "models_dir": str(MODELS_DIR),
            "gvhmr_path": str(gvhmr_path),
            "vitpose_path": str(vitpose_path),
            "hmr2_path": str(hmr2_path),
            "body_models_path": str(MODELS_DIR / "body_models"),
            "cache_model": cache_model,
            "dpvo_model": dpvo_model,
        }

        return (config,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LoadGVHMRModels": LoadGVHMRModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadGVHMRModels": "Load GVHMR Models",
}
