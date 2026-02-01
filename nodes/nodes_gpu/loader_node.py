"""
LoadGVHMRModels Node - Downloads and verifies GVHMR model files
"""

import os
import sys
from pathlib import Path

# Add vendor path for GVHMR
VENDOR_PATH = Path(__file__).parent / "vendor"
sys.path.insert(0, str(VENDOR_PATH))

# Import logger
from hmr4d.utils.pylogger import Log


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

    def __init__(self):
        # Models are stored in ComfyUI/models/motion_capture/, not in the custom node repo
        # Go up 5 levels: loader_node.py -> nodes_gpu -> nodes -> ComfyUI-MotionCapture -> custom_nodes -> ComfyUI
        self.models_dir = Path(__file__).parent.parent.parent.parent.parent / "models" / "motion_capture"

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
                cache_dir=str(self.models_dir / "_hf_cache"),
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
            "SMPL_FEMALE.pkl": "4_SMPLhub/SMPL/X_pkl/SMPL_FEMALE.pkl",
            "SMPL_MALE.pkl": "4_SMPLhub/SMPL/X_pkl/SMPL_MALE.pkl",
            "SMPL_NEUTRAL.pkl": "4_SMPLhub/SMPL/X_pkl/SMPL_NEUTRAL.pkl",
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
                cache_dir=str(self.models_dir / "_hf_cache"),
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
        smpl_dir = self.models_dir / "body_models" / "smpl"
        smplx_dir = self.models_dir / "body_models" / "smplx"

        smpl_files = ["SMPL_FEMALE.pkl", "SMPL_MALE.pkl", "SMPL_NEUTRAL.pkl"]
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
                f"See {self.models_dir}/README.md for detailed instructions.\n"
                + "="*80
            )
            raise FileNotFoundError(error_msg)

        Log.info(f"[LoadGVHMRModels] SMPL body models found")
        return True

    def load_models(self, model_path_override="", cache_model=False):
        """Download models if needed and return config for GVHMRInference."""

        Log.info("[LoadGVHMRModels] Checking GVHMR models...")

        # Define model paths
        gvhmr_path = self.models_dir / "gvhmr" / "gvhmr_siga24_release.ckpt"
        vitpose_path = self.models_dir / "vitpose" / "vitpose-h-multi-coco.pth"
        hmr2_path = self.models_dir / "hmr2" / "epoch=10-step=25000.ckpt"

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

        # Return config dict (models will be loaded by GVHMRInference)
        config = {
            "models_dir": str(self.models_dir),
            "gvhmr_path": str(gvhmr_path),
            "vitpose_path": str(vitpose_path),
            "hmr2_path": str(hmr2_path),
            "body_models_path": str(self.models_dir / "body_models"),
            "cache_model": cache_model,
        }

        return (config,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LoadGVHMRModels": LoadGVHMRModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadGVHMRModels": "Load GVHMR Models",
}
