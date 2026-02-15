"""
LoadGVHMRModels Node - Downloads and verifies GVHMR model files.

Lightweight node: only validates paths and returns config strings.
No torch, no model loading -- all heavy work happens in GVHMRInference.
"""

import os
import sys
import zipfile
from pathlib import Path
import folder_paths

# Add vendor path for logger only
VENDOR_PATH = Path(__file__).parent / "vendor"
sys.path.insert(0, str(VENDOR_PATH))

MODELS_DIR = Path(folder_paths.models_dir) / "motion_capture"

from hmr4d.utils.pylogger import Log


class LoadGVHMRModels:
    """
    ComfyUI node for checking/downloading GVHMR model files.
    Returns a config dict of paths (strings only) for GVHMRInference.
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
                    "tooltip": "Download DPVO model for moving camera scenarios (~100MB)"
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
            hf_hub_download(
                repo_id=config["repo_id"],
                filename=config["filename"],
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False,
            )
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
            import tempfile
            with tempfile.TemporaryDirectory(dir=str(MODELS_DIR)) as tmp_dir:
                hf_hub_download(
                    repo_id="lithiumice/models_hub",
                    filename=hf_files[model_name],
                    local_dir=tmp_dir,
                    local_dir_use_symlinks=False,
                )
                downloaded = Path(tmp_dir) / hf_files[model_name]
                downloaded.rename(target_path)
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

        for filename in smpl_files:
            file_path = smpl_dir / filename
            if not file_path.exists():
                Log.info(f"[LoadGVHMRModels] {filename} not found, downloading from HuggingFace...")
                if not self.download_smpl_from_hf(filename, file_path):
                    Log.warn(f"[LoadGVHMRModels] Could not auto-download {filename}")

        for filename in smplx_files:
            file_path = smplx_dir / filename
            if not file_path.exists():
                Log.info(f"[LoadGVHMRModels] {filename} not found, downloading from HuggingFace...")
                if not self.download_smpl_from_hf(filename, file_path):
                    Log.warn(f"[LoadGVHMRModels] Could not auto-download {filename}")

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

        Log.info("[LoadGVHMRModels] SMPL body models found")
        return True

    def download_dpvo_checkpoint(self, target_dir: Path) -> bool:
        """Download DPVO checkpoint from Dropbox if missing."""
        checkpoint_path = target_dir / "dpvo.pth"

        if checkpoint_path.exists():
            Log.info(f"[LoadGVHMRModels] DPVO checkpoint found at {checkpoint_path}")
            return True

        Log.info("[LoadGVHMRModels] Downloading DPVO model from Dropbox...")
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            import requests
            from tqdm import tqdm

            url = "https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip?dl=1"
            zip_path = target_dir / "models.zip"

            response = requests.get(url, stream=True, allow_redirects=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(zip_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading DPVO") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

            Log.info("[LoadGVHMRModels] Extracting DPVO model files...")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)

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

            zip_path.unlink()

            if checkpoint_path.exists():
                Log.info(f"[LoadGVHMRModels] DPVO downloaded to {target_dir}")
                return True
            else:
                Log.error("[LoadGVHMRModels] dpvo.pth not found after extraction")
                return False

        except ImportError:
            Log.error("[LoadGVHMRModels] requests package not installed. Run: pip install requests")
            return False
        except Exception as e:
            Log.error(f"[LoadGVHMRModels] DPVO download failed: {e}")
            return False

    def load_models(self, model_path_override="", cache_model=False, load_dpvo=False):
        """Validate model paths and return config dict (strings only)."""

        Log.info("[LoadGVHMRModels] Checking GVHMR models...")

        gvhmr_path = MODELS_DIR / "gvhmr" / "gvhmr_siga24_release.ckpt"
        vitpose_path = MODELS_DIR / "vitpose" / "vitpose-h-multi-coco.pth"
        hmr2_path = MODELS_DIR / "hmr2" / "epoch=10-step=25000.ckpt"

        if model_path_override and model_path_override.strip():
            gvhmr_path = Path(model_path_override)

        self.check_and_download_model("gvhmr", gvhmr_path)
        self.check_and_download_model("vitpose", vitpose_path)
        self.check_and_download_model("hmr2", hmr2_path)

        self.check_smpl_models()

        if not all([gvhmr_path.exists(), vitpose_path.exists(), hmr2_path.exists()]):
            raise FileNotFoundError(
                "Not all required models are available. "
                "Please check error messages above or run install.py script."
            )

        Log.info("[LoadGVHMRModels] All models verified!")

        # Download DPVO checkpoint if requested (but don't load it)
        dpvo_dir = ""
        if load_dpvo:
            dpvo_path = MODELS_DIR / "dpvo"
            if self.download_dpvo_checkpoint(dpvo_path):
                dpvo_dir = str(dpvo_path)
                Log.info(f"[LoadGVHMRModels] DPVO dir: {dpvo_dir}")
            else:
                Log.warn("[LoadGVHMRModels] DPVO requested but checkpoint not available")

        # Return config -- strings and bools only, no tensors or complex objects
        config = {
            "models_dir": str(MODELS_DIR),
            "gvhmr_path": str(gvhmr_path),
            "vitpose_path": str(vitpose_path),
            "hmr2_path": str(hmr2_path),
            "body_models_path": str(MODELS_DIR / "body_models"),
            "cache_model": cache_model,
            "dpvo_dir": dpvo_dir,
        }

        return (config,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LoadGVHMRModels": LoadGVHMRModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadGVHMRModels": "Load GVHMR Models",
}
