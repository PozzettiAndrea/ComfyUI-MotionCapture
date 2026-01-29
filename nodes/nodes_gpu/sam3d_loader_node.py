"""
LoadSAM3DBodyModels Node - Loads and initializes SAM 3D Body model pipeline

Provides MHR 70-keypoint skeleton inference (vs SMPL's 24 joints),
including full hand and face detail.
"""

import os
import sys
from pathlib import Path
import torch

# Add vendor path for SAM3D Body
VENDOR_PATH = Path(__file__).parent / "vendor"
sys.path.insert(0, str(VENDOR_PATH))

# Import logging from GVHMR
from hmr4d.utils.pylogger import Log


# Global cache - persists across node executions
_SAM3D_MODEL_CACHE = {}


class LoadSAM3DBodyModels:
    """
    ComfyUI node for loading SAM 3D Body model.

    SAM 3D Body outputs MHR (Momentum Human Rig) skeleton with 70 keypoints,
    including full hand articulation (21 joints per hand) and body landmarks.

    Downloads models automatically from HuggingFace if missing.
    """

    def __init__(self):
        # Models stored in ComfyUI/models/motion_capture/sam3dbody/
        self.models_dir = Path(__file__).parent.parent.parent.parent / "models" / "motion_capture" / "sam3dbody"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "model_path_override": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional: Override default model path (folder containing model.ckpt)"
                }),
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "HuggingFace token for downloading model. Get from https://huggingface.co/settings/tokens"
                }),
            }
        }

    RETURN_TYPES = ("SAM3D_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_models"
    CATEGORY = "MotionCapture/SAM3D"

    def check_and_download_model(self, model_path: Path, hf_token: str = "") -> bool:
        """Check if model exists, download from HuggingFace if missing."""
        ckpt_path = model_path / "model.ckpt"
        mhr_path = model_path / "assets" / "mhr_model.pt"

        if ckpt_path.exists() and mhr_path.exists():
            Log.info(f"[LoadSAM3DBodyModels] Model found at {model_path}")
            return True

        if not hf_token:
            Log.warn(f"[LoadSAM3DBodyModels] Model not found at {model_path}")
            Log.warn("[LoadSAM3DBodyModels] Provide hf_token to download from HuggingFace")
            return False

        # Try to download
        try:
            from huggingface_hub import snapshot_download

            Log.info("[LoadSAM3DBodyModels] Downloading SAM 3D Body from HuggingFace...")
            Log.info("[LoadSAM3DBodyModels] Repository: facebook/sam-3d-body-dinov3")

            os.environ["HF_TOKEN"] = hf_token
            model_path.mkdir(parents=True, exist_ok=True)

            snapshot_download(
                repo_id="facebook/sam-3d-body-dinov3",
                local_dir=str(model_path),
                token=hf_token,
            )

            Log.info(f"[LoadSAM3DBodyModels] Downloaded to {model_path}")
            return True

        except Exception as e:
            Log.error(f"[LoadSAM3DBodyModels] Download failed: {e}")
            return False

    def load_models(self, model_path_override="", hf_token=""):
        """Load SAM 3D Body model."""

        # Determine model path
        if model_path_override and model_path_override.strip():
            model_path = Path(model_path_override)
        else:
            model_path = self.models_dir

        # Check cache
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_key = f"{model_path}_{device}"

        if cache_key in _SAM3D_MODEL_CACHE:
            Log.info("[LoadSAM3DBodyModels] Using cached model")
            return (_SAM3D_MODEL_CACHE[cache_key],)

        Log.info("[LoadSAM3DBodyModels] Initializing SAM 3D Body model...")

        # Check and download model
        if not self.check_and_download_model(model_path, hf_token):
            error_msg = (
                "\n" + "="*80 + "\n"
                "SAM 3D Body Model Not Found!\n\n"
                "Please download the model:\n\n"
                "Option 1: Provide HuggingFace token\n"
                "  1. Request access at https://huggingface.co/facebook/sam-3d-body-dinov3\n"
                "  2. Get your token from https://huggingface.co/settings/tokens\n"
                "  3. Enter the token in the 'hf_token' input field\n\n"
                "Option 2: Manual download\n"
                "  huggingface-cli download facebook/sam-3d-body-dinov3 --local-dir " +
                f"{model_path}\n\n"
                f"Expected structure:\n"
                f"  {model_path}/\n"
                f"    ├── model.ckpt\n"
                f"    ├── model_config.yaml\n"
                f"    └── assets/\n"
                f"        └── mhr_model.pt\n"
                + "="*80
            )
            raise FileNotFoundError(error_msg)

        # Load the model
        ckpt_path = model_path / "model.ckpt"
        mhr_path = model_path / "assets" / "mhr_model.pt"

        try:
            from sam_3d_body import load_sam_3d_body

            Log.info(f"[LoadSAM3DBodyModels] Loading model from {ckpt_path}...")

            model, model_cfg, mhr_path_used = load_sam_3d_body(
                checkpoint_path=str(ckpt_path),
                device=device,
                mhr_path=str(mhr_path),
            )

            # Get mesh faces for visualization
            faces = model.head_pose.faces.cpu().numpy()

            # Create model bundle
            model_bundle = {
                "model": model,
                "model_cfg": model_cfg,
                "device": device,
                "model_path": str(model_path),
                "mhr_path": str(mhr_path_used),
                "faces": faces,
            }

            # Cache it
            _SAM3D_MODEL_CACHE[cache_key] = model_bundle

            Log.info("[LoadSAM3DBodyModels] SAM 3D Body model loaded successfully!")
            Log.info(f"[LoadSAM3DBodyModels] Device: {device}")
            Log.info("[LoadSAM3DBodyModels] Output: MHR 70-keypoint skeleton (hands + body)")

            return (model_bundle,)

        except ImportError as e:
            raise RuntimeError(
                f"Failed to import sam_3d_body module. "
                f"Check that vendor/sam_3d_body is properly installed.\n"
                f"Error: {e}"
            ) from e
        except Exception as e:
            Log.error(f"[LoadSAM3DBodyModels] Failed to load model: {e}")
            raise


# Node registration
NODE_CLASS_MAPPINGS = {
    "LoadSAM3DBodyModels": LoadSAM3DBodyModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM3DBodyModels": "Load SAM 3D Body Models",
}
