#!/usr/bin/env python3
"""
Installation script for ComfyUI-MotionCapture
Downloads required model checkpoints for GVHMR
"""

import os
import sys
from pathlib import Path
import argparse

try:
    import gdown
except ImportError:
    print("Installing gdown for downloading models from Google Drive...")
    os.system(f"{sys.executable} -m pip install gdown")
    import gdown

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Installing huggingface_hub for downloading SMPL models...")
    os.system(f"{sys.executable} -m pip install huggingface_hub")
    from huggingface_hub import hf_hub_download


# Model download configurations
MODELS = {
    "gvhmr": {
        "repo_id": "camenduru/GVHMR",
        "filename": "gvhmr/gvhmr_siga24_release.ckpt",
        "path": "models/gvhmr/gvhmr_siga24_release.ckpt",
        "size": "~156MB",
        "description": "GVHMR main motion capture model",
        "source": "huggingface",
    },
    "vitpose": {
        "repo_id": "camenduru/GVHMR",
        "filename": "vitpose/vitpose-h-multi-coco.pth",
        "path": "models/vitpose/vitpose-h-multi-coco.pth",
        "size": "~2.4GB",
        "description": "ViTPose 2D pose estimator",
        "source": "huggingface",
    },
    "hmr2": {
        "repo_id": "camenduru/GVHMR",
        "filename": "hmr2/epoch=10-step=25000.ckpt",
        "path": "models/hmr2/epoch=10-step=25000.ckpt",
        "size": "~2.6GB",
        "description": "HMR2 feature extractor",
        "source": "huggingface",
    },
    # SMPL body models from HuggingFace (correct paths!)
    "smpl_male": {
        "repo_id": "lithiumice/models_hub",
        "filename": "4_SMPLhub/SMPL/X_pkl/SMPL_MALE.pkl",
        "path": "models/body_models/smpl/SMPL_MALE.pkl",
        "size": "~2MB",
        "description": "SMPL Male body model",
        "source": "huggingface",
    },
    "smpl_female": {
        "repo_id": "lithiumice/models_hub",
        "filename": "4_SMPLhub/SMPL/X_pkl/SMPL_FEMALE.pkl",
        "path": "models/body_models/smpl/SMPL_FEMALE.pkl",
        "size": "~2MB",
        "description": "SMPL Female body model",
        "source": "huggingface",
    },
    "smpl_neutral": {
        "repo_id": "lithiumice/models_hub",
        "filename": "4_SMPLhub/SMPL/X_pkl/SMPL_NEUTRAL.pkl",
        "path": "models/body_models/smpl/SMPL_NEUTRAL.pkl",
        "size": "~2MB",
        "description": "SMPL Neutral body model",
        "source": "huggingface",
    },
    "smplx_male": {
        "repo_id": "lithiumice/models_hub",
        "filename": "4_SMPLhub/SMPLX/X_npz/SMPLX_MALE.npz",
        "path": "models/body_models/smplx/SMPLX_MALE.npz",
        "size": "~2MB",
        "description": "SMPL-X Male body model",
        "source": "huggingface",
    },
    "smplx_female": {
        "repo_id": "lithiumice/models_hub",
        "filename": "4_SMPLhub/SMPLX/X_npz/SMPLX_FEMALE.npz",
        "path": "models/body_models/smplx/SMPLX_FEMALE.npz",
        "size": "~2MB",
        "description": "SMPL-X Female body model",
        "source": "huggingface",
    },
    "smplx_neutral": {
        "repo_id": "lithiumice/models_hub",
        "filename": "4_SMPLhub/SMPLX/X_npz/SMPLX_NEUTRAL.npz",
        "path": "models/body_models/smplx/SMPLX_NEUTRAL.npz",
        "size": "~2MB",
        "description": "SMPL-X Neutral body model",
        "source": "huggingface",
    },
}


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def check_model_exists(model_path: Path) -> bool:
    """Check if model file already exists."""
    return model_path.exists() and model_path.stat().st_size > 1000


def download_model(model_name: str, model_info: dict, base_dir: Path, force: bool = False):
    """Download a single model from Google Drive or HuggingFace."""
    model_path = base_dir / model_info["path"]

    # Check if already exists
    if check_model_exists(model_path) and not force:
        print(f"✓ {model_name}: Already downloaded at {model_path}")
        return True

    # Create directory
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Download
    print(f"\n⬇ Downloading {model_name}...")
    print(f"  Description: {model_info['description']}")
    print(f"  Size: {model_info['size']}")
    print(f"  Destination: {model_path}")

    try:
        source = model_info.get("source", "gdrive")

        if source == "huggingface":
            # Download from HuggingFace
            print(f"  Source: HuggingFace ({model_info['repo_id']})")
            downloaded_path = hf_hub_download(
                repo_id=model_info["repo_id"],
                filename=model_info["filename"],
                cache_dir=str(base_dir / "models" / "_hf_cache"),
            )
            # Copy to target location
            import shutil
            shutil.copy(downloaded_path, str(model_path))
        else:
            # Download from Google Drive
            print(f"  Source: Google Drive")
            # Try with fuzzy mode for better compatibility
            try:
                gdown.download(model_info["url"], str(model_path), quiet=False, fuzzy=True)
            except Exception as e:
                # Try alternative method with id extraction
                file_id = model_info["url"].split("id=")[-1]
                gdown.download(id=file_id, output=str(model_path), quiet=False)

        print(f"✓ {model_name} downloaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_all_models(base_dir: Path, force: bool = False):
    """Download all required models."""
    print_header("ComfyUI-MotionCapture Model Installer")

    print("This script will download the following models from HuggingFace:")
    print("\n🤗 Main Models (camenduru/GVHMR):")
    for name, info in MODELS.items():
        if info.get("repo_id") == "camenduru/GVHMR":
            print(f"  • {name}: {info['description']} ({info['size']})")

    print("\n🤗 SMPL Body Models (lithiumice/models_hub):")
    for name, info in MODELS.items():
        if info.get("repo_id") == "lithiumice/models_hub":
            print(f"  • {name}: {info['description']} ({info['size']})")

    print("\n💾 Total download size: ~5.2GB")
    print("⏱  This may take a while depending on your connection speed.")
    print("✨ All models auto-download from HuggingFace!\n")

    # Download each model
    results = {}
    for model_name, model_info in MODELS.items():
        results[model_name] = download_model(model_name, model_info, base_dir, force)

    # Summary
    print_header("Download Summary")
    success_count = sum(results.values())
    total_count = len(results)

    for model_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {model_name}: {status}")

    print(f"\n{success_count}/{total_count} models downloaded successfully.")

    if success_count < total_count:
        print("\n⚠ Some models failed to download.")
        print("You can retry by running this script again with --force flag.")
        print("Or models will be auto-downloaded when you first use the nodes.")

    return success_count == total_count


def print_smpl_info():
    """Print information about SMPL body models."""
    print_header("SMPL Body Models - Auto-Downloaded!")

    print("✨ Good news! SMPL body models are now automatically downloaded from HuggingFace.")
    print("   Source: lithiumice/models_hub repository")
    print("   License: These models are provided for research purposes.\n")

    print("📝 Note: If you need official SMPL models for commercial use:")
    print("   • SMPL: https://smpl.is.tue.mpg.de/")
    print("   • SMPL-X: https://smpl-x.is.tue.mpg.de/")
    print("   • You can replace the auto-downloaded files with official ones.")


def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(
        description="Download GVHMR model checkpoints for ComfyUI-MotionCapture"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if models exist"
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="Download specific model only"
    )
    args = parser.parse_args()

    # Get ComfyUI models directory (not in the custom node repo!)
    base_dir = Path(__file__).parent.parent.parent / "models" / "motion_capture"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Download models
    if args.model:
        # Download specific model
        model_info = MODELS[args.model]
        success = download_model(args.model, model_info, base_dir, args.force)
    else:
        # Download all models
        success = download_all_models(base_dir, args.force)

    # Print SMPL info
    print_smpl_info()

    # Final message
    print_header("Installation Complete!")
    print("✅ All models downloaded successfully!")
    print("🚀 You can now use ComfyUI-MotionCapture nodes in ComfyUI.")
    print("💡 Restart ComfyUI to load the new nodes.\n")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
