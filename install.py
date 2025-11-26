#!/usr/bin/env python3
"""
Installation script for ComfyUI-MotionCapture
Downloads required model checkpoints for GVHMR and optionally installs Blender for FBX retargeting
"""

import os
import sys
from pathlib import Path
import argparse
import platform
import urllib.request
import tarfile
import zipfile
import shutil
import subprocess

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
    # Note: SMPL-X models removed - not needed for basic motion capture
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
        print(f"[OK] {model_name}: Already downloaded at {model_path}")
        return True

    # Create directory
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Download
    print(f"\n[DOWNLOADING] {model_name}...")
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

        print(f"[OK] {model_name} downloaded successfully!")
        return True
    except Exception as e:
        print(f"[FAILED] Failed to download {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_all_models(base_dir: Path, force: bool = False):
    """Download all required models."""
    print_header("ComfyUI-MotionCapture Model Installer")

    print("This script will download the following models from HuggingFace:")
    print("\nMain Models (camenduru/GVHMR):")
    for name, info in MODELS.items():
        if info.get("repo_id") == "camenduru/GVHMR":
            print(f"  - {name}: {info['description']} ({info['size']})")

    print("\nSMPL Body Models (lithiumice/models_hub):")
    for name, info in MODELS.items():
        if info.get("repo_id") == "lithiumice/models_hub":
            print(f"  - {name}: {info['description']} ({info['size']})")

    print("\nTotal download size: ~5.2GB")
    print("This may take a while depending on your connection speed.")
    print("All models auto-download from HuggingFace!\n")

    # Download each model
    results = {}
    for model_name, model_info in MODELS.items():
        results[model_name] = download_model(model_name, model_info, base_dir, force)

    # Summary
    print_header("Download Summary")
    success_count = sum(results.values())
    total_count = len(results)

    for model_name, success in results.items():
        status = "[OK] SUCCESS" if success else "[FAILED]"
        print(f"  {model_name}: {status}")

    print(f"\n{success_count}/{total_count} models downloaded successfully.")

    if success_count < total_count:
        print("\n[WARNING] Some models failed to download.")
        print("You can retry by running this script again with --force flag.")
        print("Or models will be auto-downloaded when you first use the nodes.")

    return success_count == total_count


def print_smpl_info():
    """Print information about SMPL body models."""
    print_header("SMPL Body Models - Auto-Downloaded!")

    print("Good news! SMPL body models are now automatically downloaded from HuggingFace.")
    print("   Source: lithiumice/models_hub repository")
    print("   License: These models are provided for research purposes.\n")

    print("Note: If you need official SMPL models for commercial use:")
    print("   - SMPL: https://smpl.is.tue.mpg.de/")
    print("   - SMPL-X: https://smpl-x.is.tue.mpg.de/")
    print("   - You can replace the auto-downloaded files with official ones.")


# Blender Installation Functions

def get_platform_info():
    """Detect current platform and architecture."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        plat = "macos"
        arch = "arm64" if machine == "arm64" else "x64"
    elif system == "linux":
        plat = "linux"
        arch = "x64"
    elif system == "windows":
        plat = "windows"
        arch = "x64"
    else:
        plat = None
        arch = None

    return plat, arch


def get_blender_download_url(platform_name, architecture):
    """
    Get Blender 4.2 LTS download URL for the platform.

    Args:
        platform_name: "linux", "macos", or "windows"
        architecture: "x64" or "arm64"

    Returns:
        tuple: (download_url, version, filename) or (None, None, None) if not found
    """
    version = "4.2.3"
    base_url = "https://download.blender.org/release/Blender4.2"

    urls = {
        ("linux", "x64"): (
            f"{base_url}/blender-{version}-linux-x64.tar.xz",
            version,
            f"blender-{version}-linux-x64.tar.xz"
        ),
        ("macos", "x64"): (
            f"{base_url}/blender-{version}-macos-x64.dmg",
            version,
            f"blender-{version}-macos-x64.dmg"
        ),
        ("macos", "arm64"): (
            f"{base_url}/blender-{version}-macos-arm64.dmg",
            version,
            f"blender-{version}-macos-arm64.dmg"
        ),
        ("windows", "x64"): (
            f"{base_url}/blender-{version}-windows-x64.zip",
            version,
            f"blender-{version}-windows-x64.zip"
        ),
    }

    key = (platform_name, architecture)
    if key in urls:
        url, ver, filename = urls[key]
        print(f"[MotionCapture Install] Using Blender {ver} for {platform_name}-{architecture}")
        return url, ver, filename

    return None, None, None


def download_file(url, dest_path):
    """Download file with progress."""
    print(f"[MotionCapture Install] Downloading: {url}")
    print(f"[MotionCapture Install] Destination: {dest_path}")

    last_printed_percent = [-1]  # Use list to allow modification in nested function

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)

        # Only print every 10% to reduce verbosity
        if percent >= last_printed_percent[0] + 10 or percent >= 100:
            sys.stdout.write(f"\r[MotionCapture Install] Progress: {percent}%")
            sys.stdout.flush()
            last_printed_percent[0] = percent

    try:
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        sys.stdout.write("\n")
        sys.stdout.flush()
        print("[MotionCapture Install] Download complete!")
        return True
    except Exception as e:
        print(f"\n[MotionCapture Install] Error downloading: {e}")
        return False


def extract_archive(archive_path, extract_to):
    """Extract tar.gz, tar.xz, zip, or handle DMG (macOS)."""
    print(f"[MotionCapture Install] Extracting: {archive_path}")

    try:
        if archive_path.endswith(('.tar.gz', '.tar.xz', '.tar.bz2')):
            with tarfile.open(archive_path, 'r:*') as tar:
                tar.extractall(extract_to)
        elif archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith('.dmg'):
            print("[MotionCapture Install] DMG detected - mounting disk image...")

            mount_result = subprocess.run(
                ['hdiutil', 'attach', '-nobrowse', archive_path],
                capture_output=True,
                text=True
            )

            if mount_result.returncode != 0:
                print(f"[MotionCapture Install] Error mounting DMG: {mount_result.stderr}")
                return False

            mount_point = None
            for line in mount_result.stdout.split('\n'):
                if '/Volumes/' in line:
                    mount_point = line.split('\t')[-1].strip()
                    break

            if not mount_point:
                print("[MotionCapture Install] Error: Could not find mount point")
                return False

            try:
                blender_app = Path(mount_point) / "Blender.app"
                if blender_app.exists():
                    dest_app = Path(extract_to) / "Blender.app"
                    shutil.copytree(blender_app, dest_app)
                    print(f"[MotionCapture Install] Copied Blender.app to: {dest_app}")
                else:
                    print(f"[MotionCapture Install] Error: Blender.app not found in {mount_point}")
                    return False

            finally:
                subprocess.run(['hdiutil', 'detach', mount_point], check=False)

        else:
            print(f"[MotionCapture Install] Error: Unknown archive format: {archive_path}")
            return False

        print(f"[MotionCapture Install] Extraction complete!")
        return True

    except Exception as e:
        print(f"[MotionCapture Install] Error extracting: {e}")
        return False


def find_blender_executable(blender_dir):
    """Find the blender executable in the extracted directory."""
    plat, _ = get_platform_info()

    if plat == "windows":
        exe_pattern = "**/blender.exe"
    elif plat == "macos":
        exe_pattern = "**/MacOS/blender"
    else:  # linux
        exe_pattern = "**/blender"

    executables = list(Path(blender_dir).glob(exe_pattern))

    if executables:
        return executables[0]
    return None


def install_blender(target_dir=None):
    """
    Install Blender for FBX retargeting.

    Args:
        target_dir: Optional target directory. If None, uses lib/blender under script directory.

    Returns:
        str: Path to Blender executable, or None if installation failed.
    """
    print("\n" + "="*60)
    print("ComfyUI-MotionCapture: Blender Installation")
    print("="*60 + "\n")

    if target_dir is None:
        script_dir = Path(__file__).parent.absolute()
        target_dir = script_dir / "lib" / "blender"
    else:
        target_dir = Path(target_dir)

    # Check if Blender already installed
    blender_exe = find_blender_executable(target_dir)
    if blender_exe and blender_exe.exists():
        print("[MotionCapture Install] Blender already installed at:")
        print(f"[MotionCapture Install]   {blender_exe}")
        print("[MotionCapture Install] Skipping download.")
        return str(blender_exe)

    # Detect platform
    plat, arch = get_platform_info()
    if not plat or not arch:
        print("[MotionCapture Install] Error: Could not detect platform")
        print("[MotionCapture Install] Please install Blender manually from: https://www.blender.org/download/")
        return None

    print(f"[MotionCapture Install] Detected platform: {plat}-{arch}")

    # Get download URL
    url, version, filename = get_blender_download_url(plat, arch)
    if not url:
        print("[MotionCapture Install] Error: Could not find Blender download for your platform")
        print("[MotionCapture Install] Please install Blender manually from: https://www.blender.org/download/")
        return None

    # Create temporary download directory
    temp_dir = target_dir.parent / "_temp_blender_download"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download
        download_path = temp_dir / filename
        if not download_file(url, str(download_path)):
            return None

        # Extract
        target_dir.mkdir(parents=True, exist_ok=True)
        if not extract_archive(str(download_path), str(target_dir)):
            return None

        print("\n[MotionCapture Install] Blender installation complete!")
        print(f"[MotionCapture Install] Location: {target_dir}")

        # Find blender executable
        blender_exe = find_blender_executable(target_dir)

        if blender_exe:
            print(f"[MotionCapture Install] Blender executable: {blender_exe}")
            return str(blender_exe)
        else:
            print("[MotionCapture Install] Warning: Could not find blender executable")
            return None

    except Exception as e:
        print(f"\n[MotionCapture Install] Error during installation: {e}")
        return None

    finally:
        # Cleanup temp files
        if temp_dir.exists():
            print("[MotionCapture Install] Cleaning up temporary files...")
            shutil.rmtree(temp_dir, ignore_errors=True)


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
    parser.add_argument(
        "--install-blender-addons",
        action="store_true",
        help="Install Blender addons (VRM and BVH Retargeter) for enhanced retargeting"
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

    # Install Blender automatically
    blender_path = install_blender()
    if blender_path:
        print(f"\n[OK] Blender installed successfully at: {blender_path}")
    else:
        print("\n[WARNING] Blender installation failed. You can:")
        print("  - Install manually from https://www.blender.org/download/")
        print("  - Or run: python install.py")

    # Install Blender addons if requested
    if args.install_blender_addons:
        print("\n" + "="*60)
        print("Installing Blender Addons for BVH Retargeting")
        print("="*60 + "\n")

        try:
            from lib.blender_addon_installer import install_all_addons
            success = install_all_addons()
            if success:
                print("\n[OK] Blender addons installed successfully!")
                print("  - VRM Addon: Import/export VRM character files")
                print("  - BVH Retargeter: Advanced BVH motion retargeting")
            else:
                print("\n[WARNING] Some addons failed to install. Check logs above for details.")
        except Exception as e:
            print(f"\n[WARNING] Failed to install Blender addons: {e}")
            print("  You can install them manually:")
            print("  - VRM Addon: https://github.com/saturday06/VRM-Addon-for-Blender")
            print("  - BVH Retargeter: https://github.com/Diffeomorphic/retarget-bvh")

    # Final message
    print_header("Installation Complete!")
    print("[OK] All models downloaded successfully!")
    print("You can now use ComfyUI-MotionCapture nodes in ComfyUI.")
    print("Blender is now installed for FBX retargeting support.")
    if args.install_blender_addons:
        print("Blender addons installed for enhanced BVH retargeting.")
    else:
        print("[TIP] For BVH->VRM retargeting, run: python install.py --install-blender-addons")
    print("[TIP] Restart ComfyUI to load the new nodes.\n")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
