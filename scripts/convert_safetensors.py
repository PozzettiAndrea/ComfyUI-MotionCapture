#!/usr/bin/env python3
"""Convert MotionCapture (GVHMR) checkpoints to safetensors and upload to HuggingFace.

Usage:
    python convert_safetensors.py                     # Convert only (local)
    python convert_safetensors.py --upload             # Convert + upload to HF
    python convert_safetensors.py --upload --token FILE # Upload with token from file

SMPL body model .npz files are numpy arrays, not model weights — they are copied as-is.
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


SOURCE_REPO = "camenduru/GVHMR"
TARGET_REPO = "apozz/motion-capture-safetensors"

CONVERSIONS = [
    ("gvhmr/gvhmr_siga24_release.ckpt", "gvhmr.safetensors"),
    ("vitpose/vitpose-h-multi-coco.pth", "vitpose.safetensors"),
    ("hmr2/epoch=10-step=25000.ckpt", "hmr2.safetensors"),
]


def convert_one(src_path: Path, dst_path: Path):
    """Load a checkpoint and save as safetensors."""
    print(f"Loading {src_path}...")
    data = torch.load(str(src_path), map_location="cpu", weights_only=False)

    # Extract state_dict from various checkpoint formats
    if isinstance(data, dict):
        if "state_dict" in data:
            state_dict = data["state_dict"]
        elif "model" in data:
            state_dict = data["model"]
        elif "model_state_dict" in data:
            state_dict = data["model_state_dict"]
        else:
            # Assume it's already a flat state dict
            state_dict = data
    else:
        sys.exit(f"ERROR: unexpected format in {src_path}: {type(data)}")

    # Clean: only tensors, must be contiguous
    clean = {}
    skipped = []
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            clean[k] = v.contiguous()
        else:
            skipped.append(k)
    if skipped:
        print(f"  Skipped {len(skipped)} non-tensor keys: {skipped[:5]}...")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {dst_path} ({len(clean)} tensors)...")
    save_file(clean, str(dst_path))
    print(f"  Size: {dst_path.stat().st_size / 1e6:.1f} MB")


def convert(download_dir: Path, output_dir: Path):
    """Download original checkpoints and convert to safetensors."""
    from huggingface_hub import hf_hub_download

    output_dir.mkdir(parents=True, exist_ok=True)

    for src_rel, dst_name in CONVERSIONS:
        # Download original
        print(f"Downloading {src_rel} from {SOURCE_REPO}...")
        local_path = hf_hub_download(
            SOURCE_REPO, src_rel,
            local_dir=str(download_dir), local_dir_use_symlinks=False,
        )
        convert_one(Path(local_path), output_dir / dst_name)

    print("Conversion complete!")
    return output_dir


def upload(output_dir: Path, token: str = None):
    """Upload converted files to HuggingFace."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    try:
        api.create_repo(TARGET_REPO, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")

    print(f"Uploading to {TARGET_REPO}...")
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=TARGET_REPO,
        repo_type="model",
    )
    print(f"Upload complete: https://huggingface.co/{TARGET_REPO}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download-dir", type=Path, default=Path("/tmp/motioncapture_original"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/motioncapture_safetensors"))
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--token", type=str, default=None, help="HF token or path to token file")
    args = parser.parse_args()

    convert(args.download_dir, args.output_dir)

    if args.upload:
        token = args.token
        if token and Path(token).is_file():
            token = Path(token).read_text().strip()
        upload(args.output_dir, token=token)


if __name__ == "__main__":
    main()
