#!/usr/bin/env python3
"""
Installation script for ComfyUI-MotionCapture.

Uses comfy-env for declarative dependency management via comfy-env.toml.
Models are downloaded on-demand by Loader nodes when first used.
"""

import sys
from pathlib import Path


def main():
    print("\n" + "=" * 60)
    print("ComfyUI-MotionCapture Installation")
    print("=" * 60)

    from comfy_env import install

    node_root = Path(__file__).parent.absolute()

    # Run comfy-env install to create isolated environment
    try:
        install(config=node_root / "comfy-env.toml", mode="isolated", node_dir=node_root)
    except Exception as e:
        print(f"\n[MotionCapture] Installation FAILED: {e}")
        print("[MotionCapture] Report issues at: https://github.com/ComfyUI-MotionCapture/issues")
        return 1

    print("\n" + "=" * 60)
    print("[MotionCapture] Installation completed!")
    print("[MotionCapture] Models will be downloaded on first use.")
    print("[MotionCapture] Restart ComfyUI to load the new nodes.")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
