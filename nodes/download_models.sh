#!/usr/bin/bash
# Manual download script for GVHMR models
# Run this if install.py fails to download from Google Drive

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Models go in ComfyUI/models/motion_capture/, not in the custom node repo
# From custom_nodes/ComfyUI-MotionCapture/, go up 2 levels to ComfyUI/
COMFYUI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="$COMFYUI_DIR/models/motion_capture"

echo "========================================================================"
echo "  GVHMR Models Manual Download Helper"
echo "========================================================================"
echo ""
echo "Google Drive downloads are currently restricted."
echo "You need to manually download the model files using a browser."
echo ""

# Create directories
mkdir -p "$MODELS_DIR/gvhmr"
mkdir -p "$MODELS_DIR/vitpose"
mkdir -p "$MODELS_DIR/hmr2"

echo "📥 Please download the following files:"
echo ""
echo "1. GVHMR Model (~500MB)"
echo "   URL: https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD"
echo "   File: gvhmr_siga24_release.ckpt"
echo "   Save to: $MODELS_DIR/gvhmr/gvhmr_siga24_release.ckpt"
echo ""
echo "2. ViTPose Model (~650MB)"
echo "   URL: https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD"
echo "   File: vitpose-h-multi-coco.pth"
echo "   Save to: $MODELS_DIR/vitpose/vitpose-h-multi-coco.pth"
echo ""
echo "3. HMR2 Model (~2.3GB)"
echo "   URL: https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD"
echo "   File: epoch=10-step=25000.ckpt"
echo "   Save to: $MODELS_DIR/hmr2/epoch=10-step=25000.ckpt"
echo ""
echo "========================================================================"
echo "After downloading, the directory structure should look like:"
echo ""
echo "models/"
echo "├── gvhmr/"
echo "│   └── gvhmr_siga24_release.ckpt"
echo "├── vitpose/"
echo "│   └── vitpose-h-multi-coco.pth"
echo "├── hmr2/"
echo "│   └── epoch=10-step=25000.ckpt"
echo "└── body_models/"
echo "    ├── smpl/ (already downloaded ✓)"
echo "    └── smplx/ (already downloaded ✓)"
echo ""
echo "========================================================================"
echo ""
echo "Note: SMPL body models have already been downloaded from HuggingFace!"
echo ""

# Check what's already downloaded
echo "Current status:"
[ -f "$MODELS_DIR/gvhmr/gvhmr_siga24_release.ckpt" ] && echo "  ✓ GVHMR model found" || echo "  ✗ GVHMR model missing"
[ -f "$MODELS_DIR/vitpose/vitpose-h-multi-coco.pth" ] && echo "  ✓ ViTPose model found" || echo "  ✗ ViTPose model missing"
[ -f "$MODELS_DIR/hmr2/epoch=10-step=25000.ckpt" ] && echo "  ✓ HMR2 model found" || echo "  ✗ HMR2 model missing"
[ -f "$MODELS_DIR/body_models/smpl/SMPL_NEUTRAL.pkl" ] && echo "  ✓ SMPL models found" || echo "  ✗ SMPL models missing"

echo ""
