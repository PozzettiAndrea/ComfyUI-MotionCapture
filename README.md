# ComfyUI-MotionCapture

**[View Workflow Gallery](https://pozzettiandrea.github.io/ComfyUI-MotionCapture/)**

A ComfyUI custom node package for GVHMR-based 3D human motion capture from video. Extract SMPL parameters and 3D skeletal motion using state-of-the-art pose estimation.



https://github.com/user-attachments/assets/17638ca5-8139-40ca-b215-0d7dabf0ea73


https://github.com/user-attachments/assets/ba7a7797-713d-4750-9210-ff07bcc6bb01



## Features

- **GVHMR Integration**: World-grounded human motion recovery with gravity-view coordinates
- **SAM3 Powered**: Uses SAM3 segmentation masks for accurate person tracking (no YOLO dependency)
- **Full Pipeline**: Includes 2D pose estimation (ViTPose), feature extraction, and 3D reconstruction
- **SMPL Output**: Generates SMPL parameters for 3D mesh and skeletal animation
- **Visualization**: Renders 3D mesh overlay on input video frames
- **All-in-One**: Complete vendored package with all GVHMR code included

## Architecture

### Nodes

1. **LoadGVHMRModels**: Loads GVHMR model pipeline (downloads models if missing)
2. **GVHMRInference**: Runs motion capture inference on video with SAM3 masks

### Pipeline Flow

```
Input Video Frames → SAM3 Segmentation → GVHMR Inference → SMPL Parameters + 3D Mesh
                         ↓
                    Bounding Boxes
                         ↓
                    ViTPose (2D Keypoints)
                         ↓
                    ViT Feature Extraction
                         ↓
                    GVHMR Model
```

## Installation

### 1. Clone/Copy to ComfyUI Custom Nodes

```bash
cd ComfyUI/custom_nodes/
# This package should already be at ComfyUI-MotionCapture/
```

### 2. Install Python Dependencies

```bash
cd ComfyUI-MotionCapture
pip install -r requirements.txt
```

**Note**: This will install ~30+ dependencies including PyTorch3D, Lightning, Hydra, etc.
The requirements.txt includes specific CUDA 12.1 versions. Adjust if needed for your setup.

### 3. Download Model Checkpoints

Run the automated installer:

```bash
python install.py
```

This downloads:
- **Main Models** (Google Drive):
  - GVHMR checkpoint (~500MB)
  - ViTPose checkpoint (~650MB)
  - HMR2 feature extractor (~2.3GB)
- **SMPL Body Models** (HuggingFace):
  - All SMPL and SMPL-X models (~12MB total)

**Total download**: ~3.5GB

Or download specific models:
```bash
python install.py --model gvhmr
python install.py --model vitpose
python install.py --model hmr2
python install.py --model smpl_male
python install.py --model smplx_neutral
```

### 4. ✨ SMPL Models (Now Auto-Downloaded!)

**Great news**: SMPL body models are now **automatically downloaded** from HuggingFace!

- **Source**: `lithiumice/models_hub` repository
- **License**: Research purposes (community-provided)
- **Auto-download**: Both the installer and node will download if missing

**For commercial use**, you may want official SMPL models:
- SMPL: https://smpl.is.tue.mpg.de/ (requires registration)
- SMPL-X: https://smpl-x.is.tue.mpg.de/ (requires registration)
- Simply replace the auto-downloaded files with official ones

### 5. Restart ComfyUI

After installation, restart ComfyUI to load the new nodes.

## Usage

### Basic Workflow

1. **Load Video**: Use ComfyUI video loader to load your input video
2. **Segment Person**: Use SAM3 node to generate person segmentation masks
3. **Load Models**: Connect `LoadGVHMRModels` node
4. **Run Inference**: Connect video frames, SAM3 masks, and model to `GVHMRInference`
5. **Output**: Get SMPL parameters, 3D mesh, and visualization

### Node Parameters

#### LoadGVHMRModels
- **model_path_override** (optional): Override default GVHMR checkpoint path

#### GVHMRInference
- **images** (required): Video frames as IMAGE tensor
- **masks** (required): SAM3 segmentation masks as MASK tensor
- **model** (required): Model bundle from LoadGVHMRModels
- **static_camera** (default: True): Set to True if camera is stationary
- **focal_length_mm** (default: 0): Camera focal length in mm (0 = auto)
  - For smartphones: typically 13-77mm
  - For zoom cameras: 135-300mm
- **bbox_scale** (default: 1.2): Expand bounding box by this factor

### Example Workflow

```
VideoLoader → SAM3Segmentation → [images, masks]
                                         ↓
LoadGVHMRModels → [model] ───────→ GVHMRInference
                                         ↓
                            [smpl_params, visualization, info]
```

## Outputs

### SMPL_PARAMS
Dictionary containing:
- **global**: Global SMPL parameters (world coordinates)
  - `body_pose`: Body joint rotations (B, L, 63)
  - `betas`: Shape parameters (B, L, 10)
  - `global_orient`: Global orientation (B, L, 3)
  - `transl`: Global translation (B, L, 3)
- **incam**: In-camera SMPL parameters (camera coordinates)
- **K_fullimg**: Camera intrinsics matrix

### Visualization
IMAGE tensor with 3D mesh rendered over input frames

### Info
String with processing information (frame count, settings, etc.)

## Directory Structure

```
ComfyUI/
├── models/
│   └── motion_capture/        # Models stored here (not in custom_nodes!)
│       ├── gvhmr/
│       ├── vitpose/
│       ├── hmr2/
│       └── body_models/
│
└── custom_nodes/
    └── ComfyUI-MotionCapture/
        ├── __init__.py        # ComfyUI entry point
        ├── requirements.txt   # Python dependencies
        ├── install.py         # Model download script
        ├── .gitignore        # Excludes models from repo
        ├── README.md         # This file
        │
        ├── nodes/
        │   ├── loader_node.py     # LoadGVHMRModels node
        │   ├── inference_node.py  # GVHMRInference node
        │   └── utils.py          # Utility functions
        │
        └── vendor/
            └── hmr4d/         # Vendored GVHMR codebase
```

## Technical Details

### GVHMR Model
- **Paper**: "World-Grounded Human Motion Recovery via Gravity-View Coordinates" (SIGGRAPH Asia 2024)
- **Repository**: https://github.com/zju3dv/GVHMR
- **Innovation**: Gravity-View coordinate system for world-grounded pose estimation
- **Performance**: Processes ~1430 frames in 280ms on RTX 4090

### Dependencies
- **PyTorch**: Deep learning framework
- **PyTorch3D**: 3D transformations and rendering
- **Lightning**: Training framework (used by GVHMR)
- **Hydra**: Configuration management
- **ViTPose**: 2D pose estimation
- **SMPL/SMPL-X**: Parametric human body models

## Troubleshooting

### "SMPL Body Models Not Found"
- You must manually download SMPL models (see Installation step 4)
- Check that files are in correct directories with correct names

### "Failed to download model checkpoints"
- Check internet connection
- Try manual download from the URLs in `install.py`
- Or let nodes auto-download on first use

### "CUDA out of memory"
- Reduce batch size (process fewer frames at once)
- Use smaller video resolution
- Close other GPU-intensive applications

### "ViTPose extraction failed"
- Ensure bounding boxes from SAM3 masks are valid
- Check that person is clearly visible in frames
- Try adjusting bbox_scale parameter

## Performance

- **GPU Required**: CUDA-capable GPU recommended (tested on RTX 4090)
- **Processing Speed**: ~5 FPS on RTX 4090 (depends on video resolution)
- **Memory Usage**: ~8-12GB VRAM for 1080p video
- **Supported Resolutions**: 480p to 4K (higher = slower but more accurate)

## Limitations

- **Single Person**: Currently processes one person per video
- **Visible Person**: Requires person to be mostly visible (SAM3 must segment)
- **No Occlusion Handling**: Heavy occlusions may cause artifacts
- **Static Scenes Preferred**: Moving camera support is experimental

## Credits

### GVHMR
```
@inproceedings{qiu2024gvhmr,
  title={World-Grounded Human Motion Recovery via Gravity-View Coordinates},
  author={Qiu, Zehong and Wang, Qingshan and Peng, Zhenbo and others},
  booktitle={SIGGRAPH Asia},
  year={2024}
}
```

### Components
- **ViTPose**: https://github.com/ViTAE-Transformer/ViTPose
- **HMR2**: https://github.com/shubham-goel/4D-Humans
- **SMPL**: https://smpl.is.tue.mpg.de/
- **SMPL-X**: https://smpl-x.is.tue.mpg.de/

## License

This package vendors GVHMR code which has its own license. Please check the original GVHMR repository for licensing terms.

SMPL and SMPL-X body models require separate licenses from their respective providers.

## Support

For issues and questions:
- Check the [GVHMR repository](https://github.com/zju3dv/GVHMR)
- Open an issue on this repository
- Ensure all dependencies and models are properly installed

## Version

**Current Version**: 0.1.0 (First Draft)

This is an initial implementation. Future updates may include:
- Multi-person support
- Real-time webcam processing
- Additional export formats (FBX, BVH, glTF)
- Improved visualization options
- Better error handling and recovery
