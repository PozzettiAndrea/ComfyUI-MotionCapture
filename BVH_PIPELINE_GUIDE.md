# BVH Pipeline Guide

Complete guide for using the SMPL → BVH → VRoid retargeting pipeline.

## Overview

The BVH pipeline provides better motion retargeting quality by using the industry-standard BVH format as an intermediate representation:

```
Video → GVHMR → SMPL → BVH → VRM/FBX Character
```

## Installation

### 1. Install Dependencies

```bash
cd /home/shadeform/mocap_node/ComfyUI/custom_nodes/ComfyUI-MotionCapture
pip install -r requirements.txt
```

### 2. Install Blender (Required for Retargeting)

```bash
python install.py --install-blender
```

This downloads and installs Blender 4.2 LTS locally.

### 3. Install Blender Addons (Required for BVH → VRM)

```bash
python install.py --install-blender-addons
```

This installs:
- **VRM Addon**: For importing/exporting VRM character files
- **BVH Retargeter**: For advanced BVH motion retargeting

## Workflow Setup

### Step 1: Generate SMPL Motion Data

First, you need SMPL motion data. You can either:

**Option A: Run Motion Capture from Video**

1. Add `LoadGVHMRModels` node
2. Add `GVHMRInference` node
3. Connect them and process your video
4. Use `SaveSMPL` to save the output as `.npz`

**Option B: Use Existing NPZ File**

If you already have SMPL data:

1. Add `LoadSMPL` node
2. Click "refresh" button
3. Select your `.npz` file from the dropdown

### Step 2: Convert SMPL to BVH

1. Add `SMPLtoBVH` node
2. Connect `smpl_params` from LoadSMPL (or GVHMRInference)
3. Configure parameters:
   - **output_path**: Where to save BVH file (default: `output/motion.bvh`)
   - **fps**: Frame rate (default: 30)
   - **scale**: Skeleton scale factor (1.0 = meters, 100.0 = centimeters)

### Step 3: Preview BVH Animation (Optional)

1. Add `BVHViewer` node
2. Connect `bvh_data` from SMPLtoBVH
3. Visualize the skeleton animation before retargeting
4. Use playback controls to verify motion quality

### Step 4: Load Your Character

1. Add `LoadFBXCharacter` node
2. Click "refresh" to browse available characters
3. Select your VRM or FBX character file

**Supported Character Types:**
- VRM (VRoid models)
- FBX (with humanoid rig)

### Step 5: Retarget BVH to Character

1. Add `BVHtoFBX` node
2. Connect inputs:
   - `bvh_data` from SMPLtoBVH
   - `character_path` from LoadFBXCharacter
3. Configure parameters:
   - **output_path**: Where to save result (e.g., `output/retargeted.fbx`)
   - **character_type**: Usually "auto" (auto-detects VRM/FBX)
   - **output_format**: "fbx" or "vrm"

### Complete Node Graph

```
┌─────────────┐     ┌────────────┐     ┌───────────┐
│  LoadSMPL   │────▶│ SMPLtoBVH  │────▶│ BVHViewer │
└─────────────┘     └─────┬──────┘     └───────────┘
                          │
                          ▼
                    ┌───────────┐
                    │ BVHtoFBX  │◀───┐
                    └───────────┘    │
                                     │
                         ┌───────────┴──────┐
                         │ LoadFBXCharacter │
                         └──────────────────┘
```

## Troubleshooting

### Error: "BVH file not found"

**Cause**: The SMPLtoBVH node failed to execute.

**Solution**:
1. Make sure LoadSMPL successfully loaded an NPZ file
2. Check the SMPLtoBVH node output for errors
3. Verify the output directory exists and is writable

### Error: "Blender not found"

**Cause**: Blender is not installed or not in PATH.

**Solution**:
```bash
python install.py --install-blender
```

### Error: "VRM addon export not available"

**Cause**: VRM addon is not installed in Blender.

**Solution**:
```bash
python install.py --install-blender-addons
```

### Error: "No armature found in character file"

**Cause**: The character FBX/VRM file doesn't contain a skeleton.

**Solution**: Use a rigged character model. VRoid characters always have armatures.

### Warning: "IS_CHANGED() missing argument"

**Cause**: Old cached node definition.

**Solution**: Restart ComfyUI to reload the updated node code.

## Tips for Best Results

### 1. Scale Adjustment

Different characters may need different scales:
- **VRoid characters**: Usually work well with scale `1.0`
- **Mixamo characters**: May need scale `0.01` or `100.0`
- Experiment to find the right scale for your character

### 2. FPS Matching

Make sure the FPS in SMPLtoBVH matches your source video:
- Most videos: 30 FPS
- Cinema/film: 24 FPS
- Gaming: 60 FPS

### 3. BVH Viewer Usage

Use the BVH viewer to check:
- Motion looks correct before retargeting
- No excessive jitter or noise
- Root motion is present (character moves through space)

### 4. Output Format Selection

- Use **FBX** for maximum compatibility (Unity, Unreal, Blender, etc.)
- Use **VRM** if you need to maintain VRM-specific features (springs, blend shapes)

## Advanced Usage

### Custom Bone Mapping

If retargeting fails due to bone naming issues, you can:

1. Import BVH and character into Blender manually
2. Use BVH Retargeter addon UI for custom bone mapping
3. Save the mapping configuration
4. Modify the BVHtoFBX node script to use your mapping

### Batch Processing

To process multiple NPZ files:

1. Create a ComfyUI workflow with the BVH pipeline
2. Use ComfyUI's batch processing features
3. Or write a Python script that loads/executes the workflow multiple times

### Export BVH Only

If you just want BVH files without retargeting:

1. Use `SMPLtoBVH` node alone
2. The BVH files can be used in:
   - Unity (with BVH importer)
   - Unreal Engine
   - Blender
   - Motion Builder
   - Any other 3D software supporting BVH

## Example Workflows

### Quick Test Workflow

Load the example workflow:
```
workflows/smpl_to_bvh_pipeline.json
```

This shows the complete pipeline from SMPL → BVH → VRM.

### Full Video-to-VRM Workflow

1. LoadGVHMRModels
2. GVHMRInference (process video)
3. SaveSMPL (save NPZ)
4. SMPLtoBVH (convert to BVH)
5. BVHViewer (preview)
6. LoadFBXCharacter (load VRoid model)
7. BVHtoFBX (retarget and export)

## FAQ

**Q: Why use BVH instead of direct SMPL → VRoid mapping?**

A: BVH provides:
- Better retargeting quality with mature tools
- Compatibility with industry-standard software
- Easier debugging and iteration
- Reusable motion files across multiple characters

**Q: Can I use this with non-VRoid characters?**

A: Yes! It works with any rigged FBX character. The BVH Retargeter addon supports:
- Mixamo rigs
- Rigify (Blender)
- UE5 Mannequin
- Custom rigs (with manual bone mapping)

**Q: Do I need to download SMPL models separately?**

A: No, they're automatically downloaded from HuggingFace when you run `install.py`.

**Q: Can I export to game engines?**

A: Yes! The output FBX files work in:
- Unity (drag and drop)
- Unreal Engine (import FBX)
- Godot (import FBX)
- Any engine supporting FBX

**Q: What if the retargeting quality is poor?**

A: Try:
1. Adjusting the scale parameter in SMPLtoBVH
2. Using BVH Retargeter's manual bone mapping in Blender
3. Checking if your character's T-pose matches SMPL's T-pose
4. Increasing the FPS for smoother motion

## Support

For issues or questions:
1. Check the error messages in ComfyUI console
2. Verify all installation steps completed
3. Test with the example workflow first
4. Check that Blender and addons are properly installed

## License

This pipeline uses:
- SMPL body models (research license)
- VRM Addon (MIT license)
- BVH Retargeter (GPL license)
- Blender (GPL license)

Make sure to comply with respective licenses for your use case.
