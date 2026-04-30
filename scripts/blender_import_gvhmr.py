"""
GVHMR → Blender Import Script
==============================
Imports a rigged + animated FBX character and animated camera poses produced
by the ComfyUI-MotionCapture pipeline into the current Blender scene.

Usage
-----
1. Edit the CONFIGURE section below with your file paths and FPS.
2. Open Blender's Scripting workspace.
3. Paste / open this file, then click "Run Script".

What this script does
---------------------
- Optionally imports the animated FBX (rigged character retargeted with SMPL
  motion) produced by the "BVH to FBX Retargeter" node.
- Creates a Blender Camera object animated with the predicted camera trajectory
  (position, rotation, and focal length keyframed per frame).
- Sets the scene frame range and FPS to match the captured clip.

Workflow in ComfyUI
-------------------
1. GVHMR Inference        →  npz_path  +  camera_npz_path
2. SMPL to BVH Converter  →  bvh_data           (takes npz_path as input)
3. BVH to FBX Retargeter  →  output_path (.fbx)  (takes bvh_data + your
                                                   rigged character FBX/VRM)
4. Run this script in Blender pointing at the .fbx and the two .npz files.

Note on SMPLToFBX
-----------------
The "SMPL to FBX Retargeting" node exists in the repo but requires a
SMPL_PARAMS input that no node currently outputs — it is not wirable in the
ComfyUI graph yet.  Use the BVH route above instead.

Coordinate conventions
----------------------
GVHMR world frame  : Y-up, gravity-aligned (same as SMPL convention).
GVHMR camera frame : OpenGL-style (-Z forward, Y up) — this is because
                     R_cam2world is derived from SMPL body rotations that live
                     in the same Y-up space.
Blender world      : Z-up.
Blender camera     : -Z forward, Y up  (identical to GVHMR camera frame).

Therefore the only conversion needed is world Y-up → Z-up, applied uniformly
to all positions and rotation matrices.  The camera-local axes need no flip.

If the imported camera appears to face the wrong direction, try toggling the
FLIP_CAMERA_Y boolean in the CONFIGURE section — this adds a 180° X-axis
rotation that swaps between OpenGL and OpenCV camera conventions.
"""

import bpy
import numpy as np
from mathutils import Matrix, Vector, Euler
import math
import os


# ── CONFIGURE ────────────────────────────────────────────────────────────────

# Path to the main NPZ from "GVHMR Inference" (always required)
NPZ_PATH = "/mnt/share/dev-lbringer/ComfyUI_nightly/output/smpl_params_0014.npz"

# Path to the separate camera NPZ from "GVHMR Inference"
# (produced only when moving_camera=True; leave empty "" to load camera data
#  from the main NPZ if it was embedded there)
CAMERA_NPZ_PATH = "/mnt/share/dev-lbringer/ComfyUI_nightly/output/camera_trajectory_0009.npz"

# Path to the SMPC .bin file from "SMPL Viewer with Camera" node.
# When set, uses the Procrustes-REFINED camera (same as what the viewer shows)
# instead of the raw NPZ camera — fixes height and alignment offsets.
# Takes priority over CAMERA_NPZ_PATH when provided.
SMPC_BIN_PATH = "/mnt/share/dev-lbringer/ComfyUI_nightly/output/smpl_camera_mesh_0013.bin"

# Path to the BVH from "SMPL to BVH Converter" node.
# Imports as a native Blender armature — fully editable in Pose Mode / NLA editor.
# Can be used alongside GLB_PATH (the GLB provides the skin mesh, the BVH the rig).
# Set to "" to skip.
BVH_PATH = ""  # disabled — using the GLB's built-in armature instead

# Path to the animated FBX from "BVH to FBX Retargeter" node.
# This is a fully rigged character with baked SMPL motion — editable in Blender.
# Set to "" to skip or use GLB_PATH instead.
FBX_PATH = ""

# Path to the GLB from "SMPL to GLB Animation" node (simpler, no rig).
# Used only when FBX_PATH is empty. Set to "" to skip body import entirely.
GLB_PATH = "/mnt/share/dev-lbringer/ComfyUI_nightly/output/smpl_anim_0009.glb"

# Frames-per-second of the original video fed to GVHMR Inference.
FPS = 24

# Sensor width used to convert pixel focal length → mm (Blender default: 36 mm).
SENSOR_WIDTH_MM = 36.0

# Set to True if the imported camera faces the wrong direction.
# Adds a 180° rotation around X (OpenCV ↔ OpenGL camera convention swap).
FLIP_CAMERA_Y = True

# Path to SMPLX_NEUTRAL.npz — used to auto-compute the Z offset needed to align
# the BVH armature with the GLB mesh.  The GLB root is at (transl + J[0]) while
# the BVH root is at (transl), so the armature needs to be shifted up by J[0].y
# (which becomes a Z offset after Y-up → Z-up conversion).
# Set to "" to disable auto-computation and use BVH_Z_OFFSET manually instead.
SMPLX_MODEL_PATH = "/mnt/share/dev-lbringer/ComfyUI_nightly/models/motion_capture/body_models/smplx/SMPLX_NEUTRAL.npz"

# Manual Z offset applied to the BVH armature (in Blender units = metres).
# Ignored when SMPLX_MODEL_PATH is set and the file is found.
# Set this if auto-computation is unavailable and the rig floats above the mesh.
BVH_Z_OFFSET = 0.0

# ── END CONFIGURE ─────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_bvh_z_offset(npz_path, smplx_model_path):
    """
    Compute the Z offset to apply to the BVH armature so it aligns with the GLB.

    The GLB root node is animated at (transl + J[0]) while the BVH root is at
    (transl).  After Y-up → Z-up conversion both J[0].y and transl.y map to Z,
    so the BVH armature needs to be shifted by +J[0].y to match the GLB.
    """
    data = np.load(smplx_model_path, allow_pickle=True)
    v_template = data['v_template'].astype(np.float64)                 # (V, 3)
    shapedirs  = data['shapedirs'][:, :, :10].astype(np.float64)       # (V, 3, 10)
    J_reg = data['J_regressor']
    if isinstance(J_reg, np.ndarray) and J_reg.ndim == 0:
        J_reg = J_reg.item()
    if hasattr(J_reg, 'toarray'):
        J_reg = J_reg.toarray()
    J_reg = np.asarray(J_reg, dtype=np.float64)

    # Use subject's betas if available, else neutral
    betas = np.zeros(10, dtype=np.float64)
    try:
        subj = np.load(npz_path)
        if 'betas' in subj:
            b = subj['betas']
            if b.ndim == 2:
                b = b[0]
            betas = b[:10].astype(np.float64)
    except Exception:
        pass

    v_shaped = v_template + np.einsum('vck,k->vc', shapedirs, betas)
    J0 = (J_reg @ v_shaped)[0]   # pelvis in Y-up SMPL template space
    # J0[1] is the Y component; after Y→Z conversion this becomes Blender Z
    return float(J0[1])


def _load_camera_from_smpc_bin(bin_path):
    """
    Parse the SMPC .bin file written by 'SMPL Viewer with Camera' and extract
    the Procrustes-refined R_cam2world, t_cam2world, K_fullimg, img dimensions,
    and fps.  Returns (R, t, K, img_w, img_h, fps) or raises on failure.

    Binary layout (matches smpl_camera_viewer_node.py):
      "SMPC"           4 bytes  magic
      num_frames       uint32
      num_vertices     uint32
      num_faces        uint32
      fps              float32
      mesh_color       64 bytes (null-padded UTF-8)
      has_camera       uint32
      img_width        uint32
      img_height       uint32
      vertex_data      F * V * 3 * float32
      face_data        num_faces * 3 * uint32
      [if has_camera]
        R_cam2world    F * 3 * 3 * float32
        t_cam2world    F * 3 * float32
        K_fullimg      F * 3 * 3 * float32
    """
    import struct
    with open(bin_path, "rb") as f:
        magic = f.read(4)
        if magic != b"SMPC":
            raise ValueError(f"Not an SMPC file: {bin_path}")
        num_frames  = struct.unpack("<I", f.read(4))[0]
        num_verts   = struct.unpack("<I", f.read(4))[0]
        num_faces   = struct.unpack("<I", f.read(4))[0]
        fps         = struct.unpack("<f", f.read(4))[0]
        f.read(64)                                           # mesh_color
        has_camera  = struct.unpack("<I", f.read(4))[0]
        img_w       = struct.unpack("<I", f.read(4))[0]
        img_h       = struct.unpack("<I", f.read(4))[0]
        # skip vertex data + face data
        f.read(num_frames * num_verts * 3 * 4)
        f.read(num_faces * 3 * 4)
        if not has_camera:
            return None, None, None, img_w, img_h, fps
        R = np.frombuffer(f.read(num_frames * 9 * 4),  dtype=np.float32).reshape(num_frames, 3, 3).astype(np.float64)
        t = np.frombuffer(f.read(num_frames * 3 * 4),  dtype=np.float32).reshape(num_frames, 3).astype(np.float64)
        K = np.frombuffer(f.read(num_frames * 9 * 4),  dtype=np.float32).reshape(num_frames, 3, 3).astype(np.float64)
    print(f"  Loaded Procrustes-refined camera from SMPC bin ({num_frames} frames, fps={fps})")
    return R, t, K, img_w, img_h, fps

# Rotation matrix: world Y-up (GVHMR) → world Z-up (Blender)
#   Blender X = GVHMR X
#   Blender Y = -GVHMR Z
#   Blender Z =  GVHMR Y
R_Y2Z = np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0],
], dtype=np.float64)

# Optional camera convention flip (180° around X)
R_FLIP = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1],
], dtype=np.float64)


def _load_camera_data(npz_path, camera_npz_path):
    """Load R_cam2world, t_cam2world, K_fullimg, img_width, img_height."""
    R, t, K, img_w, img_h = None, None, None, 1920, 1080

    def _try_extract(src):
        nonlocal img_w, img_h
        if 'R_cam2world' in src and 't_cam2world' in src and 'K_fullimg' in src:
            r = src['R_cam2world'].astype(np.float64)
            _t = src['t_cam2world'].astype(np.float64)
            k = src['K_fullimg'].astype(np.float64)
            if 'img_width' in src:
                img_w = int(src['img_width'].flat[0])
            if 'img_height' in src:
                img_h = int(src['img_height'].flat[0])
            return r, _t, k
        elif 'R_w2c' in src and 't_w2c' in src and 'K_fullimg' in src:
            # Old format: world-to-cam → cam-to-world
            R_w2c = src['R_w2c'].astype(np.float64)
            t_w2c = src['t_w2c'].astype(np.float64)
            r = R_w2c.transpose(0, 2, 1)
            _t = -np.einsum('fij,fj->fi', r, t_w2c)
            k = src['K_fullimg'].astype(np.float64)
            if 'img_width' in src:
                img_w = int(src['img_width'].flat[0])
            if 'img_height' in src:
                img_h = int(src['img_height'].flat[0])
            print("  [warning] Using old R_w2c/t_w2c format — re-run GVHMR for best results.")
            return r, _t, k
        return None, None, None

    # 1. Try separate camera NPZ first
    if camera_npz_path and os.path.isfile(camera_npz_path):
        cam_data = np.load(camera_npz_path)
        R, t, K = _try_extract(cam_data)
        if R is not None:
            print(f"  Loaded camera data from: {camera_npz_path}")

    # 2. Fall back to main NPZ
    if R is None and os.path.isfile(npz_path):
        main_data = np.load(npz_path)
        R, t, K = _try_extract(main_data)
        if R is not None:
            print(f"  Loaded camera data embedded in: {npz_path}")

    if R is None:
        print("  [warning] No camera trajectory found — camera object will not be animated.")

    return R, t, K, img_w, img_h


def _apply_world_conversion(R_c2w, t_c2w):
    """Convert all camera matrices from GVHMR Y-up to Blender Z-up."""
    F = R_c2w.shape[0]
    R_out = np.zeros_like(R_c2w)
    t_out = np.zeros_like(t_c2w)
    cam_flip = R_FLIP if FLIP_CAMERA_Y else np.eye(3)
    for f in range(F):
        # Convert rotation: R_blender = R_y2z @ R_c2w @ cam_flip
        R_out[f] = R_Y2Z @ R_c2w[f] @ cam_flip
        # Convert translation: t_blender = R_y2z @ t_c2w
        t_out[f] = R_Y2Z @ t_c2w[f]
    return R_out, t_out


def _rotation_matrix_to_blender_matrix(R, t):
    """Build a 4×4 Blender Matrix from a 3×3 numpy rotation and 3-vector translation."""
    mat = Matrix([
        [R[0, 0], R[0, 1], R[0, 2], t[0]],
        [R[1, 0], R[1, 1], R[1, 2], t[1]],
        [R[2, 0], R[2, 1], R[2, 2], t[2]],
        [0,       0,       0,       1   ],
    ])
    return mat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n=== GVHMR → Blender Import ===")

    # -- Validate paths -------------------------------------------------------
    global NPZ_PATH
    if NPZ_PATH and not os.path.isfile(NPZ_PATH):
        print(f"  [warning] NPZ not found (will be skipped): {NPZ_PATH}")
        NPZ_PATH = ""

    # -- Load camera data -----------------------------------------------------
    print("[1/4] Loading camera data …")
    bin_fps = None
    if SMPC_BIN_PATH and os.path.isfile(SMPC_BIN_PATH):
        R_c2w, t_c2w, K, img_w, img_h, bin_fps = _load_camera_from_smpc_bin(SMPC_BIN_PATH)
    elif SMPC_BIN_PATH:
        print(f"  [warning] SMPC bin not found, falling back to NPZ: {SMPC_BIN_PATH}")
        R_c2w, t_c2w, K, img_w, img_h = _load_camera_data(NPZ_PATH, CAMERA_NPZ_PATH)
    else:
        R_c2w, t_c2w, K, img_w, img_h = _load_camera_data(NPZ_PATH, CAMERA_NPZ_PATH)

    num_frames = R_c2w.shape[0] if R_c2w is not None else None

    # -- Configure scene ------------------------------------------------------
    print("[2/4] Configuring scene …")
    scene = bpy.context.scene
    effective_fps = bin_fps if (bin_fps and bin_fps > 0) else FPS
    scene.render.fps = int(round(effective_fps))
    scene.render.fps_base = scene.render.fps / effective_fps  # handles non-integer FPS
    if num_frames is not None:
        scene.frame_start = 1
        scene.frame_end = num_frames
    scene.render.resolution_x = img_w
    scene.render.resolution_y = img_h
    print(f"  Scene: {num_frames} frames @ {effective_fps} fps, {img_w}×{img_h}")

    # -- Import body (FBX or GLB) ---------------------------------------------
    print("[3/4] Importing body …")
    if FBX_PATH and os.path.isfile(FBX_PATH):
        bpy.ops.import_scene.fbx(filepath=FBX_PATH, automatic_bone_orientation=True)
        print(f"  Imported FBX: {FBX_PATH}")
    elif FBX_PATH:
        print(f"  [warning] FBX not found, skipping: {FBX_PATH}")
    elif GLB_PATH and os.path.isfile(GLB_PATH):
        before_import = set(bpy.data.objects.keys())
        bpy.ops.import_scene.gltf(filepath=GLB_PATH)
        print(f"  Imported GLB: {GLB_PATH}")

        # Clear custom bone shapes (icospheres) assigned by the GLTF importer
        # and switch to STICK display — no bone positions are touched.
        new_objs = [bpy.data.objects[n] for n in bpy.data.objects.keys() if n not in before_import]
        glb_armatures = [o for o in new_objs if o.type == 'ARMATURE']
        for arm in glb_armatures:
            arm.data.display_type = 'OCTAHEDRAL'
            for pb in arm.pose.bones:
                pb.custom_shape = None
            print(f"  Set STICK display on armature: {arm.name}")
    elif GLB_PATH:
        print(f"  [warning] GLB not found, skipping: {GLB_PATH}")
    else:
        print("  No body path set — skipping body import.")

    # -- Import BVH armature --------------------------------------------------
    if BVH_PATH and os.path.isfile(BVH_PATH):
        # Track objects before import so we can find the new armature
        before = set(bpy.data.objects.keys())
        bpy.ops.import_anim.bvh(
            filepath=BVH_PATH,
            use_fps_scale=False,   # keep rotations as-is, don't rescale to scene FPS
            update_scene_fps=False,
            update_scene_duration=False,
        )
        new_objs = [bpy.data.objects[n] for n in bpy.data.objects.keys() if n not in before]
        bvh_armatures = [o for o in new_objs if o.type == 'ARMATURE']

        # Compute and apply Z offset to align BVH root with GLB root
        z_offset = BVH_Z_OFFSET
        if SMPLX_MODEL_PATH and os.path.isfile(SMPLX_MODEL_PATH):
            try:
                z_offset = _compute_bvh_z_offset(NPZ_PATH, SMPLX_MODEL_PATH)
                print(f"  BVH Z offset (auto from J[0]): {z_offset:.4f} m")
            except Exception as e:
                print(f"  [warning] Could not auto-compute BVH Z offset: {e} — using BVH_Z_OFFSET={BVH_Z_OFFSET}")
        elif BVH_Z_OFFSET != 0.0:
            print(f"  BVH Z offset (manual): {z_offset:.4f} m")

        for arm in bvh_armatures:
            arm.location.z += z_offset

        print(f"  Imported BVH armature: {BVH_PATH}")
    elif BVH_PATH:
        print(f"  [warning] BVH not found, skipping: {BVH_PATH}")

    # -- Create / reuse camera ------------------------------------------------
    print("[4/4] Building animated camera …")

    base_name = "GVHMR_Camera"
    idx = 1
    CAM_NAME = base_name
    while CAM_NAME in bpy.data.objects:
        CAM_NAME = f"{base_name}.{idx:03d}"
        idx += 1

    cam_data = bpy.data.cameras.new(name=CAM_NAME)
    cam_data.sensor_width = SENSOR_WIDTH_MM
    cam_obj = bpy.data.objects.new(CAM_NAME, cam_data)
    scene.collection.objects.link(cam_obj)

    # Make it the active scene camera
    scene.camera = cam_obj

    if R_c2w is not None:
        R_bl, t_bl = _apply_world_conversion(R_c2w, t_c2w)

        for f in range(num_frames):
            frame_idx = f + 1  # Blender frames are 1-based

            # Camera world matrix
            mat = _rotation_matrix_to_blender_matrix(R_bl[f], t_bl[f])
            cam_obj.matrix_world = mat

            # Keyframe location and rotation
            cam_obj.keyframe_insert(data_path="location", frame=frame_idx)
            cam_obj.keyframe_insert(data_path="rotation_euler", frame=frame_idx)

            # Focal length from intrinsics (K[f][0,0] = fx in pixels)
            if K is not None:
                fx = float(K[f][0, 0])
                focal_mm = fx * SENSOR_WIDTH_MM / img_w
                cam_data.lens = focal_mm
                cam_data.keyframe_insert(data_path="lens", frame=frame_idx)

        # Set interpolation to LINEAR for all camera fcurves (avoids overshoots)
        if cam_obj.animation_data and cam_obj.animation_data.action:
            for fc in cam_obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'
        if cam_data.animation_data and cam_data.animation_data.action:
            for fc in cam_data.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'

        print(f"  Camera '{CAM_NAME}' animated over {num_frames} frames.")
        print(f"  Camera position frame 1: {tuple(round(v, 3) for v in t_bl[0])}")
        if K is not None:
            fx0 = float(K[0][0, 0])
            print(f"  Focal length frame 1: {fx0 * SENSOR_WIDTH_MM / img_w:.1f} mm  (fx={fx0:.1f} px)")
    else:
        print(f"  Camera '{CAM_NAME}' created (no animation — no camera data found).")

    # Jump to frame 1
    scene.frame_set(1)

    print("\nDone! Tips:")
    print("  • Press Numpad-0 to look through the GVHMR_Camera.")
    print("  • Press Space to play back the animation.")
    if FLIP_CAMERA_Y is False:
        print("  • If the camera faces the wrong direction, set FLIP_CAMERA_Y = True and re-run.")


main()
