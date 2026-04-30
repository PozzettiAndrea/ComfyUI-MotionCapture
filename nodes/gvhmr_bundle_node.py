"""
GVHMRBundle Node - collect ComfyUI-MotionCapture outputs into one folder.
"""

import json
import shutil
import struct
from datetime import datetime
from pathlib import Path
from typing import Dict

import folder_paths
import numpy as np
import scipy.sparse as sp
from scipy.spatial.transform import Rotation as R

from .glb_export_node import NUM_BODY_JOINTS, SMPLX_55_JOINT_NAMES, _top_k_weights
from .shared_utils import next_sequential_filename


def _clean_optional_path(value):
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"none", "null"} else value


def _resolve_path(path_value):
    path_value = _clean_optional_path(path_value)
    if not path_value:
        return None

    path = Path(path_value).expanduser()
    if path.exists():
        return path.resolve()

    output_candidate = Path(folder_paths.get_output_directory()) / path_value
    if output_candidate.exists():
        return output_candidate.resolve()

    input_candidate = Path(folder_paths.get_input_directory()) / path_value
    if input_candidate.exists():
        return input_candidate.resolve()

    return None


def _copy_file(src_path, bundle_dir, used_names):
    suffix = src_path.suffix
    stem = src_path.stem
    dst_name = src_path.name
    index = 1
    while dst_name in used_names:
        dst_name = f"{stem}_{index:03d}{suffix}"
        index += 1

    dst_path = bundle_dir / dst_name
    shutil.copy2(src_path, dst_path)
    used_names.add(dst_name)
    return dst_path


def _default_smplx_model_path():
    models_dir = Path(folder_paths.models_dir) / "motion_capture" / "body_models" / "smplx"
    neutral = models_dir / "SMPLX_NEUTRAL.npz"
    return neutral.resolve() if neutral.exists() else None


def _extract_camera_from_npz(src):
    if "R_cam2world" in src and "t_cam2world" in src and "K_fullimg" in src:
        return (
            src["R_cam2world"].astype(np.float64),
            src["t_cam2world"].astype(np.float64),
            src["K_fullimg"].astype(np.float64),
        )
    if "R_w2c" in src and "t_w2c" in src and "K_fullimg" in src:
        r_w2c = src["R_w2c"].astype(np.float64)
        t_w2c = src["t_w2c"].astype(np.float64)
        r_c2w = r_w2c.transpose(0, 2, 1)
        t_c2w = -np.einsum("fij,fj->fi", r_c2w, t_w2c)
        return r_c2w, t_c2w, src["K_fullimg"].astype(np.float64)
    return None, None, None


def _load_camera_from_smpc_bin(bin_path):
    with open(bin_path, "rb") as f:
        magic = f.read(4)
        if magic != b"SMPC":
            raise ValueError(f"Not an SMPC file: {bin_path}")
        num_frames = struct.unpack("<I", f.read(4))[0]
        num_verts = struct.unpack("<I", f.read(4))[0]
        num_faces = struct.unpack("<I", f.read(4))[0]
        fps = struct.unpack("<f", f.read(4))[0]
        f.read(64)
        has_camera = struct.unpack("<I", f.read(4))[0]
        img_w = struct.unpack("<I", f.read(4))[0]
        img_h = struct.unpack("<I", f.read(4))[0]
        f.read(num_frames * num_verts * 3 * 4)
        f.read(num_faces * 3 * 4)
        if not has_camera:
            return None, None, None, img_w, img_h, fps
        r = np.frombuffer(f.read(num_frames * 9 * 4), dtype=np.float32).reshape(num_frames, 3, 3).astype(np.float64)
        t = np.frombuffer(f.read(num_frames * 3 * 4), dtype=np.float32).reshape(num_frames, 3).astype(np.float64)
        k = np.frombuffer(f.read(num_frames * 9 * 4), dtype=np.float32).reshape(num_frames, 3, 3).astype(np.float64)
    return r, t, k, img_w, img_h, fps


def _load_camera_for_maya(source_paths, fallback_fps):
    img_w, img_h = 1920, 1080
    camera_fps = fallback_fps

    if source_paths.get("smpc_bin") is not None:
        r, t, k, img_w, img_h, smpc_fps = _load_camera_from_smpc_bin(source_paths["smpc_bin"])
        if r is not None:
            return r, t, k, img_w, img_h, smpc_fps or camera_fps, "smpc_bin"

    for key in ("camera_npz", "npz"):
        src = source_paths.get(key)
        if src is None:
            continue
        data = np.load(str(src))
        r, t, k = _extract_camera_from_npz(data)
        if r is None:
            continue
        if "img_width" in data:
            img_w = int(data["img_width"].flat[0])
        if "img_height" in data:
            img_h = int(data["img_height"].flat[0])
        return r, t, k, img_w, img_h, camera_fps, key

    return None, None, None, img_w, img_h, camera_fps, ""


def _write_maya_camera_json(bundle_dir, source_paths, fps):
    r_c2w, t_c2w, k_fullimg, img_w, img_h, camera_fps, source = _load_camera_for_maya(source_paths, fps)
    if r_c2w is None:
        return None

    frames = []
    sensor_width_mm = 36.0
    for idx in range(r_c2w.shape[0]):
        r = r_c2w[idx]
        t = t_c2w[idx]
        matrix = [
            float(r[0, 0]), float(r[0, 1]), float(r[0, 2]), 0.0,
            float(r[1, 0]), float(r[1, 1]), float(r[1, 2]), 0.0,
            float(r[2, 0]), float(r[2, 1]), float(r[2, 2]), 0.0,
            float(t[0]), float(t[1]), float(t[2]), 1.0,
        ]
        focal_length_mm = None
        if k_fullimg is not None and img_w > 0:
            focal_length_mm = float(k_fullimg[idx][0, 0]) * sensor_width_mm / img_w
        frames.append({
            "frame": idx + 1,
            "matrix": matrix,
            "focal_length_mm": focal_length_mm,
        })

    camera_json = {
        "schema": "comfyui-motioncapture.maya-camera",
        "schema_version": 1,
        "source": source,
        "fps": float(camera_fps),
        "img_width": int(img_w),
        "img_height": int(img_h),
        "sensor_width_mm": sensor_width_mm,
        "camera_name": "GVHMR_Camera",
        "coordinate_system": "maya_y_up_camera_minus_z_forward",
        "frames": frames,
    }

    camera_path = bundle_dir / "camera_maya.json"
    with open(camera_path, "w", encoding="utf-8") as f:
        json.dump(camera_json, f, indent=2)
        f.write("\n")
    return camera_path


def _write_maya_smplx_rig_json(bundle_dir, npz_path, fps, gender="neutral", hand_pose="halfway"):
    data = np.load(str(npz_path))
    body_pose = data["body_pose"]
    global_orient = data["global_orient"]
    betas = data["betas"]
    transl = data.get("transl", None)
    if transl is None:
        transl = np.zeros((body_pose.shape[0], 3), dtype=np.float32)

    num_frames = int(body_pose.shape[0])
    data_dir = Path(__file__).parent / "body_model"
    models_dir = Path(folder_paths.models_dir) / "motion_capture" / "body_models" / "smplx"
    smplx_path = models_dir / f"SMPLX_{gender.upper()}.npz"
    if not smplx_path.exists():
        raise FileNotFoundError(f"SMPLX model not found: {smplx_path}")

    smplx_data = np.load(str(smplx_path), allow_pickle=True)
    v_template = smplx_data["v_template"].astype(np.float64)
    shapedirs = smplx_data["shapedirs"][:, :, :10].astype(np.float64)
    j_regressor = smplx_data["J_regressor"]
    lbs_weights = smplx_data["weights"].astype(np.float64)
    parents_55 = smplx_data["kintree_table"][0].astype(np.int32)
    parents_55[0] = -1

    if isinstance(j_regressor, np.ndarray) and j_regressor.ndim == 0:
        j_regressor = j_regressor.item()
    if hasattr(j_regressor, "toarray"):
        j_regressor = j_regressor.toarray()
    j_regressor = np.asarray(j_regressor, dtype=np.float64)

    smplx2smpl = sp.load_npz(str(data_dir / "smplx2smpl_sparse.npz")).toarray().astype(np.float64)
    faces = np.load(str(data_dir / "smpl_faces.npy")).astype(np.int32)

    beta0 = betas[0].astype(np.float64)
    v_shaped_smplx = v_template + np.einsum("vck,k->vc", shapedirs, beta0)
    mesh_positions = (smplx2smpl @ v_shaped_smplx).astype(np.float32)

    joints = (j_regressor @ v_shaped_smplx).astype(np.float32)
    smpl_weights_55 = smplx2smpl @ lbs_weights
    skin_indices, skin_weights = _top_k_weights(smpl_weights_55, k=4)

    body_pose_22 = np.concatenate([
        global_orient.reshape(-1, 1, 3),
        body_pose.reshape(-1, NUM_BODY_JOINTS, 3),
    ], axis=1).astype(np.float64)

    hands_meanl = np.asarray(smplx_data.get("hands_meanl", np.zeros(45)), dtype=np.float64)
    hands_meanr = np.asarray(smplx_data.get("hands_meanr", np.zeros(45)), dtype=np.float64)
    hand_aa = np.zeros((33, 3), dtype=np.float64)
    if hand_pose == "halfway":
        hand_scale = 1.0
    elif hand_pose == "closed":
        hand_scale = 2.0
    else:
        hand_scale = 0.0
    if hand_scale != 0.0:
        hand_aa[3:18] = (hands_meanl * hand_scale).reshape(15, 3)
        hand_aa[18:33] = (hands_meanr * hand_scale).reshape(15, 3)

    hand_pose_rest = np.tile(hand_aa, (num_frames, 1, 1))
    full_pose = np.concatenate([body_pose_22, hand_pose_rest], axis=1)
    eulers_xyz = R.from_rotvec(full_pose.reshape(-1, 3)).as_euler("xyz", degrees=True).reshape(num_frames, 55, 3)
    translations = (transl + joints[0]).astype(np.float32)

    rig_json = {
        "schema": "comfyui-motioncapture.maya-smplx-rig",
        "schema_version": 1,
        "source": str(npz_path),
        "fps": float(fps),
        "gender": gender,
        "hand_pose": hand_pose,
        "joint_names": SMPLX_55_JOINT_NAMES,
        "joint_parents": parents_55.astype(int).tolist(),
        "joint_positions": joints.astype(float).tolist(),
        "mesh_positions": mesh_positions.astype(float).tolist(),
        "faces": faces.astype(int).tolist(),
        "skin_indices": skin_indices.astype(int).tolist(),
        "skin_weights": skin_weights.astype(float).tolist(),
        "animation": {
            "root_translations": translations.astype(float).tolist(),
            "joint_eulers_xyz_deg": eulers_xyz.astype(np.float32).astype(float).tolist(),
        },
    }

    rig_path = bundle_dir / "smplx_rig_maya.json"
    with open(rig_path, "w", encoding="utf-8") as f:
        json.dump(rig_json, f, separators=(",", ":"))
        f.write("\n")
    return rig_path


def _make_bundle_dir(bundle_name):
    output_dir = Path(folder_paths.get_output_directory())
    safe_prefix = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in bundle_name.strip())
    if not safe_prefix:
        safe_prefix = "gvhmr_bundle"

    bundle_folder_name = next_sequential_filename(output_dir, safe_prefix, "")
    bundle_dir = output_dir / bundle_folder_name
    bundle_dir.mkdir(parents=True, exist_ok=False)
    return bundle_dir


def _add_files_to_bundle(source_paths, bundle_dir, copy_files):
    files: Dict[str, str] = {}
    absolute_sources: Dict[str, str] = {}
    used_names = set()

    for key, src in source_paths.items():
        if src is None:
            continue
        absolute_sources[key] = str(src)
        if copy_files:
            dst = _copy_file(src, bundle_dir, used_names)
            files[key] = dst.name
        else:
            files[key] = str(src)

    return files, absolute_sources


def _write_manifest(bundle_dir, manifest):
    manifest_path = bundle_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest_path


class GVHMRBlenderBundle:
    """
    Create a Blender-focused bundle for the GVHMR Blender add-on.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "npz_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "SMPL params .npz from GVHMR Inference or Save SMPL Motion",
                }),
                "bundle_name": ("STRING", {
                    "default": "gvhmr_blender",
                    "multiline": False,
                    "tooltip": "Folder name prefix inside the ComfyUI output folder",
                }),
            },
            "optional": {
                "camera_npz_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Camera trajectory .npz from GVHMR Inference",
                }),
                "smpc_bin_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "SMPC .bin from SMPL Viewer with Camera. This is preferred for Blender camera import.",
                }),
                "glb_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Animated GLB from SMPL to GLB Animation",
                }),
                "fbx_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Animated FBX from a retarget/export node",
                }),
                "bvh_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional BVH motion file",
                }),
                "fps": ("INT", {
                    "default": 24,
                    "min": 1,
                    "max": 240,
                    "step": 1,
                }),
                "copy_files": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Copy all referenced files into the bundle folder. Disable to write references only.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("bundle_dir", "manifest_path", "info")
    FUNCTION = "create_blender_bundle"
    OUTPUT_NODE = True
    CATEGORY = "MotionCapture/GVHMR"

    def create_blender_bundle(
        self,
        npz_path: str,
        bundle_name: str = "gvhmr_blender",
        camera_npz_path: str = "",
        smpc_bin_path: str = "",
        glb_path: str = "",
        fbx_path: str = "",
        bvh_path: str = "",
        fps: int = 24,
        copy_files: bool = True,
    ):
        bundle_dir = _make_bundle_dir(bundle_name)
        source_paths = {
            "npz": _resolve_path(npz_path),
            "camera_npz": _resolve_path(camera_npz_path),
            "smpc_bin": _resolve_path(smpc_bin_path),
            "glb": _resolve_path(glb_path),
            "fbx": _resolve_path(fbx_path),
            "bvh": _resolve_path(bvh_path),
        }

        if source_paths["npz"] is None:
            raise FileNotFoundError(f"SMPL NPZ file not found: {npz_path}")

        files, absolute_sources = _add_files_to_bundle(source_paths, bundle_dir, copy_files)

        smplx_model = _default_smplx_model_path()
        if smplx_model is not None:
            files["smplx_model"] = str(smplx_model)

        manifest = {
            "schema": "comfyui-motioncapture.gvhmr-blender-bundle",
            "schema_version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "generator": "ComfyUI-MotionCapture GVHMRBlenderBundle",
            "files": files,
            "absolute_sources": absolute_sources,
            "scene": {
                "fps": int(fps),
                "sensor_width_mm": 36.0,
                "flip_camera_y": True,
                "camera_name": "GVHMR_Camera",
            },
            "notes": {
                "camera_priority": "smpc_bin, then camera_npz, then npz embedded camera",
                "body_priority": "fbx, then glb, with optional bvh",
            },
        }

        manifest_path = _write_manifest(bundle_dir, manifest)

        copied_count = len([key for key in files if key != "smplx_model"])
        info = (
            "GVHMR Blender Bundle Complete\n"
            f"Bundle: {bundle_dir}\n"
            f"Manifest: {manifest_path}\n"
            f"Files {'copied' if copy_files else 'referenced'}: {copied_count}\n"
            "Blender: select this bundle folder in the GVHMR add-on.\n"
        )

        return (str(bundle_dir), str(manifest_path), info)


class GVHMRMayaBundle:
    """
    Create a Maya-focused bundle with an FBX/Alembic body and camera_maya.json.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bundle_name": ("STRING", {
                    "default": "gvhmr_maya",
                    "multiline": False,
                    "tooltip": "Folder name prefix inside the ComfyUI output folder",
                }),
            },
            "optional": {
                "camera_npz_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Camera trajectory .npz from GVHMR Inference.",
                }),
                "smpc_bin_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "SMPC .bin from SMPL Viewer with Camera.",
                }),
                "npz_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "SMPL params .npz. Used to generate a Maya SMPLX-55 skeletal rig, and as camera fallback if it embeds camera data.",
                }),
                "gender": (["neutral", "male", "female"], {
                    "default": "neutral",
                    "tooltip": "SMPLX body model gender used for the optional Maya skeletal rig.",
                }),
                "hand_pose": (["halfway", "open", "closed"], {
                    "default": "halfway",
                    "tooltip": "Default finger pose for the optional Maya SMPLX-55 rig.",
                }),
                "fps": ("INT", {
                    "default": 24,
                    "min": 1,
                    "max": 240,
                    "step": 1,
                }),
                "copy_files": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Copy all referenced files into the bundle folder. Disable to write references only.",
                }),
                "camera_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Allow a Maya bundle with only the animated camera and no FBX/Alembic body.",
                }),
                "allow_smpc_mesh": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Allow SMPC .bin as a Maya diagnostic animated mesh cache when no FBX/Alembic body is provided.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("bundle_dir", "manifest_path", "info")
    FUNCTION = "create_maya_bundle"
    OUTPUT_NODE = True
    CATEGORY = "MotionCapture/GVHMR"

    def create_maya_bundle(
        self,
        bundle_name: str = "gvhmr_maya",
        fbx_path: str = "",
        alembic_path: str = "",
        camera_npz_path: str = "",
        smpc_bin_path: str = "",
        npz_path: str = "",
        gender: str = "neutral",
        hand_pose: str = "halfway",
        fps: int = 24,
        copy_files: bool = True,
        camera_only: bool = False,
        allow_smpc_mesh: bool = True,
    ):
        bundle_dir = _make_bundle_dir(bundle_name)
        source_paths = {
            "camera_npz": _resolve_path(camera_npz_path),
            "smpc_bin": _resolve_path(smpc_bin_path),
            "npz": _resolve_path(npz_path),
        }

        has_smpc_mesh = source_paths["smpc_bin"] is not None and allow_smpc_mesh
        has_smplx_rig = source_paths["npz"] is not None
        if not has_smpc_mesh and not has_smplx_rig and not camera_only:
            raise FileNotFoundError(
                "Maya bundle needs an SMPL NPZ for the SMPLX rig, or an SMPC mesh cache for diagnostics. "
                "Enable camera_only to export only the camera."
            )

        maya_camera_path = _write_maya_camera_json(bundle_dir, source_paths, fps)
        if maya_camera_path is None:
            raise FileNotFoundError("No camera data found. Provide smpc_bin_path, camera_npz_path, or NPZ with embedded camera data.")

        files, absolute_sources = _add_files_to_bundle(source_paths, bundle_dir, copy_files)
        files["maya_camera_json"] = maya_camera_path.name
        maya_smplx_rig_path = None
        if has_smplx_rig:
            maya_smplx_rig_path = _write_maya_smplx_rig_json(
                bundle_dir,
                source_paths["npz"],
                fps,
                gender=gender,
                hand_pose=hand_pose,
            )
            files["maya_smplx_rig_json"] = maya_smplx_rig_path.name

        manifest = {
            "schema": "comfyui-motioncapture.gvhmr-maya-bundle",
            "schema_version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "generator": "ComfyUI-MotionCapture GVHMRMayaBundle",
            "files": files,
            "absolute_sources": absolute_sources,
            "scene": {
                "fps": int(fps),
                "camera_name": "GVHMR_Camera",
                "smplx_rig_name": "GVHMR_SMPLX",
            },
            "notes": {
                "body_priority": "maya_smplx_rig_json, then diagnostic smpc mesh cache. GLB, FBX, and Alembic are intentionally not included for this Maya path.",
                "camera": "Maya importer reads maya_camera_json directly and does not need NumPy.",
                "maya_smplx_rig_json": "Experimental SMPLX-55 skeletal rig payload generated from the SMPL NPZ.",
            },
        }

        manifest_path = _write_manifest(bundle_dir, manifest)

        body_kind = (
            "SMPLX skeletal rig"
            if has_smplx_rig
            else "SMPC mesh cache"
            if has_smpc_mesh
            else "camera only"
        )
        info = (
            "GVHMR Maya Bundle Complete\n"
            f"Bundle: {bundle_dir}\n"
            f"Manifest: {manifest_path}\n"
            f"Body: {body_kind}\n"
            f"SMPLX Rig: {'yes' if maya_smplx_rig_path is not None else 'no'}\n"
            "Maya: select this bundle folder in the GVHMR importer.\n"
        )

        return (str(bundle_dir), str(manifest_path), info)


NODE_CLASS_MAPPINGS = {
    "GVHMRBlenderBundle": GVHMRBlenderBundle,
    "GVHMRMayaBundle": GVHMRMayaBundle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GVHMRBlenderBundle": "GVHMR Bundle for Blender",
    "GVHMRMayaBundle": "GVHMR Bundle for Maya",
}
