"""
SMPLToGLB Node - Export SMPL motion capture as animated GLB (skinned mesh + skeletal animation).
"""

import json
import struct
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import folder_paths

from .shared_utils import next_sequential_filename
from .smpl_to_bvh_node import SMPL_21_JOINT_NAMES, SMPL_21_PARENTS

logger = logging.getLogger("SMPLToGLB")

# Number of body joints (excluding root pelvis for body_pose, including it for skeleton)
NUM_BODY_JOINTS = 21
NUM_JOINTS = 22  # 1 root + 21 body


def _compute_vertex_normals(vertices, faces):
    """Compute per-vertex normals by averaging incident face normals."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    # Accumulate face normals to vertices
    normals = np.zeros_like(vertices)
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)
    # Normalize
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-8)
    return (normals / lengths).astype(np.float32)


def _collapse_smplx_weights_to_smpl22(weights_55, parents_55):
    """
    Collapse SMPLX 55-joint weights to 22 SMPL body joints.
    Joints 22-54 (hands, face) are mapped to their nearest body ancestor.
    """
    # Build collapse map: for each joint >= 22, find ancestor <= 21
    collapse_map = {}
    for j in range(22, 55):
        p = j
        while p >= 22:
            p = parents_55[p]
        collapse_map[j] = p

    weights_22 = weights_55[:, :22].copy()
    for j in range(22, 55):
        weights_22[:, collapse_map[j]] += weights_55[:, j]

    return weights_22


def _top_k_weights(weights, k=4):
    """Pick top-k influences per vertex, zero out rest, renormalize."""
    n_verts = weights.shape[0]
    joint_indices = np.zeros((n_verts, k), dtype=np.uint8)
    joint_weights = np.zeros((n_verts, k), dtype=np.float32)

    for i in range(n_verts):
        w = weights[i]
        top_idx = np.argsort(w)[-k:][::-1]
        top_w = w[top_idx]
        total = top_w.sum()
        if total > 0:
            top_w /= total
        joint_indices[i] = top_idx.astype(np.uint8)
        joint_weights[i] = top_w.astype(np.float32)

    return joint_indices, joint_weights


def _axis_angle_to_quat(aa):
    """
    Convert axis-angle (F, J, 3) to quaternions (F, J, 4) in (x, y, z, w) format.
    """
    F, J, _ = aa.shape
    quats = np.zeros((F, J, 4), dtype=np.float32)
    for f in range(F):
        rot = R.from_rotvec(aa[f])  # (J,) Rotation objects
        q = rot.as_quat()  # (J, 4) in (x, y, z, w) -- matches glTF
        quats[f] = q
    return quats


def _pad_to_4(data):
    """Pad bytes to 4-byte alignment."""
    remainder = len(data) % 4
    if remainder:
        data += b'\x00' * (4 - remainder)
    return data


def _build_glb(gltf_json, bin_data):
    """
    Build a GLB binary from glTF JSON and binary buffer data.
    GLB format: 12-byte header + JSON chunk + BIN chunk.
    """
    json_str = json.dumps(gltf_json, separators=(',', ':'))
    json_bytes = json_str.encode('utf-8')
    # Pad JSON to 4-byte alignment with spaces (glTF spec)
    json_pad = (4 - len(json_bytes) % 4) % 4
    json_bytes += b' ' * json_pad
    json_chunk_length = len(json_bytes)

    # Pad BIN to 4-byte alignment with zeros
    bin_pad = (4 - len(bin_data) % 4) % 4
    bin_data_padded = bin_data + b'\x00' * bin_pad
    bin_chunk_length = len(bin_data_padded)

    total_length = 12 + 8 + json_chunk_length + 8 + bin_chunk_length

    glb = bytearray()
    # Header
    glb += struct.pack('<4sII', b'glTF', 2, total_length)
    # JSON chunk
    glb += struct.pack('<II', json_chunk_length, 0x4E4F534A)
    glb += json_bytes
    # BIN chunk
    glb += struct.pack('<II', bin_chunk_length, 0x004E4942)
    glb += bin_data_padded

    return bytes(glb)


class SMPLToGLB:
    """
    Export SMPL motion capture data as an animated GLB file with skinned mesh
    and skeletal animation. The output can be imported into Blender, Three.js, etc.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "npz_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to .npz file with SMPL parameters (from GVHMR Inference)"
                }),
            },
            "optional": {
                "fps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "tooltip": "Animation frames per second"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "export_glb"
    CATEGORY = "MotionCapture/GVHMR"
    OUTPUT_NODE = True

    def export_glb(self, npz_path="", fps=30):
        if not npz_path or not npz_path.strip():
            raise ValueError("npz_path is required")
        npz_file = Path(npz_path)
        if not npz_file.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_file}")

        logger.info(f"[SMPLToGLB] Loading SMPL parameters from: {npz_path}")
        data = np.load(str(npz_file))
        body_pose = data['body_pose']        # (F, 63)
        global_orient = data['global_orient']  # (F, 3)
        betas = data['betas']                # (F, 10)
        transl = data.get('transl', None)    # (F, 3) or None
        if transl is None:
            transl = np.zeros((body_pose.shape[0], 3), dtype=np.float32)

        num_frames = body_pose.shape[0]
        logger.info(f"[SMPLToGLB] {num_frames} frames, fps={fps}")

        # ---- Load SMPLX model data ----
        data_dir = Path(__file__).parent / "body_model"
        models_dir = Path(folder_paths.models_dir) / "motion_capture" / "body_models" / "smplx"
        smplx_path = models_dir / "SMPLX_NEUTRAL.npz"
        if not smplx_path.exists():
            raise FileNotFoundError(f"SMPLX model not found: {smplx_path}")

        smplx_data = np.load(str(smplx_path), allow_pickle=True)
        v_template = smplx_data['v_template'].astype(np.float64)     # (10475, 3)
        shapedirs = smplx_data['shapedirs'][:, :, :10].astype(np.float64)  # (10475, 3, 10)
        J_regressor = smplx_data['J_regressor']  # sparse or dense (55, 10475)
        lbs_weights = smplx_data['weights'].astype(np.float64)       # (10475, 55)
        parents_55 = smplx_data['kintree_table'][0].astype(np.int32)  # (55,)
        parents_55[0] = -1

        # Handle J_regressor (may be sparse object wrapped in 0-d array)
        if isinstance(J_regressor, np.ndarray) and J_regressor.ndim == 0:
            J_regressor = J_regressor.item()
        if hasattr(J_regressor, 'toarray'):
            J_regressor = J_regressor.toarray()
        J_regressor = np.asarray(J_regressor, dtype=np.float64)  # (55, 10475)

        # Load SMPLX->SMPL vertex mapping
        import scipy.sparse as sp
        smplx2smpl = sp.load_npz(str(data_dir / "smplx2smpl_sparse.npz")).toarray().astype(np.float64)  # (6890, 10475)

        faces = np.load(str(data_dir / "smpl_faces.npy")).astype(np.int32)  # (13776, 3)

        # ---- Rest-pose mesh (use first frame betas) ----
        beta0 = betas[0].astype(np.float64)  # (10,)
        v_shaped_smplx = v_template + np.einsum('vck,k->vc', shapedirs, beta0)  # (10475, 3)
        v_shaped_smpl = smplx2smpl @ v_shaped_smplx  # (6890, 3)
        positions = v_shaped_smpl.astype(np.float32)
        normals = _compute_vertex_normals(positions, faces)

        # ---- Skeleton ----
        J_all = J_regressor @ v_shaped_smplx  # (55, 3)
        J = J_all[:NUM_JOINTS].astype(np.float64)  # (22, 3) -- body joints only

        # ---- Skinning weights ----
        smpl_weights_55 = smplx2smpl @ lbs_weights  # (6890, 55)
        weights_22 = _collapse_smplx_weights_to_smpl22(smpl_weights_55, parents_55)  # (6890, 22)
        joint_indices, joint_weights = _top_k_weights(weights_22, k=4)

        # ---- Inverse bind matrices ----
        ibms = np.zeros((NUM_JOINTS, 4, 4), dtype=np.float32)
        for j in range(NUM_JOINTS):
            ibms[j] = np.eye(4, dtype=np.float32)
            ibms[j, :3, 3] = -J[j].astype(np.float32)

        # ---- Animation data ----
        # Combine global_orient + body_pose -> (F, 22, 3) axis-angle
        full_pose = np.concatenate([
            global_orient.reshape(-1, 1, 3),
            body_pose.reshape(-1, NUM_BODY_JOINTS, 3)
        ], axis=1).astype(np.float64)  # (F, 22, 3)

        quats = _axis_angle_to_quat(full_pose)  # (F, 22, 4) in (x,y,z,w)
        timestamps = (np.arange(num_frames, dtype=np.float32) / fps)
        # Root translation = skeleton root position + transl offset
        # In SMPL, transl is added on top of FK output: verts = LBS(...) + transl
        # In glTF, the animated translation REPLACES the node's rest translation (J[0])
        # So we need: animated_transl = J[0] + transl
        translations = (transl + J[0]).astype(np.float32)  # (F, 3)

        # ---- Build GLB binary buffer ----
        buf = bytearray()
        buffer_views = []
        accessors = []

        def add_buffer_view(data_bytes, target=None):
            """Append data to buffer, return bufferView index."""
            # Align to 4 bytes
            pad = (4 - len(buf) % 4) % 4
            buf.extend(b'\x00' * pad)
            offset = len(buf)
            buf.extend(data_bytes)
            bv = {"buffer": 0, "byteOffset": offset, "byteLength": len(data_bytes)}
            if target is not None:
                bv["target"] = target
            idx = len(buffer_views)
            buffer_views.append(bv)
            return idx

        def add_accessor(bv_idx, comp_type, count, dtype_str, min_val=None, max_val=None):
            """Add accessor, return accessor index."""
            acc = {
                "bufferView": bv_idx,
                "componentType": comp_type,
                "count": count,
                "type": dtype_str,
            }
            if min_val is not None:
                acc["min"] = min_val
            if max_val is not None:
                acc["max"] = max_val
            idx = len(accessors)
            accessors.append(acc)
            return idx

        # glTF component types
        CT_BYTE = 5120
        CT_UBYTE = 5121
        CT_USHORT = 5123
        CT_UINT = 5125
        CT_FLOAT = 5126

        # Target types
        TGT_ARRAY = 34962
        TGT_ELEMENT = 34963

        # 1. Positions
        pos_data = positions.tobytes()
        pos_bv = add_buffer_view(pos_data, TGT_ARRAY)
        pos_min = positions.min(axis=0).tolist()
        pos_max = positions.max(axis=0).tolist()
        pos_acc = add_accessor(pos_bv, CT_FLOAT, len(positions), "VEC3", pos_min, pos_max)

        # 2. Normals
        norm_data = normals.tobytes()
        norm_bv = add_buffer_view(norm_data, TGT_ARRAY)
        norm_acc = add_accessor(norm_bv, CT_FLOAT, len(normals), "VEC3")

        # 3. Indices (uint16 -- max index 6889 < 65535)
        indices_u16 = faces.astype(np.uint16).flatten()
        idx_data = indices_u16.tobytes()
        idx_bv = add_buffer_view(idx_data, TGT_ELEMENT)
        idx_acc = add_accessor(idx_bv, CT_USHORT, len(indices_u16), "SCALAR",
                               [int(indices_u16.min())], [int(indices_u16.max())])

        # 4. JOINTS_0 (uint8)
        joints_data = joint_indices.tobytes()
        joints_bv = add_buffer_view(joints_data, TGT_ARRAY)
        joints_acc = add_accessor(joints_bv, CT_UBYTE, len(joint_indices), "VEC4")

        # 5. WEIGHTS_0 (float32)
        weights_data = joint_weights.tobytes()
        weights_bv = add_buffer_view(weights_data, TGT_ARRAY)
        weights_acc = add_accessor(weights_bv, CT_FLOAT, len(joint_weights), "VEC4")

        # 6. Inverse bind matrices (column-major for glTF)
        ibms_col = np.transpose(ibms, (0, 2, 1)).astype(np.float32)  # to column-major
        ibm_data = ibms_col.tobytes()
        ibm_bv = add_buffer_view(ibm_data)
        ibm_acc = add_accessor(ibm_bv, CT_FLOAT, NUM_JOINTS, "MAT4")

        # 7. Animation timestamps
        ts_data = timestamps.tobytes()
        ts_bv = add_buffer_view(ts_data)
        ts_acc = add_accessor(ts_bv, CT_FLOAT, num_frames, "SCALAR",
                              [float(timestamps[0])], [float(timestamps[-1])])

        # 8. Root translations
        tl_data = translations.tobytes()
        tl_bv = add_buffer_view(tl_data)
        tl_acc = add_accessor(tl_bv, CT_FLOAT, num_frames, "VEC3")

        # 9. Per-joint rotation quaternions
        rot_accs = []
        for j in range(NUM_JOINTS):
            q = quats[:, j, :].astype(np.float32).copy()  # (F, 4)
            q_data = q.tobytes()
            q_bv = add_buffer_view(q_data)
            q_acc = add_accessor(q_bv, CT_FLOAT, num_frames, "VEC4")
            rot_accs.append(q_acc)

        # ---- Build glTF JSON ----
        # Nodes: 0 = armature root, 1..22 = joint nodes, 23 = mesh node
        # Joint nodes are arranged so node index = joint_index + 1
        # (node 0 is the armature container)

        nodes = []

        # Node 0: Armature root (contains skeleton + mesh)
        armature_children = [NUM_JOINTS + 1]  # mesh node
        # Find root joints (pelvis = joint 0 -> node 1)
        armature_children.append(1)
        nodes.append({
            "name": "Armature",
            "children": armature_children,
        })

        # Nodes 1..22: Joint nodes
        for j in range(NUM_JOINTS):
            node = {"name": SMPL_21_JOINT_NAMES[j]}
            # Translation = offset from parent (or absolute for root)
            if SMPL_21_PARENTS[j] == -1:
                node["translation"] = J[j].tolist()
            else:
                parent_j = SMPL_21_PARENTS[j]
                offset = (J[j] - J[parent_j]).tolist()
                node["translation"] = offset

            # Find children of this joint
            children = []
            for c in range(NUM_JOINTS):
                if SMPL_21_PARENTS[c] == j:
                    children.append(c + 1)  # node index = joint index + 1
            if children:
                node["children"] = children

            nodes.append(node)

        # Node 23: Mesh node
        nodes.append({
            "name": "SMPLMesh",
            "mesh": 0,
            "skin": 0,
        })

        # Skin
        joint_node_indices = list(range(1, NUM_JOINTS + 1))
        skin = {
            "inverseBindMatrices": ibm_acc,
            "joints": joint_node_indices,
            "skeleton": 1,  # root joint node
            "name": "SMPLSkin",
        }

        # Mesh
        mesh = {
            "primitives": [{
                "attributes": {
                    "POSITION": pos_acc,
                    "NORMAL": norm_acc,
                    "JOINTS_0": joints_acc,
                    "WEIGHTS_0": weights_acc,
                },
                "indices": idx_acc,
                "mode": 4,  # TRIANGLES
            }],
            "name": "SMPLMesh",
        }

        # Animation
        samplers = []
        channels = []

        # Root translation sampler + channel
        transl_sampler_idx = len(samplers)
        samplers.append({
            "input": ts_acc,
            "output": tl_acc,
            "interpolation": "LINEAR",
        })
        channels.append({
            "sampler": transl_sampler_idx,
            "target": {
                "node": 1,  # root joint node
                "path": "translation",
            },
        })

        # Per-joint rotation samplers + channels
        for j in range(NUM_JOINTS):
            sampler_idx = len(samplers)
            samplers.append({
                "input": ts_acc,
                "output": rot_accs[j],
                "interpolation": "LINEAR",
            })
            channels.append({
                "sampler": sampler_idx,
                "target": {
                    "node": j + 1,  # joint node index
                    "path": "rotation",
                },
            })

        animation = {
            "name": "SMPLAnimation",
            "samplers": samplers,
            "channels": channels,
        }

        # Assemble glTF
        gltf = {
            "asset": {"version": "2.0", "generator": "ComfyUI-MotionCapture"},
            "scene": 0,
            "scenes": [{"nodes": [0], "name": "Scene"}],
            "nodes": nodes,
            "meshes": [mesh],
            "skins": [skin],
            "animations": [animation],
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": [{"byteLength": len(buf)}],
        }

        # ---- Write GLB ----
        glb_bytes = _build_glb(gltf, bytes(buf))

        output_dir = Path(folder_paths.get_output_directory())
        glb_filename = next_sequential_filename(output_dir, "smpl_anim", ".glb")
        glb_path = output_dir / glb_filename
        with open(glb_path, "wb") as f:
            f.write(glb_bytes)

        size_mb = glb_path.stat().st_size / (1024 * 1024)
        logger.info(f"[SMPLToGLB] Wrote {glb_filename} ({size_mb:.1f} MB) -- "
                     f"{num_frames} frames, {NUM_JOINTS} joints, "
                     f"{len(positions)} vertices, {len(faces)} faces")

        return {"ui": {}, "result": (str(glb_path),)}


NODE_CLASS_MAPPINGS = {
    "SMPLToGLB": SMPLToGLB,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SMPLToGLB": "SMPL to GLB Animation",
}
