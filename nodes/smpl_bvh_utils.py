"""Shared SMPL skeleton constants and BVH conversion utilities."""

import logging
import numpy as np

log = logging.getLogger("motioncapture")

# SMPL skeleton configuration for BVH conversion (22 joints)
SMPL_BONE_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"
]

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

SMPL_OFFSETS = [
    [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0],
    [0, 1, 0], [0, -1, 0], [0, -1, 0], [0, 1, 0], [0, -0.5, 0.5], [0, -0.5, 0.5],
    [0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0], [-1, 0, 0],
    [1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]
]


def axis_angle_to_euler_zxy(axis_angle):
    """Convert axis-angle to ZXY Euler angles (BVH standard)."""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return [0.0, 0.0, 0.0]
    axis = axis_angle / angle
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    x, y, z = axis

    # Rotation matrix
    R = np.array([
        [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
        [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
        [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]
    ])

    # Extract ZXY Euler
    if abs(R[2, 1]) < 0.99999:
        x_rot = np.arcsin(-R[2, 1])
        y_rot = np.arctan2(R[2, 0], R[2, 2])
        z_rot = np.arctan2(R[0, 1], R[1, 1])
    else:
        x_rot = np.pi / 2 if R[2, 1] < 0 else -np.pi / 2
        y_rot = np.arctan2(-R[0, 2], R[0, 0])
        z_rot = 0

    return [np.degrees(z_rot), np.degrees(x_rot), np.degrees(y_rot)]


def smpl_to_bvh(smpl_params, output_path, fps=30):
    """Convert SMPL parameters to BVH file."""
    body_pose = smpl_params.get('body_pose')
    global_orient = smpl_params.get('global_orient')
    transl = smpl_params.get('transl')

    if body_pose is None:
        raise ValueError("No body_pose in SMPL params")

    num_frames = body_pose.shape[0]
    body_pose = body_pose.reshape(num_frames, 21, 3)

    # Build BVH header
    lines = ["HIERARCHY", "ROOT Pelvis", "{", "\tOFFSET 0.0 0.0 0.0",
             "\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"]

    def add_joint(idx, depth):
        indent = "\t" * depth
        children = [i for i, p in enumerate(SMPL_PARENTS) if p == idx]

        if children:
            for child_idx in children:
                child_name = SMPL_BONE_NAMES[child_idx]
                child_offset = SMPL_OFFSETS[child_idx]
                lines.append(f"{indent}JOINT {child_name}")
                lines.append(f"{indent}{{")
                lines.append(f"{indent}\tOFFSET {child_offset[0]*10:.4f} {child_offset[1]*10:.4f} {child_offset[2]*10:.4f}")
                lines.append(f"{indent}\tCHANNELS 3 Zrotation Xrotation Yrotation")
                add_joint(child_idx, depth + 1)
                lines.append(f"{indent}}}")
        else:
            offset = SMPL_OFFSETS[idx]
            lines.append(f"{indent}End Site")
            lines.append(f"{indent}{{")
            lines.append(f"{indent}\tOFFSET {offset[0]*5:.4f} {offset[1]*5:.4f} {offset[2]*5:.4f}")
            lines.append(f"{indent}}}")

    add_joint(0, 1)
    lines.append("}")

    # Motion section
    lines.append("MOTION")
    lines.append(f"Frames: {num_frames}")
    lines.append(f"Frame Time: {1.0/fps:.6f}")

    for frame in range(num_frames):
        values = []

        # Root position (convert SMPL Y-up to BVH Z-up)
        if transl is not None:
            t = transl[frame]
            values.extend([t[0]*100, t[2]*100, t[1]*100])  # Scale and swap Y/Z
        else:
            values.extend([0, 0, 0])

        # Root rotation
        if global_orient is not None:
            euler = axis_angle_to_euler_zxy(global_orient[frame])
            values.extend(euler)
        else:
            values.extend([0, 0, 0])

        # Body pose rotations
        for joint_idx in range(21):
            euler = axis_angle_to_euler_zxy(body_pose[frame, joint_idx])
            values.extend(euler)

        lines.append(" ".join(f"{v:.4f}" for v in values))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    log.info("Created BVH: %s (%d frames)", output_path, num_frames)
    return output_path
