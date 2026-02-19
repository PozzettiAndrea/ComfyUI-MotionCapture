import numpy as np


def rotation_matrix_to_quaternion(R):
    """
    Convert 3x3 rotation matrix R to quaternion [w, x, y, z].
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S = 4 * qw
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2  # S = 4 * qx
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2  # S = 4 * qy
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2  # S = 4 * qz
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.
    """
    qw, qx, qy, qz = q
    R = np.array(
        [
            [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2],
        ]
    )
    return R


def slerp(q0, q1, t):
    """
    Spherical linear interpolation (SLERP) between two quaternions.

    Args:
      q0, q1: numpy arrays, shape (4,), representing quaternions [w, x, y, z]
      t: interpolation coefficient, 0 <= t <= 1

    Returns:
      Interpolated quaternion, shape (4,)
    """
    dot = np.dot(q0, q1)
    # If dot product is negative, negate to ensure shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # When quaternions are very close, use linear interpolation and normalize
        result = q0 + t * (q1 - q0)
        result = result / np.linalg.norm(result)
        return result

    theta_0 = np.arccos(dot)  # Angle between the two quaternions
    theta = theta_0 * t  # Interpolated angle
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (s0 * q0) + (s1 * q1)


def lerp_missing_frames(T_w2c_list, sample_idxs):
    """
    Smoothly interpolate transformation matrices for all frames from known frames.
    - Translation: linear interpolation
    - Rotation: SLERP (spherical linear interpolation) for smooth rotation transitions

    Args:
        T_w2c_list (numpy.ndarray): Known transformation matrices, shape (F, 4, 4)
        sample_idxs (list or numpy.ndarray): Indices of known frames in the original sequence
          (assumes first index is 0, last is F_all - 1)

    Returns:
        numpy.ndarray: Transformation matrices for all frames, shape (F_all, 4, 4)
    """
    sample_idxs = np.array(sample_idxs)
    # Determine total frame count from last known frame index (assumes 0-indexed)
    F_all = sample_idxs[-1] + 1
    new_T_list = []

    # Separate translation and rotation components
    translations = np.array([T[:3, 3] for T in T_w2c_list])
    rotations = np.array([T[:3, :3] for T in T_w2c_list])
    # Convert rotation matrices to quaternions
    quaternions = np.array([rotation_matrix_to_quaternion(R) for R in rotations])

    for i in range(F_all):
        # If this is a known frame, use its transformation matrix directly
        if i in sample_idxs:
            known_index = np.where(sample_idxs == i)[0][0]
            new_T_list.append(T_w2c_list[known_index])
        else:
            # Find neighboring known frames
            next_known = np.searchsorted(sample_idxs, i)
            prev_known = next_known - 1
            # Calculate interpolation ratio t
            t_interp = (i - sample_idxs[prev_known]) / (sample_idxs[next_known] - sample_idxs[prev_known])
            # Translation: linear interpolation
            trans_interp = (1 - t_interp) * translations[prev_known] + t_interp * translations[next_known]
            # Rotation: SLERP interpolation
            q0 = quaternions[prev_known]
            q1 = quaternions[next_known]
            q_interp = slerp(q0, q1, t_interp)
            rot_interp = quaternion_to_rotation_matrix(q_interp)
            # Construct final 4x4 transformation matrix
            T_interp = np.eye(4)
            T_interp[:3, :3] = rot_interp
            T_interp[:3, 3] = trans_interp
            new_T_list.append(T_interp)

    return np.array(new_T_list)
