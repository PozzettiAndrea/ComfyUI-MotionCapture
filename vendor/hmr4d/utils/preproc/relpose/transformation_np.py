import numpy as np


def rotation_matrix_to_quaternion(R):
    """
    将 3x3 旋转矩阵 R 转换为四元数 [w, x, y, z] 的形式。
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
    将四元数 [w, x, y, z] 转换为 3x3 旋转矩阵。
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
    对两个四元数 q0 和 q1 进行球面线性插值（SLERP）。

    参数：
      q0, q1: numpy 数组，形状为 (4,)，表示四元数 [w, x, y, z]
      t: 插值系数，0 <= t <= 1

    返回：
      插值后的四元数，形状为 (4,)
    """
    dot = np.dot(q0, q1)
    # 如果点积为负，取相反数以保证取短路径
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # 当两个四元数非常接近时，直接使用线性插值再归一化
        result = q0 + t * (q1 - q0)
        result = result / np.linalg.norm(result)
        return result

    theta_0 = np.arccos(dot)  # 两个四元数之间的角度
    theta = theta_0 * t  # 插值后的角度
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (s0 * q0) + (s1 * q1)


def lerp_missing_frames(T_w2c_list, sample_idxs):
    """
    对给定的 T_w2c_list（已知帧的变换矩阵）进行平滑插值，生成所有帧的变换矩阵。
    其中：
      - 平移部分采用线性插值；
      - 旋转部分采用自实现的SLERP球面线性插值，保证旋转过渡平滑。

    参数：
        T_w2c_list (numpy.ndarray): 形状为 (F, 4, 4) 的已知变换矩阵数组
        sample_idxs (list 或 numpy.ndarray): 长度为 F 的已知帧在原始序列中的索引
          （假设第一个索引为 0，最后一个为 F_all - 1）

    返回：
        numpy.ndarray: 形状为 (F_all, 4, 4) 的所有帧的变换矩阵，缺失帧通过平滑插值填充。
    """
    sample_idxs = np.array(sample_idxs)
    # 根据最后一个已知帧索引确定总帧数（假设索引从 0 开始）
    F_all = sample_idxs[-1] + 1
    new_T_list = []

    # 分离出平移和旋转部分
    translations = np.array([T[:3, 3] for T in T_w2c_list])
    rotations = np.array([T[:3, :3] for T in T_w2c_list])
    # 将旋转矩阵转换为四元数
    quaternions = np.array([rotation_matrix_to_quaternion(R) for R in rotations])

    for i in range(F_all):
        # 如果该帧为已知帧，直接使用对应的变换矩阵
        if i in sample_idxs:
            known_index = np.where(sample_idxs == i)[0][0]
            new_T_list.append(T_w2c_list[known_index])
        else:
            # 定位左右两侧已知帧
            next_known = np.searchsorted(sample_idxs, i)
            prev_known = next_known - 1
            # 计算插值比例 t
            t_interp = (i - sample_idxs[prev_known]) / (sample_idxs[next_known] - sample_idxs[prev_known])
            # 平移部分：线性插值
            trans_interp = (1 - t_interp) * translations[prev_known] + t_interp * translations[next_known]
            # 旋转部分：自实现 SLERP 插值
            q0 = quaternions[prev_known]
            q1 = quaternions[next_known]
            q_interp = slerp(q0, q1, t_interp)
            rot_interp = quaternion_to_rotation_matrix(q_interp)
            # 构造最终的 4x4 变换矩阵
            T_interp = np.eye(4)
            T_interp[:3, :3] = rot_interp
            T_interp[:3, 3] = trans_interp
            new_T_list.append(T_interp)

    return np.array(new_T_list)
