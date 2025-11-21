import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from .viz2d import plot_matches, plot_images, save_plot

# ================================
# Visualization
# ================================


def visualize_matches(img0, img1, kp0, kp1, output_dir):
    """Visualize the matched features between two images."""
    plot_images([img0, img1], ["Image 0", "Image 1"])
    plot_matches(kp0, kp1)
    save_plot(Path(output_dir) / "matches.png")


def visualize_T_w2c_rotations(T_w2c_list, output_dir):
    """可视化相机旋转轨迹，并考虑 OpenCV 坐标系转换，
    使得相机的 x 轴保持右向，z 轴（光轴）变为水平前向，
    而相机的 y 轴（朝下）对应于 plt 的 -z 轴（即向下）。"""

    # 定义转换矩阵：将 OpenCV 坐标 (x:right, y:down, z:forward)
    # 转换为 world 坐标 (x:right, y:forward, z:up)
    R_align = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

    # 原始光轴：在 OpenCV 中通常为 [0, 0, 1]
    normal_vector = np.array([0, 0, 1])
    aligned_rotated_normals = []

    # 对每一帧的旋转矩阵，先计算光轴旋转后的方向，再进行坐标转换
    for T in T_w2c_list:
        R = T[:3, :3]
        rotated_normal = R.T @ normal_vector
        # 应用对齐变换
        aligned_normal = R_align @ rotated_normal
        aligned_rotated_normals.append(aligned_normal)
    aligned_rotated_normals = np.array(aligned_rotated_normals)

    # ------------------ 3D 可视化 ------------------
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制调整后的旋转法向量轨迹
    ax.plot(aligned_rotated_normals[:, 0], aligned_rotated_normals[:, 1], aligned_rotated_normals[:, 2], "b-")
    ax.scatter(aligned_rotated_normals[:, 0], aligned_rotated_normals[:, 1], aligned_rotated_normals[:, 2], c="r", s=10)

    # 标记起点与终点
    ax.scatter(
        aligned_rotated_normals[0, 0],
        aligned_rotated_normals[0, 1],
        aligned_rotated_normals[0, 2],
        c="g",
        s=100,
        marker="o",
        label="Start",
    )
    ax.scatter(
        aligned_rotated_normals[-1, 0],
        aligned_rotated_normals[-1, 1],
        aligned_rotated_normals[-1, 2],
        c="m",
        s=100,
        marker="o",
        label="End",
    )

    # 绘制单位球以便参考：对球面上每个点也应用相同的转换
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    sphere_points = np.stack([x, y, z], axis=-1)
    sphere_points_aligned = sphere_points @ R_align.T
    X_aligned = sphere_points_aligned[:, :, 0]
    Y_aligned = sphere_points_aligned[:, :, 1]
    Z_aligned = sphere_points_aligned[:, :, 2]
    ax.plot_wireframe(X_aligned, Y_aligned, Z_aligned, color="gray", alpha=0.2)

    # 设置坐标轴比例和范围
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])

    ax.set_xlabel("X")
    ax.set_ylabel("Y (Forward)")
    ax.set_zlabel("Z (Up)")
    ax.set_title("Camera Rotation T_w2c_list (3D) - Aligned to Camera Conventions")
    ax.legend()

    # 添加帧数标记
    frame_count = len(T_w2c_list)
    interval = max(1, frame_count // 10)
    for i in range(0, frame_count, interval):
        ax.text(
            aligned_rotated_normals[i, 0],
            aligned_rotated_normals[i, 1],
            aligned_rotated_normals[i, 2],
            f"{i}",
            fontsize=8,
        )

    plt.savefig(Path(output_dir) / "rotation_trajectory_aligned.png")
    plt.show()


def visualize_rotation_angles(T_w2c_list, output_dir):
    """Visualize rotation as Euler angles over time."""
    # Extract rotation matrices
    rotations = [T[:3, :3] for T in T_w2c_list]

    # Convert to Euler angles (in degrees)
    euler_angles = []
    for R in rotations:
        # Convert rotation matrix to Euler angles
        # Using the 'xyz' convention - roll, pitch, yaw
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0

        # Convert to degrees
        euler_angles.append([np.degrees(roll), np.degrees(pitch), np.degrees(yaw)])

    euler_angles = np.array(euler_angles)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    frame_indices = np.arange(len(T_w2c_list))

    ax.plot(frame_indices, euler_angles[:, 0], "r-", label="Roll")
    ax.plot(frame_indices, euler_angles[:, 1], "g-", label="Pitch")
    ax.plot(frame_indices, euler_angles[:, 2], "b-", label="Yaw")

    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("Camera Rotation: Euler Angles Over Time")
    ax.legend()
    ax.grid(True)

    plt.savefig(Path(output_dir) / "rotation_angles.png")
    plt.show()


# ================================
# Video Reader
# ================================


def read_video_frame_np(video_path, frame_index):
    # Use opencv to read frame at frame_index
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    # Convert to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


# ================================
# Camera
# ================================


def focal_length_from_mm(width, height, mm=24):
    """
    Convert full-frame focal length to image sensor focal length.
    """
    diag_fullframe = (24**2 + 36**2) ** 0.5
    diag_img = (width**2 + height**2) ** 0.5
    focal_length = diag_img / diag_fullframe * mm
    return focal_length
