"""
SMPLtoBVH Node - Convert SMPL motion data to BVH format
"""

from pathlib import Path
from typing import Dict, Tuple
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from hmr4d.utils.pylogger import Log


# SMPL skeleton hierarchy - 21-joint variant (GVHMR, no hands) (22 total with root)
SMPL_21_JOINT_NAMES = [
    'Pelvis',       # 0 (root)
    'L_Hip',        # 1
    'R_Hip',        # 2
    'Spine1',       # 3
    'L_Knee',       # 4
    'R_Knee',       # 5
    'Spine2',       # 6
    'L_Ankle',      # 7
    'R_Ankle',      # 8
    'Spine3',       # 9
    'L_Foot',       # 10
    'R_Foot',       # 11
    'Neck',         # 12
    'L_Collar',     # 13
    'R_Collar',     # 14
    'Head',         # 15
    'L_Shoulder',   # 16
    'R_Shoulder',   # 17
    'L_Elbow',      # 18
    'R_Elbow',      # 19
    'L_Wrist',      # 20
    'R_Wrist',      # 21
]

# SMPL skeleton hierarchy - 23-joint variant (full SMPL) (24 total with root)
SMPL_23_JOINT_NAMES = [
    'Pelvis',       # 0 (root)
    'L_Hip',        # 1
    'R_Hip',        # 2
    'Spine1',       # 3
    'L_Knee',       # 4
    'R_Knee',       # 5
    'Spine2',       # 6
    'L_Ankle',      # 7
    'R_Ankle',      # 8
    'Spine3',       # 9
    'L_Foot',       # 10
    'R_Foot',       # 11
    'Neck',         # 12
    'L_Collar',     # 13
    'R_Collar',     # 14
    'Head',         # 15
    'L_Shoulder',   # 16
    'R_Shoulder',   # 17
    'L_Elbow',      # 18
    'R_Elbow',      # 19
    'L_Wrist',      # 20
    'R_Wrist',      # 21
    'L_Hand',       # 22
    'R_Hand',       # 23
]

# Parent indices for 21-joint variant (no hands)
SMPL_21_PARENTS = [
    -1,  # 0: Pelvis (root)
    0,   # 1: L_Hip
    0,   # 2: R_Hip
    0,   # 3: Spine1
    1,   # 4: L_Knee
    2,   # 5: R_Knee
    3,   # 6: Spine2
    4,   # 7: L_Ankle
    5,   # 8: R_Ankle
    6,   # 9: Spine3
    7,   # 10: L_Foot
    8,   # 11: R_Foot
    9,   # 12: Neck
    9,   # 13: L_Collar
    9,   # 14: R_Collar
    12,  # 15: Head
    13,  # 16: L_Shoulder
    14,  # 17: R_Shoulder
    16,  # 18: L_Elbow
    17,  # 19: R_Elbow
    18,  # 20: L_Wrist
    19,  # 21: R_Wrist
]

# Parent indices for 23-joint variant (with hands)
SMPL_23_PARENTS = [
    -1,  # 0: Pelvis (root)
    0,   # 1: L_Hip
    0,   # 2: R_Hip
    0,   # 3: Spine1
    1,   # 4: L_Knee
    2,   # 5: R_Knee
    3,   # 6: Spine2
    4,   # 7: L_Ankle
    5,   # 8: R_Ankle
    6,   # 9: Spine3
    7,   # 10: L_Foot
    8,   # 11: R_Foot
    9,   # 12: Neck
    9,   # 13: L_Collar
    9,   # 14: R_Collar
    12,  # 15: Head
    13,  # 16: L_Shoulder
    14,  # 17: R_Shoulder
    16,  # 18: L_Elbow
    17,  # 19: R_Elbow
    18,  # 20: L_Wrist
    19,  # 21: R_Wrist
    20,  # 22: L_Hand
    21,  # 23: R_Hand
]

# Joint offsets in meters (approximate SMPL T-pose)
SMPL_OFFSETS = {
    0: [0.0, 0.0, 0.0],          # Pelvis (root, no offset)
    1: [-0.1, -0.04, 0.0],       # L_Hip
    2: [0.1, -0.04, 0.0],        # R_Hip
    3: [0.0, 0.1, 0.0],          # Spine1
    4: [0.0, -0.4, 0.0],         # L_Knee
    5: [0.0, -0.4, 0.0],         # R_Knee
    6: [0.0, 0.2, 0.0],          # Spine2
    7: [0.0, -0.4, 0.0],         # L_Ankle
    8: [0.0, -0.4, 0.0],         # R_Ankle
    9: [0.0, 0.2, 0.0],          # Spine3
    10: [0.0, -0.05, 0.1],       # L_Foot
    11: [0.0, -0.05, 0.1],       # R_Foot
    12: [0.0, 0.1, 0.0],         # Neck
    13: [-0.15, 0.0, 0.0],       # L_Collar
    14: [0.15, 0.0, 0.0],        # R_Collar
    15: [0.0, 0.15, 0.0],        # Head
    16: [-0.1, 0.0, 0.0],        # L_Shoulder
    17: [0.1, 0.0, 0.0],         # R_Shoulder
    18: [0.0, -0.25, 0.0],       # L_Elbow
    19: [0.0, -0.25, 0.0],       # R_Elbow
    20: [0.0, -0.25, 0.0],       # L_Wrist
    21: [0.0, -0.25, 0.0],       # R_Wrist
    22: [0.0, -0.05, 0.0],       # L_Hand
    23: [0.0, -0.05, 0.0],       # R_Hand
}


class SMPLtoBVH:
    """
    Convert SMPL motion parameters to BVH (Biovision Hierarchy) format.
    BVH is a standard format for skeletal animation used across many 3D tools.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "smpl_params": ("SMPL_PARAMS",),
                "output_path": ("STRING", {
                    "default": "output/motion.bvh",
                    "multiline": False,
                }),
                "fps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                }),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 100.0,
                    "step": 0.01,
                    "round": 0.01,
                }),
            },
        }

    RETURN_TYPES = ("BVH_DATA", "STRING", "STRING")
    RETURN_NAMES = ("bvh_data", "file_path", "info")
    FUNCTION = "convert_to_bvh"
    OUTPUT_NODE = True
    CATEGORY = "MotionCapture/BVH"

    def convert_to_bvh(
        self,
        smpl_params: Dict,
        output_path: str,
        fps: int = 30,
        scale: float = 1.0,
    ) -> Tuple[Dict, str, str]:
        """
        Convert SMPL parameters to BVH file format.

        Args:
            smpl_params: SMPL parameters from GVHMRInference or LoadSMPL
            output_path: Path to save BVH file
            fps: Frames per second for the animation
            scale: Scale factor for the skeleton (1.0 = meters, 100.0 = centimeters)

        Returns:
            Tuple of (bvh_data_dict, file_path, info_string)
        """
        try:
            Log.info("[SMPLtoBVH] Converting SMPL to BVH format...")

            # Prepare output directory
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Ensure .bvh extension
            if not output_path.suffix == '.bvh':
                output_path = output_path.with_suffix('.bvh')

            # Extract global parameters
            global_params = smpl_params.get("global", {})

            # Get motion data
            body_pose = global_params.get("body_pose")  # [F, 69] or [F, 23, 3]
            global_orient = global_params.get("global_orient")  # [F, 3]
            transl = global_params.get("transl")  # [F, 3]

            if body_pose is None or global_orient is None:
                raise ValueError("Missing required SMPL parameters: body_pose or global_orient")

            # Convert to numpy
            if isinstance(body_pose, torch.Tensor):
                body_pose = body_pose.cpu().numpy()
            if isinstance(global_orient, torch.Tensor):
                global_orient = global_orient.cpu().numpy()
            if transl is not None and isinstance(transl, torch.Tensor):
                transl = transl.cpu().numpy()
            else:
                transl = np.zeros((body_pose.shape[0], 3))

            # Auto-detect number of joints and reshape body_pose
            if len(body_pose.shape) == 2:
                # body_pose is [F, num_params] format
                num_params = body_pose.shape[1]
                num_body_joints = num_params // 3  # e.g., 63/3=21 or 69/3=23

                Log.info(f"[SMPLtoBVH] Detected {num_body_joints}-joint SMPL variant ({num_params} parameters)")

                # Validate parameter count
                if num_params % 3 != 0:
                    raise ValueError(f"Invalid body_pose size: {num_params} is not divisible by 3")

                body_pose = body_pose.reshape(-1, num_body_joints, 3)
            else:
                # Already in [F, J, 3] format
                num_body_joints = body_pose.shape[1]

            # Select appropriate skeleton configuration
            if num_body_joints == 21:
                joint_names = SMPL_21_JOINT_NAMES
                parent_indices = SMPL_21_PARENTS
                Log.info("[SMPLtoBVH] Using 21-joint skeleton (GVHMR variant, no hands)")
            elif num_body_joints == 23:
                joint_names = SMPL_23_JOINT_NAMES
                parent_indices = SMPL_23_PARENTS
                Log.info("[SMPLtoBVH] Using 23-joint skeleton (full SMPL with hands)")
            else:
                raise ValueError(f"Unsupported number of body joints: {num_body_joints}. Expected 21 or 23.")

            # Combine global_orient and body_pose: [F, num_total_joints, 3]
            if len(global_orient.shape) == 2:
                global_orient = global_orient[:, np.newaxis, :]  # [F, 1, 3]
            full_pose = np.concatenate([global_orient, body_pose], axis=1)  # [F, num_body_joints+1, 3]

            num_frames = full_pose.shape[0]
            num_total_joints = full_pose.shape[1]
            frame_time = 1.0 / fps

            # Convert axis-angle rotations to Euler angles (ZXY order, BVH standard)
            euler_rotations = self._axis_angle_to_euler(full_pose)  # [F, num_total_joints, 3]

            # Write BVH file
            bvh_content = self._write_bvh(
                euler_rotations,
                transl,
                frame_time,
                scale,
                joint_names,
                parent_indices
            )

            # Save to file
            with open(output_path, 'w') as f:
                f.write(bvh_content)

            # Prepare BVH data structure for next nodes
            bvh_data = {
                "file_path": str(output_path.absolute()),
                "num_frames": num_frames,
                "fps": fps,
                "frame_time": frame_time,
                "scale": scale,
                "rotations": euler_rotations,
                "translations": transl,
                "joint_names": joint_names,
                "num_joints": num_total_joints,
            }

            info = (
                f"SMPLtoBVH Complete\n"
                f"Output: {output_path}\n"
                f"Frames: {num_frames}\n"
                f"FPS: {fps}\n"
                f"Frame time: {frame_time:.4f}s\n"
                f"Scale: {scale}x\n"
                f"Joints: {num_total_joints} ({num_body_joints} body + 1 root)\n"
                f"Variant: {'GVHMR (no hands)' if num_body_joints == 21 else 'Full SMPL'}\n"
            )

            Log.info(f"[SMPLtoBVH] Converted {num_frames} frames to {output_path}")
            return (bvh_data, str(output_path.absolute()), info)

        except Exception as e:
            error_msg = f"SMPLtoBVH failed: {str(e)}"
            Log.error(error_msg)
            import traceback
            traceback.print_exc()
            return ({}, "", error_msg)

    def _axis_angle_to_euler(self, axis_angle: np.ndarray) -> np.ndarray:
        """
        Convert axis-angle rotations to Euler angles (ZXY order for BVH).

        Args:
            axis_angle: [F, J, 3] axis-angle rotations

        Returns:
            [F, J, 3] Euler angles in degrees (ZXY order)
        """
        num_frames, num_joints, _ = axis_angle.shape
        euler = np.zeros((num_frames, num_joints, 3))

        for frame in range(num_frames):
            for joint in range(num_joints):
                aa = axis_angle[frame, joint]

                # Skip if rotation is zero
                if np.allclose(aa, 0):
                    continue

                # Convert axis-angle to rotation matrix using scipy
                rot = R.from_rotvec(aa)

                # Convert to Euler angles (ZXY order, intrinsic)
                # BVH uses ZXY Euler angles in degrees
                euler_angles = rot.as_euler('zxy', degrees=True)
                euler[frame, joint] = euler_angles

        return euler

    def _write_bvh(
        self,
        rotations: np.ndarray,
        translations: np.ndarray,
        frame_time: float,
        scale: float,
        joint_names: list,
        parent_indices: list
    ) -> str:
        """
        Write BVH file content.

        Args:
            rotations: [F, num_joints, 3] Euler angles in degrees
            translations: [F, 3] root translations
            frame_time: time per frame in seconds
            scale: scale factor for skeleton
            joint_names: list of joint names
            parent_indices: list of parent indices for each joint

        Returns:
            BVH file content as string
        """
        num_frames = rotations.shape[0]
        num_joints = rotations.shape[1]

        # Store skeleton config for _write_joint
        self.joint_names = joint_names
        self.parent_indices = parent_indices

        # Build BVH hierarchy
        lines = ["HIERARCHY"]
        self._write_joint(lines, 0, 0, scale)

        # Write motion data
        lines.append("MOTION")
        lines.append(f"Frames: {num_frames}")
        lines.append(f"Frame Time: {frame_time:.6f}")

        # Write frame data
        for frame in range(num_frames):
            frame_data = []

            # Root translation (Pelvis)
            tx, ty, tz = translations[frame] * scale
            frame_data.extend([f"{tz:.6f}", f"{tx:.6f}", f"{ty:.6f}"])  # BVH uses Z-up

            # Root rotation (Pelvis)
            rz, rx, ry = rotations[frame, 0]
            frame_data.extend([f"{rz:.6f}", f"{rx:.6f}", f"{ry:.6f}"])

            # All other joint rotations (1 to num_joints-1)
            for joint in range(1, num_joints):
                rz, rx, ry = rotations[frame, joint]
                frame_data.extend([f"{rz:.6f}", f"{rx:.6f}", f"{ry:.6f}"])

            lines.append(" ".join(frame_data))

        return "\n".join(lines)

    def _write_joint(self, lines: list, joint_idx: int, indent_level: int, scale: float):
        """
        Recursively write joint hierarchy in BVH format.

        Args:
            lines: list to append BVH lines to
            joint_idx: current joint index
            indent_level: indentation level
            scale: scale factor for offsets
        """
        indent = "  " * indent_level
        joint_name = self.joint_names[joint_idx]
        offset = SMPL_OFFSETS.get(joint_idx, [0.0, 0.0, 0.0])  # Default to zero if joint not in offsets
        offset_scaled = [o * scale for o in offset]

        # Root joint
        if joint_idx == 0:
            lines.append(f"{indent}ROOT {joint_name}")
        else:
            lines.append(f"{indent}JOINT {joint_name}")

        lines.append(f"{indent}{{")
        lines.append(f"{indent}  OFFSET {offset_scaled[2]:.6f} {offset_scaled[0]:.6f} {offset_scaled[1]:.6f}")  # Z-up

        # Channels
        if joint_idx == 0:
            # Root has translation + rotation
            lines.append(f"{indent}  CHANNELS 6 Zposition Xposition Yposition Zrotation Xrotation Yrotation")
        else:
            # Other joints only have rotation
            lines.append(f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation")

        # Find and write children
        children = [i for i, parent in enumerate(self.parent_indices) if parent == joint_idx]

        if children:
            for child_idx in children:
                self._write_joint(lines, child_idx, indent_level + 1, scale)
        else:
            # End site for leaf joints
            lines.append(f"{indent}  End Site")
            lines.append(f"{indent}  {{")
            lines.append(f"{indent}    OFFSET 0.0 0.0 0.0")
            lines.append(f"{indent}  }}")

        lines.append(f"{indent}}}")


NODE_CLASS_MAPPINGS = {
    "SMPLtoBVH": SMPLtoBVH,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SMPLtoBVH": "SMPL to BVH Converter",
}
