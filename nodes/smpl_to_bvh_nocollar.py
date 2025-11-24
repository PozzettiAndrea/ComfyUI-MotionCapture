"""
SMPLtoBVH No-Collar Fix - Convert SMPL motion data to BVH without collar bones

This variant simplifies the skeleton by merging collar bones into shoulders.
Many BVH skeletons don't include clavicles, and having two consecutive horizontal
bones (Collar → Shoulder) can cause gimbal lock and unusual rotation behavior.

CHANGES FROM ORIGINAL:
- Remove L_Collar and R_Collar joints
- Merge collar offset into shoulder: [-0.15, 0.0, 0.0] + [-0.1, 0.0, 0.0] = [-0.25, 0.0, 0.0]
- Connect shoulders directly to Spine3 (parent: 9 instead of 13/14)
- Combine collar and shoulder rotations
"""

from pathlib import Path
from typing import Dict, Tuple
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from hmr4d.utils.pylogger import Log


# SMPL skeleton hierarchy - 19-joint variant (21-joint with collars merged)
# Note: This maps to 21-joint SMPL but outputs 19 joints in BVH
SMPL_19_JOINT_NAMES = [
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
    'Head',         # 13  (was 15, now 13 after removing collars)
    'L_Shoulder',   # 14  (was 16, merges L_Collar)
    'R_Shoulder',   # 15  (was 17, merges R_Collar)
    'L_Elbow',      # 16  (was 18)
    'R_Elbow',      # 17  (was 19)
    'L_Wrist',      # 18  (was 20)
    'R_Wrist',      # 19  (was 21)
]

# Parent indices for 19-joint variant (no collars)
SMPL_19_PARENTS = [
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
    12,  # 13: Head (parent: Neck)
    9,   # 14: L_Shoulder (parent: Spine3, not collar!)
    9,   # 15: R_Shoulder (parent: Spine3, not collar!)
    14,  # 16: L_Elbow (parent: L_Shoulder)
    15,  # 17: R_Elbow (parent: R_Shoulder)
    16,  # 18: L_Wrist (parent: L_Elbow)
    17,  # 19: R_Wrist (parent: R_Elbow)
]

# Mapping from 21-joint SMPL to 19-joint BVH (no collars)
# Maps SMPL joint index → BVH joint index (None means skip/merge)
SMPL21_TO_BVH19_MAP = {
    0: 0,    # Pelvis → Pelvis
    1: 1,    # L_Hip → L_Hip
    2: 2,    # R_Hip → R_Hip
    3: 3,    # Spine1 → Spine1
    4: 4,    # L_Knee → L_Knee
    5: 5,    # R_Knee → R_Knee
    6: 6,    # Spine2 → Spine2
    7: 7,    # L_Ankle → L_Ankle
    8: 8,    # R_Ankle → R_Ankle
    9: 9,    # Spine3 → Spine3
    10: 10,  # L_Foot → L_Foot
    11: 11,  # R_Foot → R_Foot
    12: 12,  # Neck → Neck
    13: 14,  # L_Collar → merged into L_Shoulder
    14: 15,  # R_Collar → merged into R_Shoulder
    15: 13,  # Head → Head
    16: 14,  # L_Shoulder → L_Shoulder (merge with collar)
    17: 15,  # R_Shoulder → R_Shoulder (merge with collar)
    18: 16,  # L_Elbow → L_Elbow
    19: 17,  # R_Elbow → R_Elbow
    20: 18,  # L_Wrist → L_Wrist
    21: 19,  # R_Wrist → R_Wrist
}

# Joint offsets in meters (no collar bones, merged into shoulders)
SMPL_OFFSETS_NOCOLLAR = {
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
    13: [0.0, 0.15, 0.0],        # Head (was index 15)
    14: [-0.25, 0.0, 0.0],       # L_Shoulder (MERGED: -0.15 + -0.1 from collar+shoulder)
    15: [0.25, 0.0, 0.0],        # R_Shoulder (MERGED: 0.15 + 0.1 from collar+shoulder)
    16: [0.0, -0.25, 0.0],       # L_Elbow (kept vertical for comparison)
    17: [0.0, -0.25, 0.0],       # R_Elbow (kept vertical for comparison)
    18: [0.0, -0.25, 0.0],       # L_Wrist
    19: [0.0, -0.25, 0.0],       # R_Wrist
}


class SMPLtoBVH_NoCollar:
    """
    Convert SMPL motion parameters to BVH without collar bones.

    This simplifies the skeleton by merging collar bones into shoulders,
    eliminating the double-horizontal-bone structure that can cause issues.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "smpl_params": ("SMPL_PARAMS",),
                "output_path": ("STRING", {
                    "default": "output/motion_nocollar.bvh",
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
        Convert SMPL parameters to BVH file format without collar bones.

        Args:
            smpl_params: SMPL parameters from GVHMRInference or LoadSMPL
            output_path: Path to save BVH file
            fps: Frames per second for the animation
            scale: Scale factor for the skeleton (1.0 = meters, 100.0 = centimeters)

        Returns:
            Tuple of (bvh_data_dict, file_path, info_string)
        """
        try:
            Log.info("[SMPLtoBVH No-Collar] Converting SMPL to BVH format WITHOUT collar bones...")

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

                Log.info(f"[SMPLtoBVH No-Collar] Detected {num_body_joints}-joint SMPL variant ({num_params} parameters)")

                # Validate parameter count
                if num_params % 3 != 0:
                    raise ValueError(f"Invalid body_pose size: {num_params} is not divisible by 3")

                body_pose = body_pose.reshape(-1, num_body_joints, 3)
            else:
                # Already in [F, J, 3] format
                num_body_joints = body_pose.shape[1]

            # Only support 21-joint for now (GVHMR variant)
            if num_body_joints != 21:
                raise ValueError(f"No-Collar variant only supports 21-joint SMPL. Got {num_body_joints} joints.")

            Log.info("[SMPLtoBVH No-Collar] Merging collar bones into shoulders (21 → 19 joints)")

            # Combine global_orient and body_pose: [F, 22, 3] for 21-joint
            if len(global_orient.shape) == 2:
                global_orient = global_orient[:, np.newaxis, :]  # [F, 1, 3]
            full_pose = np.concatenate([global_orient, body_pose], axis=1)  # [F, 22, 3]

            # Merge collar rotations into shoulders
            # Collar (13, 14) rotations get combined with Shoulder (16, 17)
            merged_pose = self._merge_collar_bones(full_pose)  # [F, 20, 3]

            num_frames = merged_pose.shape[0]
            num_total_joints = merged_pose.shape[1]
            frame_time = 1.0 / fps

            # Convert axis-angle rotations to Euler angles (ZXY order, BVH standard)
            euler_rotations = self._axis_angle_to_euler(merged_pose)  # [F, 20, 3]

            # Validate rotation ranges to detect potential issues
            rot_mins = np.min(euler_rotations, axis=(0, 1))
            rot_maxs = np.max(euler_rotations, axis=(0, 1))
            Log.info(f"[SMPLtoBVH No-Collar] Rotation ranges (degrees):")
            Log.info(f"  Z: [{rot_mins[0]:.1f}, {rot_maxs[0]:.1f}]")
            Log.info(f"  X: [{rot_mins[1]:.1f}, {rot_maxs[1]:.1f}]")
            Log.info(f"  Y: [{rot_mins[2]:.1f}, {rot_maxs[2]:.1f}]")

            # Write BVH file
            bvh_content = self._write_bvh(
                euler_rotations,
                transl,
                frame_time,
                scale,
                SMPL_19_JOINT_NAMES,
                SMPL_19_PARENTS
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
                "joint_names": SMPL_19_JOINT_NAMES,
                "num_joints": num_total_joints,
            }

            info = (
                f"SMPLtoBVH No-Collar Complete\n"
                f"Output: {output_path}\n"
                f"Frames: {num_frames}\n"
                f"FPS: {fps}\n"
                f"Frame time: {frame_time:.4f}s\n"
                f"Scale: {scale}x\n"
                f"Joints: {num_total_joints} (merged collars into shoulders)\n"
                f"Fix: Collar bones removed (simpler hierarchy)\n"
            )

            Log.info(f"[SMPLtoBVH No-Collar] Converted {num_frames} frames to {output_path}")
            return (bvh_data, str(output_path.absolute()), info)

        except Exception as e:
            error_msg = f"SMPLtoBVH No-Collar failed: {str(e)}"
            Log.error(error_msg)
            import traceback
            traceback.print_exc()
            return ({}, "", error_msg)

    def _merge_collar_bones(self, full_pose: np.ndarray) -> np.ndarray:
        """
        Merge collar bone rotations into shoulder rotations.

        SMPL 21-joint has:
          13: L_Collar, 14: R_Collar, 16: L_Shoulder, 17: R_Shoulder

        We combine collar+shoulder rotations and remove collar joints.

        Args:
            full_pose: [F, 22, 3] axis-angle rotations (21 body + 1 root)

        Returns:
            [F, 20, 3] axis-angle rotations (19 joints + 1 root, no collars)
        """
        num_frames = full_pose.shape[0]
        merged = np.zeros((num_frames, 20, 3))

        # For each frame, combine rotations
        for frame in range(num_frames):
            output_idx = 0

            for smpl_idx in range(22):  # 0 to 21 (22 total joints)
                if smpl_idx == 13 or smpl_idx == 14:
                    # Skip collar bones, will merge with shoulders
                    continue

                # Get target BVH index
                bvh_idx = SMPL21_TO_BVH19_MAP.get(smpl_idx)

                if smpl_idx == 16:  # L_Shoulder - merge with L_Collar (13)
                    # Combine rotations: collar → shoulder
                    collar_rot = R.from_rotvec(full_pose[frame, 13])
                    shoulder_rot = R.from_rotvec(full_pose[frame, 16])
                    combined_rot = collar_rot * shoulder_rot  # Rotation composition
                    merged[frame, bvh_idx] = combined_rot.as_rotvec()

                elif smpl_idx == 17:  # R_Shoulder - merge with R_Collar (14)
                    # Combine rotations: collar → shoulder
                    collar_rot = R.from_rotvec(full_pose[frame, 14])
                    shoulder_rot = R.from_rotvec(full_pose[frame, 17])
                    combined_rot = collar_rot * shoulder_rot  # Rotation composition
                    merged[frame, bvh_idx] = combined_rot.as_rotvec()

                else:
                    # Copy rotation as-is
                    merged[frame, bvh_idx] = full_pose[frame, smpl_idx]

        return merged

    def _get_bvh_joint_order(self, parent_indices: list) -> list:
        """
        Get the depth-first traversal order of joints for BVH motion data.
        BVH requires motion data to be in the same order as joints appear in hierarchy.

        Args:
            parent_indices: list of parent indices for each joint

        Returns:
            list of joint indices in depth-first order
        """
        joint_order = []

        def traverse(joint_idx):
            joint_order.append(joint_idx)
            # Find children of this joint
            children = [i for i, parent in enumerate(parent_indices) if parent == joint_idx]
            for child_idx in children:
                traverse(child_idx)

        # Start from root (joint 0)
        traverse(0)
        return joint_order

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
                # BVH uses ZXY intrinsic Euler angles in degrees
                # IMPORTANT: Use uppercase 'ZXY' for intrinsic rotations (not lowercase 'zxy' for extrinsic)
                euler_angles = rot.as_euler('ZXY', degrees=True)
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

        # Get BVH joint order (depth-first traversal)
        joint_order = self._get_bvh_joint_order(parent_indices)

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

            # Write joints in BVH hierarchy order (depth-first traversal)
            for joint_idx in joint_order:
                if joint_idx == 0:
                    # Root has translation + rotation
                    tx, ty, tz = translations[frame] * scale
                    frame_data.extend([f"{tx:.6f}", f"{ty:.6f}", f"{tz:.6f}"])
                    rz, rx, ry = rotations[frame, joint_idx]
                    frame_data.extend([f"{rz:.6f}", f"{rx:.6f}", f"{ry:.6f}"])
                else:
                    # Other joints only have rotation
                    rz, rx, ry = rotations[frame, joint_idx]
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
        offset = SMPL_OFFSETS_NOCOLLAR.get(joint_idx, [0.0, 0.0, 0.0])
        offset_scaled = [o * scale for o in offset]

        # Root joint
        if joint_idx == 0:
            lines.append(f"{indent}ROOT {joint_name}")
        else:
            lines.append(f"{indent}JOINT {joint_name}")

        lines.append(f"{indent}{{")
        lines.append(f"{indent}  OFFSET {offset_scaled[0]:.6f} {offset_scaled[1]:.6f} {offset_scaled[2]:.6f}")  # X, Y, Z order

        # Channels
        if joint_idx == 0:
            # Root has translation + rotation
            lines.append(f"{indent}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
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
    "SMPLtoBVH_NoCollar": SMPLtoBVH_NoCollar,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SMPLtoBVH_NoCollar": "SMPL to BVH (No Collar)",
}
