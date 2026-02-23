"""
GVHMRInference Node - Performs motion capture inference on video with SAM3 masks
"""

import os
from pathlib import Path
import torch
import folder_paths
import numpy as np
import cv2
from typing import Dict, Tuple
from tqdm import tqdm

# Import GVHMR components
from .motion_utils.pylogger import Log
from .motion_utils.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, create_camera_sensor
from .motion_utils.geo_transform import compute_cam_angvel
from .motion_utils.pytorch3d_shim import axis_angle_to_matrix
from .body_model.smplx_utils import make_smplx
from .motion_utils.simple_vo import SimpleVO

# Check DPVO availability
DPVO_AVAILABLE = False
try:
    from .dpvo.dpvo import DPVO
    from .dpvo.config import cfg as dpvo_cfg
    from .dpvo.utils import Timer
    DPVO_AVAILABLE = True
    Log.info("[GVHMRInference] DPVO is available")
except Exception:
    Log.info("[GVHMRInference] DPVO not installed - only SimpleVO will be available")

# Import local utilities (renamed to avoid conflict with ComfyUI's utils package)
from .gvhmr_utils import (
    extract_bbox_from_numpy_mask,
    bbox_to_xyxy,
    expand_bbox,
)

import gc
import tempfile as tempfile_module


from .shared_utils import next_sequential_filename as _next_sequential_filename


def _clear_cuda_memory():
    """Clear GPU memory cache between pipeline stages."""
    import comfy.model_management
    gc.collect()
    comfy.model_management.soft_empty_cache()


def _log_memory(label):
    """Log current process RSS memory usage (actual current, not peak)."""
    import comfy.model_management
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_mb = int(line.split()[1]) / 1024  # kB -> MB
                    break
            else:
                rss_mb = -1
        gpu_mb = ""
        if comfy.model_management.get_torch_device().type == "cuda":
            gpu_mb = f", GPU={torch.cuda.memory_allocated() / 1024**2:.0f} MB"
        Log.info(f"[Memory] {label}: RSS={rss_mb:.0f} MB{gpu_mb}")
    except Exception:
        Log.info(f"[Memory] {label}: (unable to read /proc/self/status)")



def _save_tensor_to_disk(tensor, prefix="tensor"):
    """Save tensor to temp file and return path."""
    import safetensors.torch
    fd, path = tempfile_module.mkstemp(suffix=".safetensors", prefix=prefix)
    os.close(fd)
    safetensors.torch.save_file({"t": tensor.cpu().contiguous()}, path)
    return path


def _load_tensor_from_disk(path, device="cpu"):
    """Load tensor from disk and delete temp file."""
    import safetensors.torch
    sd = safetensors.torch.load_file(path, device=str(device))
    os.remove(path)
    return sd["t"]



def _read_video_np(video_path: str) -> np.ndarray:
    """Read video file into numpy array (N, H, W, 3) RGB uint8."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def _quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices.

    Args:
        quaternions: Tensor of shape (N, 4) with quaternions in [w, x, y, z] format

    Returns:
        Rotation matrices of shape (N, 3, 3)
    """
    w, x, y, z = quaternions.unbind(-1)

    # Compute rotation matrix elements
    xx = 2 * x * x
    yy = 2 * y * y
    zz = 2 * z * z
    xy = 2 * x * y
    xz = 2 * x * z
    yz = 2 * y * z
    wx = 2 * w * x
    wy = 2 * w * y
    wz = 2 * w * z

    R = torch.stack([
        torch.stack([1 - yy - zz, xy - wz, xz + wy], dim=-1),
        torch.stack([xy + wz, 1 - xx - zz, yz - wx], dim=-1),
        torch.stack([xz - wy, yz + wx, 1 - xx - yy], dim=-1),
    ], dim=-2)

    return R


def _run_dpvo(images_np: np.ndarray, intrinsics: torch.Tensor, dpvo_dir: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run DPVO visual odometry on images.

    Args:
        images_np: RGB images as numpy array (N, H, W, 3)
        intrinsics: Camera intrinsics matrix (3, 3) or (N, 3, 3)
        dpvo_dir: Path to DPVO model directory containing dpvo.pth and config.yaml

    Returns:
        Tuple of (R_w2c, t_w2c) - rotation matrices (N, 3, 3) and translations (N, 3)
    """
    if not DPVO_AVAILABLE:
        raise RuntimeError("DPVO is not installed")

    if not dpvo_dir:
        raise RuntimeError("DPVO directory not configured. Enable load_dpvo in LoadGVHMRModels.")

    dpvo_path = Path(dpvo_dir)
    checkpoint_path = str(dpvo_path / "dpvo.pth")
    config_path = dpvo_path / "config.yaml"

    if not Path(checkpoint_path).exists():
        raise RuntimeError(f"DPVO checkpoint not found: {checkpoint_path}")

    Log.info(f"[GVHMRInference] Running DPVO with checkpoint: {checkpoint_path}")

    num_frames, height, width = images_np.shape[:3]

    # Get intrinsics values
    if intrinsics.dim() == 3:
        K = intrinsics[0]
    else:
        K = intrinsics
    fx, fy = K[0, 0].item(), K[1, 1].item()
    cx, cy = K[0, 2].item(), K[1, 2].item()

    # Load DPVO config
    if config_path.exists():
        dpvo_cfg.merge_from_file(str(config_path))
    else:
        Log.warn(f"[GVHMRInference] DPVO config not found at {config_path}, using defaults")
    cfg = dpvo_cfg

    slam = DPVO(cfg, checkpoint_path, ht=height, wd=width, viz=False)

    # Process frames
    for i, frame in enumerate(tqdm(images_np, desc="DPVO")):
        # DPVO expects BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        intrinsics_arr = np.array([fx, fy, cx, cy])
        slam(i, frame_bgr, intrinsics_arr)

    # Get trajectory
    # DPVO outputs poses as (N, 7): [tx, ty, tz, qx, qy, qz, qw]
    poses, tstamps = slam.terminate()

    if poses is None or len(poses) == 0:
        raise RuntimeError("DPVO failed to estimate camera poses")

    # Convert to torch tensors
    poses = torch.from_numpy(poses).float()

    # Extract translation (first 3 elements)
    t_w2c = poses[:, :3]

    # Extract quaternion [qx, qy, qz, qw] -> convert to [w, x, y, z]
    quat_xyzw = poses[:, 3:7]
    quat_wxyz = torch.cat([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], dim=-1)

    # Convert quaternion to rotation matrix
    R_w2c = _quaternion_to_matrix(quat_wxyz)

    # DPVO gives camera-to-world, we need world-to-camera
    R_w2c = R_w2c.transpose(-1, -2)  # Inverse of rotation = transpose
    t_w2c = -torch.bmm(R_w2c, t_w2c.unsqueeze(-1)).squeeze(-1)

    Log.info(f"[GVHMRInference] DPVO complete: {len(poses)} poses estimated")

    return R_w2c, t_w2c


class GVHMRInference:
    """
    ComfyUI node for GVHMR motion capture inference.
    Takes video frames and SAM3 masks, outputs SMPL parameters and 3D mesh.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {
                    "tooltip": "Input video (e.g. from Load Video node). Frames are read directly from the video file — no large float32 tensors in RAM."
                }),
                "video_mask": ("VIDEO", {
                    "tooltip": "Mask video where the person is white on black background (e.g. SAM3 mask rendered to video). Must have the same or similar frame count as video."
                }),
                "config": ("GVHMR_CONFIG",),  # Config from LoadGVHMRModels
                "moving_camera": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable if camera is moving (runs visual odometry to estimate camera motion)"
                }),
            },
            "optional": {
                "focal_length_mm": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 300,
                    "tooltip": "Camera focal length in mm (0 = auto-estimate). Ignored if intrinsics input is connected. Phones: 13-77mm typical"
                }),
                "bbox_scale": ("FLOAT", {
                    "default": 1.2,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Expand bounding box by this factor to ensure full person capture"
                }),
                "vo_method": (["simple_vo", "dpvo"], {
                    "default": "simple_vo",
                    "tooltip": "Visual odometry method. DPVO is more accurate but requires extra dependencies (dpvo package + checkpoint)"
                }),
                "vo_scale": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Scale factor for visual odometry processing (lower = faster, only used when static_camera=False)"
                }),
                "vo_step": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 30,
                    "tooltip": "Frame step for feature matching in VO (higher = faster but less accurate, only used when static_camera=False)"
                }),
                "intrinsics": ("INTRINSICS", {
                    "tooltip": "Camera intrinsics matrix (3x3). Connect from DepthAnything V3 or other source. Overrides focal_length_mm if provided."
                }),
                "chunk_size": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Frames per chunk during preprocessing. Higher = faster but uses more RAM. Lower = slower but saves RAM. 32 is a good default, use 10-16 if low on RAM."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("npz_path", "camera_npz_path", "info")
    FUNCTION = "run_inference"
    CATEGORY = "MotionCapture/GVHMR"

    @classmethod
    def _load_models(cls, config: Dict) -> Dict:
        """Load GVHMR models based on config."""
        import comfy.model_management
        import comfy.model_patcher
        from pathlib import Path
        from .gvhmr import DemoPL, Pipeline, NetworkEncoderRoPE, EnDecoder
        from .vitpose import VitPoseExtractor, Extractor

        Log.info("[GVHMRInference] Loading GVHMR models...")

        gvhmr_path = Path(config["gvhmr_path"])

        # Resolve precision using ComfyUI native detection
        precision = config.get("precision", "auto")
        if precision == "auto":
            device = comfy.model_management.get_torch_device()
            if comfy.model_management.should_use_bf16(device):
                dtype = torch.bfloat16
            elif comfy.model_management.should_use_fp16(device):
                dtype = torch.float16
            else:
                dtype = torch.float32
        elif precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        Log.info(f"[GVHMRInference] Precision: {dtype}")

        # Build GVHMR model directly (no Hydra)
        Log.info(f"[GVHMRInference] Loading GVHMR from {gvhmr_path}...")
        device = str(comfy.model_management.get_torch_device())

        with torch.device("meta"):
            denoiser3d = NetworkEncoderRoPE(
                output_dim=151,
                max_len=120,
                cliffcam_dim=3,
                cam_angvel_dim=6,
                imgseq_dim=1024,
                latent_dim=512,
                num_layers=12,
                num_heads=8,
                mlp_ratio=4.0,
                pred_cam_dim=3,
                static_conf_dim=6,
                dropout=0.1,
                avgbeta=True,
            )
            endecoder = EnDecoder(
                stats_name="MM_V1_AMASS_LOCAL_BEDLAM_CAM",
                noise_pose_k=10,
            )
            pipeline = Pipeline(
                denoiser3d=denoiser3d,
                endecoder=endecoder,
                normalize_cam_angvel=True,
                weights=None,
                static_conf=None,
            )
            model_gvhmr = DemoPL(pipeline=pipeline)

        model_gvhmr.load_pretrained_model(str(gvhmr_path))
        model_gvhmr.eval()

        # Safety net: reinitialize any stray meta-device buffers before .to()
        # (persistent=False buffers aren't in checkpoint and may still be on meta)
        for module in model_gvhmr.modules():
            for name, buf in list(module._buffers.items()):
                if buf is not None and buf.device.type == "meta":
                    Log.warn(f"[GVHMRInference] Reinitializing meta buffer: {type(module).__name__}.{name}")
                    module._buffers[name] = torch.zeros_like(buf, device="cpu")

        model_gvhmr.to(dtype=dtype)
        # Keep the endecoder (SMPL body model + affine decoder) in float32.
        # It's not part of the neural network — just topology data and mean/std stats.
        # Postprocessing utility libraries (IK, quaternion, matrix) assume float32.
        model_gvhmr.pipeline.endecoder.float()

        # Initialize preprocessing components (paths from config, stay on CPU)
        Log.info("[GVHMRInference] Initializing ViTPose extractor...")
        vitpose_extractor = VitPoseExtractor(dtype=dtype, ckpt_path=config.get("vitpose_path"))

        Log.info("[GVHMRInference] Initializing feature extractor...")
        feature_extractor = Extractor(dtype=dtype, ckpt_path=config.get("hmr2_path"))

        # Wrap all models in ModelPatcher for ComfyUI VRAM management
        from .lowvram import _enable_lowvram_cast

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        _enable_lowvram_cast(model_gvhmr)
        gvhmr_patcher = comfy.model_patcher.ModelPatcher(
            model_gvhmr, load_device=load_device, offload_device=offload_device,
        )

        _enable_lowvram_cast(vitpose_extractor.pose)
        vitpose_patcher = comfy.model_patcher.ModelPatcher(
            vitpose_extractor.pose, load_device=load_device, offload_device=offload_device,
        )

        _enable_lowvram_cast(feature_extractor.extractor)
        hmr2_patcher = comfy.model_patcher.ModelPatcher(
            feature_extractor.extractor, load_device=load_device, offload_device=offload_device,
        )

        gc.collect()
        _log_memory("After all models loaded")

        model_bundle = {
            "gvhmr": model_gvhmr,
            "vitpose_extractor": vitpose_extractor,
            "feature_extractor": feature_extractor,
            "gvhmr_patcher": gvhmr_patcher,
            "vitpose_patcher": vitpose_patcher,
            "hmr2_patcher": hmr2_patcher,
            "device": device,
            "paths": config,
        }

        Log.info("[GVHMRInference] All models loaded successfully!")
        return model_bundle

    @classmethod
    def _get_or_load_model(cls, config: Dict) -> Dict:
        """Get cached model or load new one based on config."""
        # Load fresh model each time — comfy-env handles GPU memory management
        return cls._load_models(config)

    def prepare_data_from_videos(
        self,
        video,
        video_mask,
        model_bundle: Dict,
        static_camera: bool,
        focal_length_mm: int,
        bbox_scale: float,
        vo_method: str,
        vo_scale: float,
        vo_step: int,
        intrinsics: torch.Tensor = None,
        dpvo_dir: str = "",
        chunk_size: int = 32,
    ) -> Dict:
        """
        Prepare data dictionary for GVHMR inference from VIDEO inputs.
        Reads frames chunk-by-chunk from video files to minimize RAM usage.
        """
        import comfy.model_management
        import av

        device = model_bundle["device"]

        # Get frame counts and dimensions from video objects
        video_frame_count = video.get_frame_count()
        mask_frame_count = video_mask.get_frame_count()
        width, height = video.get_dimensions()
        mask_width, mask_height = video_mask.get_dimensions()

        if video_frame_count != mask_frame_count:
            Log.warn(f"[GVHMRInference] Video frames ({video_frame_count}) != mask frames ({mask_frame_count}). Using minimum.")
        batch_size = min(video_frame_count, mask_frame_count)

        Log.info(f"[GVHMRInference] Video: {video_frame_count} frames @ {width}x{height}")
        Log.info(f"[GVHMRInference] Mask: {mask_frame_count} frames @ {mask_width}x{mask_height}")
        Log.info(f"[GVHMRInference] Processing {batch_size} frames, static_camera={static_camera}")
        _log_memory("Start of prepare_data_from_videos")

        # --- Phase 1: Extract bounding boxes from mask video ---
        Log.info("[GVHMRInference] Phase 1: Extracting bounding boxes from mask video...")
        mask_source = video_mask.get_stream_source()
        bboxes_xywh = []
        with av.open(mask_source, mode='r') as container:
            stream = next(s for s in container.streams if s.type == 'video')
            frame_idx = 0
            for frame in container.decode(stream):
                if frame_idx >= batch_size:
                    break
                mask_np = frame.to_ndarray(format='gray')  # (H, W) uint8
                bbox = extract_bbox_from_numpy_mask(mask_np)
                bboxes_xywh.append(bbox)
                frame_idx += 1

        Log.info(f"[GVHMRInference] Extracted {len(bboxes_xywh)} bounding boxes")
        _log_memory("After bbox extraction from mask video")

        # Scale bboxes if mask and video have different resolutions
        if (mask_width, mask_height) != (width, height):
            Log.warn(f"[GVHMRInference] Mask resolution ({mask_width}x{mask_height}) != video resolution ({width}x{height}). Scaling bboxes.")
            scale_x = width / mask_width
            scale_y = height / mask_height
            bboxes_xywh = [
                [int(b[0] * scale_x), int(b[1] * scale_y), int(b[2] * scale_x), int(b[3] * scale_y)]
                for b in bboxes_xywh
            ]

        # Expand bounding boxes
        bboxes_xywh = [
            expand_bbox(bbox, scale=bbox_scale, img_width=width, img_height=height)
            for bbox in bboxes_xywh
        ]

        # Convert to xyxy format for processing
        bboxes_xyxy = torch.tensor([bbox_to_xyxy(bbox) for bbox in bboxes_xywh], dtype=torch.float32)

        # Get bbx_xys format (used by GVHMR)
        bbx_xys = get_bbx_xys_from_xyxy(bboxes_xyxy, base_enlarge=1.0).float()
        Log.info(f"[GVHMRInference] bboxes_xyxy: {bboxes_xyxy.shape}, bbx_xys: {bbx_xys.shape}")

        # --- Phase 2: Two-pass frame processing (GPU memory optimization) ---
        # Only one large model on GPU at a time (~2.7 GB peak vs ~5.2 GB).
        # Cropped 256×256 tensors saved between passes (~0.75 MB/frame).
        from .vitpose.feat_extractor import get_batch

        vitpose_extractor = model_bundle["vitpose_extractor"]
        feature_extractor = model_bundle["feature_extractor"]

        CHUNK_SIZE = max(1, chunk_size)

        # --- Pass 1: ViTPose (decode video + crop + extract keypoints) ---
        # Load ViTPose onto GPU (ModelPatcher automatically offloads other models)
        comfy.model_management.load_models_gpu([model_bundle["vitpose_patcher"]])
        _log_memory("ViTPose on GPU")

        all_kp2d = []
        saved_crops = []  # (chunk_tensor, chunk_bbx_processed) for Pass 2

        # For moving camera: write temp video incrementally
        temp_video_path = None
        _video_writer = None
        if not static_camera:
            import tempfile
            temp_video_path = tempfile.mktemp(suffix=".mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            _video_writer = cv2.VideoWriter(temp_video_path, fourcc, 30, (width, height))

        Log.info(f"[GVHMRInference] Pass 1/2 (ViTPose): {batch_size} frames in chunks of {CHUNK_SIZE}...")

        video_source = video.get_stream_source()
        with av.open(video_source, mode='r') as container:
            stream = next(s for s in container.streams if s.type == 'video')
            chunk_frames = []
            frame_idx = 0
            chunk_start = 0

            for frame in container.decode(stream):
                if frame_idx >= batch_size:
                    break
                frame_np = frame.to_ndarray(format='rgb24')  # (H, W, 3) uint8
                chunk_frames.append(frame_np)
                frame_idx += 1

                # Process chunk when full or at end of video
                if len(chunk_frames) >= CHUNK_SIZE or frame_idx >= batch_size:
                    chunk_end = chunk_start + len(chunk_frames)
                    Log.info(f"[GVHMRInference] ViTPose chunk [{chunk_start}:{chunk_end}] ({len(chunk_frames)} frames)")

                    chunk_np = np.stack(chunk_frames)  # (N, H, W, 3) uint8

                    # Write to temp video for VO (if moving camera)
                    if _video_writer is not None:
                        for _frame in chunk_np:
                            _video_writer.write(cv2.cvtColor(_frame, cv2.COLOR_RGB2BGR))

                    # Crop/resize chunk via get_batch
                    chunk_bbx_xys = bbx_xys[chunk_start:chunk_end]
                    chunk_tensor, chunk_bbx_processed = get_batch(chunk_np, chunk_bbx_xys, img_ds=1.0, path_type="np")

                    # ViTPose on chunk
                    chunk_kp2d = vitpose_extractor.extract(chunk_tensor, chunk_bbx_processed, img_ds=1.0)
                    all_kp2d.append(chunk_kp2d.cpu())

                    # Save crops for HMR2 pass (256×256 tensors — tiny)
                    saved_crops.append((chunk_tensor.cpu(), chunk_bbx_processed.clone()))

                    del chunk_np, chunk_kp2d
                    chunk_frames = []
                    gc.collect()
                    _clear_cuda_memory()
                    _log_memory(f"After ViTPose chunk [{chunk_start}:{chunk_end}]")
                    chunk_start = chunk_end

        if _video_writer is not None:
            _video_writer.release()
            _video_writer = None

        # --- Pass 2: HMR2 (iterate saved crops — no video decoding) ---
        # Load HMR2 onto GPU (ModelPatcher automatically offloads ViTPose)
        comfy.model_management.load_models_gpu([model_bundle["hmr2_patcher"]])
        _log_memory("HMR2 on GPU")

        all_f_imgseq = []
        Log.info(f"[GVHMRInference] Pass 2/2 (HMR2): {len(saved_crops)} chunks...")

        for i, (chunk_tensor, chunk_bbx_processed) in enumerate(saved_crops):
            Log.info(f"[GVHMRInference] HMR2 chunk {i+1}/{len(saved_crops)}")
            chunk_features = feature_extractor.extract_video_features(chunk_tensor, chunk_bbx_processed, img_ds=1.0)
            all_f_imgseq.append(chunk_features.cpu())
            del chunk_features
            _clear_cuda_memory()

        del saved_crops
        gc.collect()
        _clear_cuda_memory()
        _log_memory("After HMR2 pass")

        # Concatenate results from all chunks
        kp2d = torch.cat(all_kp2d, dim=0)
        del all_kp2d
        f_imgseq = torch.cat(all_f_imgseq, dim=0)
        del all_f_imgseq
        gc.collect()

        Log.info(f"[GVHMRInference] ViTPose complete: kp2d={kp2d.shape} dtype={kp2d.dtype}")
        Log.info(f"[GVHMRInference] HMR2 complete: f_imgseq={f_imgseq.shape} dtype={f_imgseq.dtype}")
        _log_memory("After all chunks processed")

        # Offload features to disk for long videos to save GPU memory
        f_imgseq_path = None
        if f_imgseq.shape[0] > 300:
            f_imgseq_path = _save_tensor_to_disk(f_imgseq, "f_imgseq_")
            del f_imgseq
            f_imgseq = None
            _clear_cuda_memory()
            Log.info(f"[GVHMRInference] Features offloaded to disk: {f_imgseq_path}")
        else:
            f_imgseq = f_imgseq.cpu()
            _clear_cuda_memory()
            Log.info("[GVHMRInference] Features moved to CPU, GPU memory cleared")

        # Camera intrinsics: use provided K, or estimate from focal_length_mm, or auto-estimate
        if intrinsics is not None:
            Log.info(f"[GVHMRInference] Using provided camera intrinsics, input shape: {intrinsics.shape}")
            # Squeeze extra dimensions (DA3 outputs [1, 1, 3, 3])
            K = intrinsics.squeeze()
            while K.dim() > 3:
                K = K.squeeze(0)
            # Handle different input shapes
            if K.dim() == 2:  # (3, 3) - single matrix
                K_fullimg = K.unsqueeze(0).repeat(batch_size, 1, 1)
            elif K.dim() == 3 and K.shape[0] == 1:  # (1, 3, 3)
                K_fullimg = K.repeat(batch_size, 1, 1)
            elif K.dim() == 3 and K.shape[0] == batch_size:  # (B, 3, 3)
                K_fullimg = K
            else:
                Log.warn(f"[GVHMRInference] Unexpected intrinsics shape {K.shape}, falling back to estimation")
                K_fullimg = estimate_K(width, height).repeat(batch_size, 1, 1)
            Log.info(f"[GVHMRInference] Intrinsics fx={K_fullimg[0,0,0]:.1f}, fy={K_fullimg[0,1,1]:.1f}, cx={K_fullimg[0,0,2]:.1f}, cy={K_fullimg[0,1,2]:.1f}")
        elif focal_length_mm > 0:
            Log.info(f"[GVHMRInference] Using focal length: {focal_length_mm}mm")
            K_fullimg = create_camera_sensor(width, height, focal_length_mm)[2].repeat(batch_size, 1, 1)
        else:
            Log.info("[GVHMRInference] Auto-estimating camera intrinsics (53° FOV)")
            K_fullimg = estimate_K(width, height).repeat(batch_size, 1, 1)

        # Handle camera motion
        Log.info(f"[GVHMRInference] Camera mode: {'STATIC' if static_camera else 'MOVING'} (vo_method={vo_method})")
        t_w2c = None  # Camera translation
        if static_camera:
            R_w2c = torch.eye(3).repeat(batch_size, 1, 1)
        else:
            # Run visual odometry to estimate camera motion
            try:
                if vo_method == "dpvo":
                    if not DPVO_AVAILABLE:
                        Log.warn("[GVHMRInference] DPVO requested but not installed, falling back to SimpleVO")
                        vo_method = "simple_vo"
                    elif not dpvo_dir:
                        Log.warn("[GVHMRInference] DPVO requested but dpvo_dir not configured. Enable load_dpvo in LoadGVHMRModels. Falling back to SimpleVO")
                        vo_method = "simple_vo"
                    else:
                        Log.info(f"[GVHMRInference] Running DPVO for moving camera (dir={dpvo_dir})...")
                        # Read frames from temp video written during chunked processing
                        dpvo_frames = _read_video_np(temp_video_path)
                        R_w2c, t_w2c = _run_dpvo(dpvo_frames, K_fullimg, dpvo_dir)
                        del dpvo_frames
                        _clear_cuda_memory()
                        Log.info(f"[GVHMRInference] DPVO complete, camera translation range: {t_w2c.abs().max().item():.3f}m")

                if vo_method == "simple_vo":
                    Log.info(f"[GVHMRInference] Running SimpleVO for moving camera (scale={vo_scale}, step={vo_step})...")
                    Log.info(f"[GVHMRInference] Using temp video: {temp_video_path} ({batch_size} frames)")

                    # Run SimpleVO using temp video written during chunked processing
                    f_mm = focal_length_mm if focal_length_mm > 0 else 24
                    Log.info(f"[GVHMRInference] SimpleVO params: f_mm={f_mm}, scale={vo_scale}, step={vo_step}, method=sift")
                    vo = SimpleVO(temp_video_path, scale=vo_scale, step=vo_step, method="sift", f_mm=f_mm)
                    T_w2c_list = vo.compute()
                    Log.info(f"[GVHMRInference] SimpleVO returned {len(T_w2c_list)} poses (expected {batch_size})")

                    # Extract rotation matrices (upper-left 3x3 of each 4x4 transform)
                    R_w2c = torch.tensor(np.stack([T[:3, :3] for T in T_w2c_list]), dtype=torch.float32)
                    # Extract translation vectors (right column of each 4x4 transform)
                    t_w2c = torch.tensor(np.stack([T[:3, 3] for T in T_w2c_list]), dtype=torch.float32)

                    Log.info(f"[GVHMRInference] R_w2c: {R_w2c.shape}, t_w2c: {t_w2c.shape}")
                    Log.info(f"[GVHMRInference] Translation range: min={t_w2c.min().item():.4f}, max={t_w2c.max().item():.4f}")
                    _clear_cuda_memory()

                # Clean up temp video file
                if temp_video_path is not None and os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                    Log.info("[GVHMRInference] Temp video cleaned up")

            except Exception as e:
                Log.error(f"[GVHMRInference] Visual odometry failed: {e}", exc_info=True)
                Log.warn("[GVHMRInference] Falling back to static camera")
                R_w2c = torch.eye(3).repeat(batch_size, 1, 1)
                t_w2c = None
                # Clean up temp video on error
                if temp_video_path is not None and os.path.exists(temp_video_path):
                    os.remove(temp_video_path)

        Log.info(f"[GVHMRInference] R_w2c: {R_w2c.shape}, t_w2c: {t_w2c.shape if t_w2c is not None else 'None'}")
        cam_angvel = compute_cam_angvel(R_w2c)
        Log.info(f"[GVHMRInference] cam_angvel: {cam_angvel.shape}")

        # Prepare data dictionary
        data = {
            "length": torch.tensor(batch_size),
            "bbx_xys": bbx_xys,
            "kp2d": kp2d,
            "K_fullimg": K_fullimg,
            "cam_angvel": cam_angvel,
            "f_imgseq": f_imgseq,
            "f_imgseq_path": f_imgseq_path,  # Path to offloaded features (None if in memory)
        }

        Log.info("[GVHMRInference] === Data dict summary ===")
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                Log.info(f"[GVHMRInference]   {k}: {v.shape} dtype={v.dtype}")
            elif v is None:
                Log.info(f"[GVHMRInference]   {k}: None")
            else:
                Log.info(f"[GVHMRInference]   {k}: {type(v).__name__} = {v}")

        # Store camera transforms for potential future use
        camera_data = {
            "R_w2c": R_w2c,
            "t_w2c": t_w2c,
            "K_fullimg": K_fullimg,
            "img_width": width,
            "img_height": height,
        }

        _log_memory("End of prepare_data_from_videos")
        return data, camera_data

    def run_inference(
        self,
        video,
        video_mask,
        config: Dict,
        moving_camera: bool = False,
        focal_length_mm: int = 0,
        bbox_scale: float = 1.2,
        vo_method: str = "simple_vo",
        vo_scale: float = 0.5,
        vo_step: int = 8,
        intrinsics: torch.Tensor = None,
        chunk_size: int = 32,
    ):
        """
        Run GVHMR inference on video with mask video.
        Reads frames directly from video files — no large float32 tensors in RAM.
        """
        import comfy.model_management
        try:
            static_camera = not moving_camera
            Log.info("=" * 60)
            Log.info("[GVHMRInference] Starting GVHMR inference with parameters:")
            Log.info(f"  video:           {video.get_frame_count()} frames @ {video.get_dimensions()}")
            Log.info(f"  video_mask:      {video_mask.get_frame_count()} frames @ {video_mask.get_dimensions()}")
            Log.info(f"  moving_camera:   {moving_camera}")
            Log.info(f"  focal_length_mm: {focal_length_mm}")
            Log.info(f"  bbox_scale:      {bbox_scale}")
            Log.info(f"  vo_method:       {vo_method}")
            Log.info(f"  vo_scale:        {vo_scale}")
            Log.info(f"  vo_step:         {vo_step}")
            Log.info(f"  intrinsics:      {intrinsics.shape if intrinsics is not None else 'None'}")
            Log.info(f"  chunk_size:      {chunk_size}")
            Log.info(f"  config keys:     {list(config.keys())}")
            Log.info(f"  dpvo_dir:        {config.get('dpvo_dir', '')}")
            Log.info("=" * 60)
            _log_memory("Start of run_inference")

            # Load models based on config
            model = self._get_or_load_model(config)

            # Get DPVO directory from config (just a path string)
            dpvo_dir = config.get("dpvo_dir", "")

            # Prepare data (reads frames from video files chunk-by-chunk)
            data, camera_data = self.prepare_data_from_videos(
                video, video_mask, model, static_camera, focal_length_mm, bbox_scale, vo_method, vo_scale, vo_step, intrinsics, dpvo_dir, chunk_size
            )

            batch_size_saved = data["length"].item()
            img_width_saved = camera_data["img_width"]
            img_height_saved = camera_data["img_height"]

            # Reload features from disk if they were offloaded
            if data.get("f_imgseq_path") is not None:
                Log.info("[GVHMRInference] Reloading features from disk...")
                data["f_imgseq"] = _load_tensor_from_disk(data["f_imgseq_path"], device="cpu")
                data["f_imgseq_path"] = None
                Log.info("[GVHMRInference] Features reloaded from disk")

            # Run GVHMR inference — load onto GPU (ModelPatcher offloads HMR2 automatically)
            comfy.model_management.load_models_gpu([model["gvhmr_patcher"]])
            _log_memory("Before GVHMR predict")
            Log.info(f"[GVHMRInference] Running GVHMR model (static_cam={static_camera})...")
            gvhmr_model = model["gvhmr"]
            device = model["device"]

            Log.info("[GVHMRInference] === Pre-predict data shapes ===")
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    Log.info(f"[GVHMRInference]   data[{k}]: {v.shape} dtype={v.dtype} device={v.device}")

            with torch.no_grad():
                pred = gvhmr_model.predict(data, static_cam=static_camera)

            # Clear GPU memory after inference
            _clear_cuda_memory()
            _log_memory("After GVHMR predict")

            # Extract SMPL parameters
            smpl_params = {
                "global": pred["smpl_params_global"],
                "incam": pred["smpl_params_incam"],
                "K_fullimg": camera_data["K_fullimg"],
                "R_w2c": camera_data["R_w2c"],
                "t_w2c": camera_data["t_w2c"],
            }

            # Save SMPL params to NPZ file (avoids tensor serialization issues)
            output_dir = Path(folder_paths.get_output_directory())
            output_dir.mkdir(parents=True, exist_ok=True)
            npz_filename = _next_sequential_filename(output_dir, "smpl_params", ".npz")
            npz_path = output_dir / npz_filename

            global_params = pred["smpl_params_global"]
            incam_params = pred["smpl_params_incam"]
            save_dict = {
                'body_pose': global_params['body_pose'].cpu().numpy(),
                'betas': global_params['betas'].cpu().numpy(),
                'global_orient': global_params['global_orient'].cpu().numpy(),
                'transl': global_params['transl'].cpu().numpy(),
                'global_orient_incam': incam_params['global_orient'].cpu().numpy(),
                'transl_incam': incam_params['transl'].cpu().numpy(),
            }

            # Save camera trajectory when moving camera is enabled
            camera_npz_path_str = ""
            t_w2c = camera_data["t_w2c"]
            K_fullimg_out = camera_data["K_fullimg"]
            img_height, img_width = img_height_saved, img_width_saved

            if not static_camera and t_w2c is not None:
                # Compute camera-to-world transform in gravity-aligned (GV) frame
                # This avoids OpenCV/OpenGL convention issues -- the result is in the
                # same Y-up coordinate system as the SMPL mesh.
                incam_params = pred["smpl_params_incam"]
                R_body_world = axis_angle_to_matrix(global_params['global_orient'])  # (F, 3, 3)
                R_body_cam = axis_angle_to_matrix(incam_params['global_orient'])     # (F, 3, 3)
                R_cam2world = R_body_world @ R_body_cam.transpose(-1, -2)            # (F, 3, 3)
                t_cam2world = global_params['transl'] - torch.bmm(
                    R_cam2world, incam_params['transl'].unsqueeze(-1)
                ).squeeze(-1)  # (F, 3)

                R_cam2world_np = R_cam2world.cpu().numpy().astype(np.float32)
                t_cam2world_np = t_cam2world.cpu().numpy().astype(np.float32)
                K_np = K_fullimg_out.cpu().numpy().astype(np.float32) if isinstance(K_fullimg_out, torch.Tensor) else np.array(K_fullimg_out, dtype=np.float32)

                Log.info(f"[GVHMRInference] Camera trajectory: R_cam2world {R_cam2world_np.shape}, t_cam2world {t_cam2world_np.shape}")
                Log.info(f"[GVHMRInference] Camera pos frame 0: [{t_cam2world_np[0,0]:.3f}, {t_cam2world_np[0,1]:.3f}, {t_cam2world_np[0,2]:.3f}]")

                # Add camera data to main NPZ
                save_dict['R_cam2world'] = R_cam2world_np
                save_dict['t_cam2world'] = t_cam2world_np
                save_dict['K_fullimg'] = K_np
                save_dict['img_width'] = np.array([img_width])
                save_dict['img_height'] = np.array([img_height])

                # Also save separate camera NPZ
                camera_npz_filename = _next_sequential_filename(output_dir, "camera_trajectory", ".npz")
                camera_npz_path = output_dir / camera_npz_filename
                np.savez(
                    str(camera_npz_path),
                    R_cam2world=R_cam2world_np,
                    t_cam2world=t_cam2world_np,
                    K_fullimg=K_np,
                    img_width=np.array([img_width]),
                    img_height=np.array([img_height]),
                )
                camera_npz_path_str = str(camera_npz_path)
                Log.info(f"[GVHMRInference] Saved camera trajectory to: {camera_npz_path}")

            np.savez(str(npz_path), **save_dict)
            Log.info(f"[GVHMRInference] Saved SMPL params to: {npz_path}")

            # Create info string
            num_frames = batch_size_saved
            vo_info = "N/A (static)" if static_camera else vo_method
            info = (
                f"GVHMR Inference Complete\n"
                f"Frames processed: {num_frames}\n"
                f"Static camera: {static_camera}\n"
                f"VO method: {vo_info}\n"
                f"Device: {device}\n"
                f"NPZ saved: {npz_path}\n"
            )

            Log.info("[GVHMRInference] Inference complete!")

            del model
            _clear_cuda_memory()

            _log_memory("Final (before return)")
            return (str(npz_path), camera_npz_path_str, info)

        except Exception as e:
            error_msg = f"GVHMR Inference failed: {str(e)}"
            Log.error(error_msg, exc_info=True)
            # Return placeholder on error
            return ("", "", error_msg)



# Node registration
NODE_CLASS_MAPPINGS = {
    "GVHMRInference": GVHMRInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GVHMRInference": "GVHMR Inference",
}
