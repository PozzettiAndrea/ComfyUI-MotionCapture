"""
GVHMRInference Node - Performs motion capture inference on video with SAM3 masks
"""

import os
import sys
from pathlib import Path
import torch
import folder_paths
import numpy as np
import cv2
from typing import Dict, Tuple
from tqdm import tqdm

# Add nodes path for local utils (needed when run as subprocess)
NODES_PATH = Path(__file__).parent
sys.path.insert(0, str(NODES_PATH))

# Add vendor path for GVHMR
VENDOR_PATH = NODES_PATH / "vendor"
sys.path.insert(0, str(VENDOR_PATH))

# Import GVHMR components
from hmr4d.utils.pylogger import Log
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, create_camera_sensor
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.net_utils import to_cuda
from hmr4d.utils.preproc.relpose.simple_vo import SimpleVO

# Check DPVO availability
DPVO_AVAILABLE = False
try:
    from dpvo.dpvo import DPVO
    from dpvo.config import cfg as dpvo_cfg
    from dpvo.utils import Timer
    DPVO_AVAILABLE = True
    Log.info("[GVHMRInference] DPVO is available")
except ImportError:
    Log.info("[GVHMRInference] DPVO not installed - only SimpleVO will be available")

# Import local utilities (renamed to avoid conflict with ComfyUI's utils package)
from gvhmr_utils import (
    extract_bboxes_from_masks,
    bbox_to_xyxy,
    expand_bbox,
    normalize_image_tensor,
    validate_masks,
    validate_images,
)

import gc
import tempfile as tempfile_module


def _clear_cuda_memory():
    """Clear CUDA memory cache between pipeline stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _save_tensor_to_disk(tensor, prefix="tensor"):
    """Save tensor to temp file and return path."""
    fd, path = tempfile_module.mkstemp(suffix=".pt", prefix=prefix)
    os.close(fd)
    torch.save(tensor.cpu(), path)
    return path


def _load_tensor_from_disk(path, device="cuda"):
    """Load tensor from disk and delete temp file."""
    tensor = torch.load(path, map_location=device, weights_only=True)
    os.remove(path)
    return tensor


def _save_temp_video(images_np: np.ndarray, fps: int = 30) -> str:
    """Save numpy images to temporary video file for SimpleVO."""
    import tempfile
    temp_path = tempfile.mktemp(suffix=".mp4")
    h, w = images_np.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))
    for frame in images_np:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return temp_path


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

    # Class-level model cache
    _cached_model = None
    _cached_config_hash = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # ComfyUI IMAGE tensor (B, H, W, C)
                "masks": ("MASK",),  # SAM3 masks (B, H, W)
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
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("npz_path", "visualization", "info")
    FUNCTION = "run_inference"
    CATEGORY = "MotionCapture/GVHMR"

    @classmethod
    def _get_config_hash(cls, config: Dict) -> str:
        """Generate hash of config for cache invalidation."""
        import hashlib
        config_str = str(sorted(config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()

    @classmethod
    def _load_models(cls, config: Dict) -> Dict:
        """Load GVHMR models based on config."""
        from pathlib import Path
        from hmr4d.configs import register_store_gvhmr
        from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
        from hmr4d.utils.preproc import VitPoseExtractor, Extractor
        from hydra import initialize_config_module, compose
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        Log.info("[GVHMRInference] Loading GVHMR models...")

        gvhmr_path = Path(config["gvhmr_path"])

        # Initialize Hydra config for GVHMR
        Log.info("[GVHMRInference] Initializing GVHMR configuration...")
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
            register_store_gvhmr()
            cfg = compose(config_name="demo", overrides=["static_cam=True", "verbose=False"])

        # Check if rendering is available
        try:
            from hmr4d.utils.vis.renderer import PYTORCH3D_AVAILABLE
            if not PYTORCH3D_AVAILABLE:
                Log.warn("[GVHMRInference] PyTorch3D not installed - visualization rendering will be disabled")
        except Exception:
            pass

        # Load GVHMR model
        Log.info(f"[GVHMRInference] Loading GVHMR from {gvhmr_path}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Instantiate DemoPL with pipeline from config
        model_cfg_dict = OmegaConf.to_container(cfg.model, resolve=True)
        model_cfg = OmegaConf.create(model_cfg_dict)

        model_gvhmr = instantiate(model_cfg, _recursive_=False)
        model_gvhmr.load_pretrained_model(str(gvhmr_path))
        model_gvhmr.eval()
        model_gvhmr.to(device)

        # Initialize preprocessing components
        Log.info("[GVHMRInference] Initializing ViTPose extractor...")
        vitpose_extractor = VitPoseExtractor()

        Log.info("[GVHMRInference] Initializing feature extractor...")
        feature_extractor = Extractor()

        # Create model bundle
        model_bundle = {
            "gvhmr": model_gvhmr,
            "vitpose_extractor": vitpose_extractor,
            "feature_extractor": feature_extractor,
            "config": cfg,
            "device": device,
            "paths": config,
        }

        Log.info("[GVHMRInference] All models loaded successfully!")
        return model_bundle

    @classmethod
    def _get_or_load_model(cls, config: Dict) -> Dict:
        """Get cached model or load new one based on config."""
        config_hash = cls._get_config_hash(config)
        cache_model = config.get("cache_model", False)

        # Check if we have a valid cached model
        if cache_model and cls._cached_model is not None and cls._cached_config_hash == config_hash:
            Log.info("[GVHMRInference] Using cached model")
            return cls._cached_model

        # Load fresh model
        model_bundle = cls._load_models(config)

        # Cache if requested
        if cache_model:
            cls._cached_model = model_bundle
            cls._cached_config_hash = config_hash
            Log.info("[GVHMRInference] Model cached for future runs")
        else:
            # Clear any existing cache
            cls._cached_model = None
            cls._cached_config_hash = None

        return model_bundle

    def prepare_data_from_masks(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        model_bundle: Dict,
        static_camera: bool,
        focal_length_mm: int,
        bbox_scale: float,
        vo_method: str,
        vo_scale: float,
        vo_step: int,
        intrinsics: torch.Tensor = None,
        dpvo_dir: str = "",
    ) -> Dict:
        """
        Prepare data dictionary for GVHMR inference from images and masks.
        """
        # Validate inputs
        validate_images(images)
        validate_masks(masks)

        device = model_bundle["device"]
        batch_size, height, width, channels = images.shape
        Log.info(f"[GVHMRInference] images: {images.shape} dtype={images.dtype}, masks: {masks.shape} dtype={masks.dtype}")
        Log.info(f"[GVHMRInference] batch_size={batch_size}, resolution={width}x{height}, static_camera={static_camera}")

        if masks.shape[0] != batch_size:
            Log.warn(f"[GVHMRInference] Mask count ({masks.shape[0]}) != frame count ({batch_size})! Trimming masks to match.")
            masks = masks[:batch_size]

        # Convert images to numpy for processing
        images_np = (images.cpu().numpy() * 255).astype(np.uint8)

        # Extract bounding boxes from masks
        Log.info("[GVHMRInference] Extracting bounding boxes from SAM3 masks...")
        bboxes_xywh = extract_bboxes_from_masks(masks)

        # Expand bounding boxes
        bboxes_xywh = [
            expand_bbox(bbox, scale=bbox_scale, img_width=width, img_height=height)
            for bbox in bboxes_xywh
        ]

        # Convert to xyxy format for processing
        bboxes_xyxy = torch.tensor([bbox_to_xyxy(bbox) for bbox in bboxes_xywh], dtype=torch.float32)

        # Get bbx_xys format (used by GVHMR)
        bbx_xys = get_bbx_xys_from_xyxy(bboxes_xyxy, base_enlarge=1.0).float()  # Already expanded above
        Log.info(f"[GVHMRInference] bboxes_xyxy: {bboxes_xyxy.shape}, bbx_xys: {bbx_xys.shape}")

        # Extract ViTPose 2D keypoints
        Log.info("[GVHMRInference] Extracting 2D pose with ViTPose...")
        vitpose_extractor = model_bundle["vitpose_extractor"]

        # Use get_batch to preprocess images for extractors
        from hmr4d.utils.preproc.vitfeat_extractor import get_batch

        # Prepare images in the right format for get_batch (expects (F, H, W, 3) RGB numpy array)
        imgs_tensor, bbx_xys_processed = get_batch(images_np, bbx_xys, img_ds=1.0, path_type="np")

        # Extract 2D keypoints with ViTPose
        kp2d = vitpose_extractor.extract(imgs_tensor, bbx_xys_processed, img_ds=1.0)
        kp2d = kp2d.cpu()  # Move to CPU immediately
        _clear_cuda_memory()
        Log.info(f"[GVHMRInference] ViTPose complete: kp2d={kp2d.shape} dtype={kp2d.dtype}")

        # Extract ViT features
        Log.info("[GVHMRInference] Extracting image features...")
        feature_extractor = model_bundle["feature_extractor"]
        f_imgseq = feature_extractor.extract_video_features(imgs_tensor, bbx_xys_processed, img_ds=1.0)

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
                        R_w2c, t_w2c = _run_dpvo(images_np, K_fullimg, dpvo_dir)
                        _clear_cuda_memory()
                        Log.info(f"[GVHMRInference] DPVO complete, camera translation range: {t_w2c.abs().max().item():.3f}m")

                if vo_method == "simple_vo":
                    Log.info(f"[GVHMRInference] Running SimpleVO for moving camera (scale={vo_scale}, step={vo_step})...")

                    # Save frames to temp video (SimpleVO requires video path)
                    temp_video = _save_temp_video(images_np)
                    Log.info(f"[GVHMRInference] Temp video saved: {temp_video} ({len(images_np)} frames)")

                    # Run SimpleVO
                    f_mm = focal_length_mm if focal_length_mm > 0 else 24
                    Log.info(f"[GVHMRInference] SimpleVO params: f_mm={f_mm}, scale={vo_scale}, step={vo_step}, method=sift")
                    vo = SimpleVO(temp_video, scale=vo_scale, step=vo_step, method="sift", f_mm=f_mm)
                    T_w2c_list = vo.compute()
                    Log.info(f"[GVHMRInference] SimpleVO returned {len(T_w2c_list)} poses (expected {batch_size})")

                    # Clean up temp file
                    os.remove(temp_video)

                    # Extract rotation matrices (upper-left 3x3 of each 4x4 transform)
                    R_w2c = torch.tensor(np.stack([T[:3, :3] for T in T_w2c_list]), dtype=torch.float32)
                    # Extract translation vectors (right column of each 4x4 transform)
                    t_w2c = torch.tensor(np.stack([T[:3, 3] for T in T_w2c_list]), dtype=torch.float32)

                    Log.info(f"[GVHMRInference] R_w2c: {R_w2c.shape}, t_w2c: {t_w2c.shape}")
                    Log.info(f"[GVHMRInference] Translation range: min={t_w2c.min().item():.4f}, max={t_w2c.max().item():.4f}")
                    _clear_cuda_memory()

            except Exception as e:
                Log.error(f"[GVHMRInference] Visual odometry failed: {e}")
                import traceback
                traceback.print_exc()
                Log.warn("[GVHMRInference] Falling back to static camera")
                R_w2c = torch.eye(3).repeat(batch_size, 1, 1)
                t_w2c = None

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
        }

        return data, camera_data

    def run_inference(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        config: Dict,
        moving_camera: bool = False,
        focal_length_mm: int = 0,
        bbox_scale: float = 1.2,
        vo_method: str = "simple_vo",
        vo_scale: float = 0.5,
        vo_step: int = 8,
        intrinsics: torch.Tensor = None,
    ):
        """
        Run GVHMR inference on images with SAM3 masks.
        """
        try:
            static_camera = not moving_camera
            Log.info("=" * 60)
            Log.info("[GVHMRInference] Starting GVHMR inference with parameters:")
            Log.info(f"  images:          {images.shape} dtype={images.dtype}")
            Log.info(f"  masks:           {masks.shape} dtype={masks.dtype}")
            Log.info(f"  moving_camera:   {moving_camera}")
            Log.info(f"  focal_length_mm: {focal_length_mm}")
            Log.info(f"  bbox_scale:      {bbox_scale}")
            Log.info(f"  vo_method:       {vo_method}")
            Log.info(f"  vo_scale:        {vo_scale}")
            Log.info(f"  vo_step:         {vo_step}")
            Log.info(f"  intrinsics:      {intrinsics.shape if intrinsics is not None else 'None'}")
            Log.info(f"  config keys:     {list(config.keys())}")
            Log.info(f"  cache_model:     {config.get('cache_model', False)}")
            Log.info(f"  dpvo_dir:        {config.get('dpvo_dir', '')}")
            Log.info("=" * 60)

            # Load models based on config
            model = self._get_or_load_model(config)

            # Get DPVO directory from config (just a path string)
            dpvo_dir = config.get("dpvo_dir", "")

            # Prepare data
            data, camera_data = self.prepare_data_from_masks(
                images, masks, model, static_camera, focal_length_mm, bbox_scale, vo_method, vo_scale, vo_step, intrinsics, dpvo_dir
            )

            # Reload features from disk if they were offloaded
            if data.get("f_imgseq_path") is not None:
                Log.info("[GVHMRInference] Reloading features from disk...")
                data["f_imgseq"] = _load_tensor_from_disk(data["f_imgseq_path"], device="cpu")
                data["f_imgseq_path"] = None
                Log.info("[GVHMRInference] Features reloaded from disk")

            # Run GVHMR inference
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

            # Extract SMPL parameters
            smpl_params = {
                "global": pred["smpl_params_global"],
                "incam": pred["smpl_params_incam"],
                "K_fullimg": camera_data["K_fullimg"],
                "R_w2c": camera_data["R_w2c"],
                "t_w2c": camera_data["t_w2c"],
            }

            # Save SMPL params to NPZ file (avoids tensor serialization issues)
            import time
            output_dir = Path(folder_paths.get_output_directory())
            output_dir.mkdir(parents=True, exist_ok=True)
            npz_filename = f"smpl_params_{int(time.time())}.npz"
            npz_path = output_dir / npz_filename

            global_params = pred["smpl_params_global"]
            np.savez(
                str(npz_path),
                body_pose=global_params['body_pose'].cpu().numpy(),
                betas=global_params['betas'].cpu().numpy(),
                global_orient=global_params['global_orient'].cpu().numpy(),
                transl=global_params['transl'].cpu().numpy(),
            )
            Log.info(f"[GVHMRInference] Saved SMPL params to: {npz_path}")

            # Create visualization
            Log.info("[GVHMRInference] Rendering visualization...")
            viz_frames = self.render_visualization(images, smpl_params, model)

            # Create info string
            num_frames = images.shape[0]
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

            # Clear model from memory if not caching
            if not config.get("cache_model", False):
                Log.info("[GVHMRInference] Clearing model from memory (cache_model=False)")
                del model
                _clear_cuda_memory()

            return (str(npz_path), viz_frames, info)

        except Exception as e:
            error_msg = f"GVHMR Inference failed: {str(e)}"
            Log.error(error_msg)
            import traceback
            traceback.print_exc()
            # Return empty results on error
            return ("", images, error_msg)

    def render_visualization(
        self,
        images: torch.Tensor,
        smpl_params: Dict,
        model: Dict,
    ) -> torch.Tensor:
        """
        Render SMPL mesh overlay on input images.
        """
        try:
            # Check if rendering is available
            from hmr4d.utils.vis.renderer import PYTORCH3D_AVAILABLE
            if not PYTORCH3D_AVAILABLE:
                Log.warn("[GVHMRInference] PyTorch3D not available - skipping visualization rendering")
                Log.info("[GVHMRInference] Returning original images (SMPL parameters were extracted successfully)")
                return images

            device = model["device"]
            batch_size, height, width, channels = images.shape

            # Initialize SMPL model
            smplx = make_smplx("supermotion").to(device)
            smplx2smpl = torch.load(
                str(Path(__file__).parent / "vendor" / "hmr4d" / "utils" / "body_model" / "smplx2smpl_sparse.pt")
            ).to(device)

            # Get SMPL vertices
            smplx_out = smplx(**to_cuda(smpl_params["incam"]))
            pred_verts = torch.stack([torch.matmul(smplx2smpl, v) for v in smplx_out.vertices])

            # Initialize renderer
            from hmr4d.utils.vis.renderer import Renderer
            faces_smpl = make_smplx("smpl").faces
            K = smpl_params["K_fullimg"][0]
            renderer = Renderer(width, height, device=device, faces=faces_smpl, K=K)

            # Render each frame
            rendered_frames = []
            images_np = (images.cpu().numpy() * 255).astype(np.uint8)

            for i in range(batch_size):
                img_rendered = renderer.render_mesh(
                    pred_verts[i].to(device),
                    images_np[i],
                    [0.8, 0.8, 0.8]  # Mesh color
                )
                rendered_frames.append(img_rendered)

            # Convert back to torch tensor
            rendered_tensor = torch.from_numpy(np.stack(rendered_frames)).float() / 255.0

            return rendered_tensor

        except Exception as e:
            Log.warn(f"[GVHMRInference] Visualization rendering failed: {e}")
            Log.info("[GVHMRInference] Returning original images (SMPL parameters were extracted successfully)")
            # Return original images if rendering fails
            return images


# Node registration
NODE_CLASS_MAPPINGS = {
    "GVHMRInference": GVHMRInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GVHMRInference": "GVHMR Inference",
}
