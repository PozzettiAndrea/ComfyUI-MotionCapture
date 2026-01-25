"""
Utility functions for ComfyUI-MotionCapture nodes
"""

import torch
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional


def extract_bboxes_from_masks(masks: torch.Tensor) -> List[List[int]]:
    """
    Extract bounding boxes from SAM3 segmentation masks.

    Args:
        masks: Tensor of shape (batch, height, width) or (batch, height, width, 1)
               Values should be in range [0, 1]

    Returns:
        List of bounding boxes in format [x, y, w, h] for each frame
    """
    bboxes = []

    # Handle different mask shapes
    if len(masks.shape) == 4:
        masks = masks.squeeze(-1)  # Remove channel dimension if present

    # Process each mask individually (granularly to avoid system RAM spike)
    for i in range(len(masks)):
        mask = masks[i]
        # Convert to uint8 for OpenCV directly from torch
        mask_uint8 = (mask * 255).to(torch.uint8).cpu().numpy()

        # Find contours
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Get the largest contour (assuming it's the person)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            bboxes.append([x, y, w, h])
        else:
            # If no contour found, use full frame
            h, w = mask.shape
            bboxes.append([0, 0, w, h])

    return bboxes


def bbox_to_xyxy(bbox: List[int]) -> List[int]:
    """
    Convert bbox from [x, y, w, h] to [x1, y1, x2, y2] format.

    Args:
        bbox: Bounding box in [x, y, w, h] format

    Returns:
        Bounding box in [x1, y1, x2, y2] format
    """
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def expand_bbox(bbox: List[int], scale: float = 1.2,
                img_width: int = None, img_height: int = None) -> List[int]:
    """
    Expand bounding box by a scale factor (useful for ensuring full person is captured).

    Args:
        bbox: Bounding box in [x, y, w, h] format
        scale: Scale factor (1.2 = 20% expansion)
        img_width: Image width for clamping (optional)
        img_height: Image height for clamping (optional)

    Returns:
        Expanded bounding box in [x, y, w, h] format
    """
    x, y, w, h = bbox

    # Calculate center
    cx = x + w / 2
    cy = y + h / 2

    # Expand
    new_w = w * scale
    new_h = h * scale

    # Recalculate top-left
    new_x = cx - new_w / 2
    new_y = cy - new_h / 2

    # Clamp to image boundaries if provided
    if img_width is not None and img_height is not None:
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_w = min(new_w, img_width - new_x)
        new_h = min(new_h, img_height - new_y)

    return [int(new_x), int(new_y), int(new_w), int(new_h)]


def normalize_image_tensor(images: torch.Tensor) -> torch.Tensor:
    """
    Normalize image tensor to range [0, 1] if needed.

    Args:
        images: Tensor of shape (batch, height, width, channels)

    Returns:
        Normalized tensor
    """
    if images.max() > 1.0:
        images = images / 255.0
    return images


def crop_image_with_bbox(image: np.ndarray, bbox: List[int]) -> np.ndarray:
    """
    Crop image using bounding box.

    Args:
        image: Image array of shape (H, W, C)
        bbox: Bounding box in [x, y, w, h] format

    Returns:
        Cropped image
    """
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]


def resize_to_model_input(image: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Resize image to model input size while maintaining aspect ratio.

    Args:
        image: Image array
        target_size: Target (height, width)

    Returns:
        Resized image
    """
    return cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)


def validate_masks(masks: torch.Tensor) -> bool:
    """
    Validate that masks tensor has the expected format.

    Args:
        masks: Mask tensor

    Returns:
        True if valid, raises ValueError otherwise
    """
    if not isinstance(masks, torch.Tensor):
        raise ValueError(f"Masks must be a torch.Tensor, got {type(masks)}")

    if len(masks.shape) not in [3, 4]:
        raise ValueError(f"Masks must be 3D or 4D tensor, got shape {masks.shape}")

    if masks.max() > 1.0 or masks.min() < 0.0:
        raise ValueError(f"Masks must be in range [0, 1], got range [{masks.min()}, {masks.max()}]")

    return True


def validate_images(images: torch.Tensor) -> bool:
    """
    Validate that images tensor has the expected format.

    Args:
        images: Image tensor

    Returns:
        True if valid, raises ValueError otherwise
    """
    if not isinstance(images, torch.Tensor):
        raise ValueError(f"Images must be a torch.Tensor, got {type(images)}")

    if len(images.shape) != 4:
        raise ValueError(f"Images must be 4D tensor (batch, height, width, channels), got shape {images.shape}")

    return True


def create_tracking_data(bboxes: List[List[int]],
                        frame_width: int,
                        frame_height: int) -> Dict:
    """
    Create tracking data structure compatible with GVHMR preprocessing.

    Args:
        bboxes: List of bounding boxes [x, y, w, h]
        frame_width: Width of frames
        frame_height: Height of frames

    Returns:
        Dictionary with tracking information
    """
    tracking_data = {
        'bboxes': bboxes,
        'frame_width': frame_width,
        'frame_height': frame_height,
        'num_frames': len(bboxes),
        'person_id': 0  # Single person tracking
    }
    return tracking_data
