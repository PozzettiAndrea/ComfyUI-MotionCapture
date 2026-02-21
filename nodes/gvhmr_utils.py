"""
Utility functions for ComfyUI-MotionCapture nodes
"""

import numpy as np
import cv2
from typing import List


def extract_bbox_from_numpy_mask(mask_uint8: np.ndarray) -> List[int]:
    """
    Extract bounding box from a single grayscale uint8 mask.

    Args:
        mask_uint8: Grayscale mask array (H, W) with values 0-255

    Returns:
        Bounding box in [x, y, w, h] format
    """
    if len(mask_uint8.shape) == 3:
        mask_uint8 = mask_uint8[:, :, 0]
    _, mask_binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return [x, y, w, h]
    else:
        h, w = mask_uint8.shape[:2]
        return [0, 0, w, h]


def bbox_to_xyxy(bbox: List[int]) -> List[int]:
    """
    Convert bbox from [x, y, w, h] to [x1, y1, x2, y2] format.
    """
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def expand_bbox(bbox: List[int], scale: float = 1.2,
                img_width: int = None, img_height: int = None) -> List[int]:
    """
    Expand bounding box by a scale factor.

    Args:
        bbox: Bounding box in [x, y, w, h] format
        scale: Scale factor (1.2 = 20% expansion)
        img_width: Image width for clamping (optional)
        img_height: Image height for clamping (optional)

    Returns:
        Expanded bounding box in [x, y, w, h] format
    """
    x, y, w, h = bbox

    cx = x + w / 2
    cy = y + h / 2

    new_w = w * scale
    new_h = h * scale

    new_x = cx - new_w / 2
    new_y = cy - new_h / 2

    if img_width is not None and img_height is not None:
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_w = min(new_w, img_width - new_x)
        new_h = min(new_h, img_height - new_y)

    return [int(new_x), int(new_y), int(new_w), int(new_h)]
