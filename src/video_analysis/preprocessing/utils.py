"""
Utility Functions for Video Preprocessing Pipeline

This module provides helper functions for common operations.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_angle_between_points(
    point1: np.ndarray,
    point2: np.ndarray
) -> float:
    """
    Calculate angle between two points in degrees.
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Angle in degrees
    """
    dy = point2[1] - point1[1]
    dx = point2[0] - point1[0]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle


def calculate_distance(
    point1: np.ndarray,
    point2: np.ndarray
) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point
        point2: Second point
        
    Returns:
        Distance
    """
    return np.linalg.norm(point2 - point1)


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    margin: float,
    frame_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Expand bounding box by margin percentage.
    
    Args:
        bbox: Bounding box (x, y, w, h)
        margin: Margin as fraction of bbox size
        frame_shape: Frame shape (height, width)
        
    Returns:
        Expanded bounding box
    """
    x, y, w, h = bbox
    frame_h, frame_w = frame_shape
    
    margin_w = int(w * margin)
    margin_h = int(h * margin)
    
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(frame_w, x + w + margin_w)
    y2 = min(frame_h, y + h + margin_h)
    
    return (x1, y1, x2 - x1, y2 - y1)


def ensure_bbox_in_bounds(
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Ensure bounding box is within frame bounds.
    
    Args:
        bbox: Bounding box (x, y, w, h)
        frame_shape: Frame shape (height, width)
        
    Returns:
        Clipped bounding box
    """
    x, y, w, h = bbox
    frame_h, frame_w = frame_shape
    
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)
    
    return (x, y, w, h)


def blend_images(
    img1: np.ndarray,
    img2: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Blend two images with alpha blending.
    
    Args:
        img1: First image
        img2: Second image
        alpha: Blending factor (0.0-1.0)
        
    Returns:
        Blended image
    """
    return cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)


def normalize_landmarks(
    landmarks: np.ndarray,
    frame_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Normalize landmarks to [0, 1] range.
    
    Args:
        landmarks: Landmarks array (N, 2)
        frame_shape: Frame shape (height, width)
        
    Returns:
        Normalized landmarks
    """
    frame_h, frame_w = frame_shape
    normalized = landmarks.copy().astype(np.float32)
    normalized[:, 0] /= frame_w
    normalized[:, 1] /= frame_h
    return normalized


def denormalize_landmarks(
    landmarks: np.ndarray,
    frame_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Denormalize landmarks from [0, 1] range to pixel coordinates.
    
    Args:
        landmarks: Normalized landmarks array (N, 2)
        frame_shape: Frame shape (height, width)
        
    Returns:
        Denormalized landmarks
    """
    frame_h, frame_w = frame_shape
    denormalized = landmarks.copy().astype(np.float32)
    denormalized[:, 0] *= frame_w
    denormalized[:, 1] *= frame_h
    return denormalized


def calculate_iou(
    bbox1: Tuple[int, int, int, int],
    bbox2: Tuple[int, int, int, int]
) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box (x, y, w, h)
        bbox2: Second bounding box (x, y, w, h)
        
    Returns:
        IoU score (0.0-1.0)
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def draw_landmarks(
    frame: np.ndarray,
    landmarks: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 2
) -> np.ndarray:
    """
    Draw landmarks on frame for visualization.
    
    Args:
        frame: Input frame
        landmarks: Landmarks array (N, 2)
        color: Point color (B, G, R)
        radius: Point radius
        
    Returns:
        Frame with landmarks drawn
    """
    result = frame.copy()
    
    for point in landmarks:
        x, y = int(point[0]), int(point[1])
        cv2.circle(result, (x, y), radius, color, -1)
    
    return result


def draw_bbox(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None
) -> np.ndarray:
    """
    Draw bounding box on frame.
    
    Args:
        frame: Input frame
        bbox: Bounding box (x, y, w, h)
        color: Box color (B, G, R)
        thickness: Line thickness
        label: Optional label text
        
    Returns:
        Frame with bounding box drawn
    """
    result = frame.copy()
    x, y, w, h = bbox
    
    # Draw rectangle
    cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    
    # Draw label if provided
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # Draw background rectangle
        cv2.rectangle(
            result,
            (x, y - text_h - baseline - 5),
            (x + text_w, y),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            result,
            label,
            (x, y - baseline - 5),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    
    return result


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def validate_frame(frame: np.ndarray) -> bool:
    """
    Validate that frame is valid for processing.
    
    Args:
        frame: Input frame
        
    Returns:
        True if frame is valid
    """
    if frame is None:
        return False
    
    if not isinstance(frame, np.ndarray):
        return False
    
    if frame.size == 0:
        return False
    
    if len(frame.shape) not in [2, 3]:
        return False
    
    if len(frame.shape) == 3 and frame.shape[2] not in [1, 3, 4]:
        return False
    
    return True


def validate_landmarks(landmarks: np.ndarray, min_points: int = 5) -> bool:
    """
    Validate that landmarks are valid.
    
    Args:
        landmarks: Landmarks array
        min_points: Minimum number of points required
        
    Returns:
        True if landmarks are valid
    """
    if landmarks is None:
        return False
    
    if not isinstance(landmarks, np.ndarray):
        return False
    
    if landmarks.size == 0:
        return False
    
    if len(landmarks.shape) != 2:
        return False
    
    if landmarks.shape[0] < min_points:
        return False
    
    if landmarks.shape[1] < 2:
        return False
    
    return True
