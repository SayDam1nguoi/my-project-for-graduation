"""
Frame Quality Filter Module

This module provides frame quality filtering functionality including blur detection,
face confidence checking, pose angle calculation, and occlusion detection.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FrameQualityFilter:
    """
    Filters out low-quality frames based on multiple criteria.
    
    Checks:
    - Blur detection using Laplacian variance
    - Face detection confidence
    - Face pose angles (yaw, pitch, roll)
    - Face occlusion
    """
    
    def __init__(
        self,
        blur_threshold: float = 100.0,
        confidence_threshold: float = 0.9,
        max_pose_angle: float = 30.0,
        check_occlusion: bool = True,
        min_face_size: Tuple[int, int] = (80, 80),
        min_fps_after_filter: float = 10.0
    ):
        """
        Initialize FrameQualityFilter.
        
        Args:
            blur_threshold: Minimum Laplacian variance (lower = more blurry)
            confidence_threshold: Minimum face detection confidence
            max_pose_angle: Maximum allowed pose angle in degrees
            check_occlusion: Whether to check for face occlusion
            min_face_size: Minimum face size (width, height)
            min_fps_after_filter: Minimum FPS to maintain after filtering
        """
        self.blur_threshold = blur_threshold
        self.confidence_threshold = confidence_threshold
        self.max_pose_angle = max_pose_angle
        self.check_occlusion = check_occlusion
        self.min_face_size = min_face_size
        self.min_fps_after_filter = min_fps_after_filter
        
        # Statistics
        self.total_frames = 0
        self.filtered_frames = 0
        self.filter_reasons: Dict[str, int] = {}
        
        logger.info(f"FrameQualityFilter initialized: blur_threshold={blur_threshold}, "
                   f"confidence_threshold={confidence_threshold}")
    
    def is_frame_valid(
        self,
        frame: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        face_confidence: Optional[float] = None,
        face_landmarks: Optional[np.ndarray] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if frame meets quality criteria.
        
        Args:
            frame: Input frame
            face_bbox: Face bounding box (x, y, w, h)
            face_confidence: Face detection confidence
            face_landmarks: Facial landmarks array
            
        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        self.total_frames += 1
        
        # Check face size
        if face_bbox is not None:
            _, _, w, h = face_bbox
            if w < self.min_face_size[0] or h < self.min_face_size[1]:
                self._record_rejection("face_too_small")
                return False, "face_too_small"
        
        # Check face confidence
        if face_confidence is not None:
            if face_confidence < self.confidence_threshold:
                self._record_rejection("low_confidence")
                return False, "low_confidence"
        
        # Check blur
        blur_score = self.calculate_blur_score(frame, face_bbox)
        if blur_score < self.blur_threshold:
            self._record_rejection("blurry")
            return False, f"blurry (score={blur_score:.1f})"
        
        # Check pose angles
        if face_landmarks is not None:
            yaw, pitch, roll = self.calculate_pose_angles(face_landmarks)
            if abs(yaw) > self.max_pose_angle or \
               abs(pitch) > self.max_pose_angle or \
               abs(roll) > self.max_pose_angle:
                self._record_rejection("extreme_pose")
                return False, f"extreme_pose (yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f})"
        
        # Check occlusion
        if self.check_occlusion and face_landmarks is not None:
            if self.detect_occlusion(frame, face_landmarks):
                self._record_rejection("occluded")
                return False, "occluded"
        
        return True, None
    
    def calculate_blur_score(
        self,
        frame: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> float:
        """
        Calculate blur score using Laplacian variance.
        
        Higher score = sharper image
        Lower score = more blurry image
        
        Args:
            frame: Input frame
            face_bbox: Optional face region to focus on
            
        Returns:
            Blur score (Laplacian variance)
        """
        # Extract face region if provided
        if face_bbox is not None:
            x, y, w, h = face_bbox
            # Ensure coordinates are within frame bounds
            h_frame, w_frame = frame.shape[:2]
            x = max(0, min(x, w_frame - 1))
            y = max(0, min(y, h_frame - 1))
            w = min(w, w_frame - x)
            h = min(h, h_frame - y)
            region = frame[y:y+h, x:x+w]
        else:
            region = frame
        
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return variance
    
    def calculate_pose_angles(
        self,
        landmarks: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate face pose angles (yaw, pitch, roll) from landmarks.
        
        Uses simplified 2D landmark-based estimation.
        
        Args:
            landmarks: Facial landmarks array (shape: [N, 2] or [N, 3])
            
        Returns:
            Tuple of (yaw, pitch, roll) in degrees
        """
        # Ensure landmarks are 2D
        if landmarks.shape[1] > 2:
            landmarks = landmarks[:, :2]
        
        # For MediaPipe 468 landmarks or dlib 68 landmarks
        # We'll use a simplified approach based on key points
        
        if len(landmarks) >= 468:  # MediaPipe format
            # Key points for MediaPipe
            left_eye = landmarks[33]  # Left eye outer corner
            right_eye = landmarks[263]  # Right eye outer corner
            nose_tip = landmarks[1]  # Nose tip
            left_mouth = landmarks[61]  # Left mouth corner
            right_mouth = landmarks[291]  # Right mouth corner
        elif len(landmarks) >= 68:  # dlib format
            # Key points for dlib
            left_eye = landmarks[36]  # Left eye outer corner
            right_eye = landmarks[45]  # Right eye outer corner
            nose_tip = landmarks[30]  # Nose tip
            left_mouth = landmarks[48]  # Left mouth corner
            right_mouth = landmarks[54]  # Right mouth corner
        else:
            # Not enough landmarks, return zero angles
            return 0.0, 0.0, 0.0
        
        # Calculate roll (rotation around z-axis)
        eye_center = (left_eye + right_eye) / 2
        d_eye = right_eye - left_eye
        roll = np.degrees(np.arctan2(d_eye[1], d_eye[0]))
        
        # Calculate yaw (rotation around y-axis)
        # Based on horizontal asymmetry
        face_width = np.linalg.norm(right_eye - left_eye)
        nose_to_left = np.linalg.norm(nose_tip - left_eye)
        nose_to_right = np.linalg.norm(nose_tip - right_eye)
        yaw_ratio = (nose_to_right - nose_to_left) / face_width
        yaw = yaw_ratio * 90  # Scale to degrees
        
        # Calculate pitch (rotation around x-axis)
        # Based on vertical position of nose relative to eyes
        mouth_center = (left_mouth + right_mouth) / 2
        eye_to_mouth = np.linalg.norm(mouth_center - eye_center)
        nose_to_eye = np.linalg.norm(nose_tip - eye_center)
        pitch_ratio = (nose_to_eye / eye_to_mouth) - 0.5
        pitch = pitch_ratio * 60  # Scale to degrees
        
        return yaw, pitch, roll
    
    def detect_occlusion(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray
    ) -> bool:
        """
        Detect if face is occluded (covered by objects).
        
        Simple heuristic: check if key facial landmarks are visible
        by analyzing local image statistics.
        
        Args:
            frame: Input frame
            landmarks: Facial landmarks array
            
        Returns:
            True if occlusion detected
        """
        # Ensure landmarks are 2D
        if landmarks.shape[1] > 2:
            landmarks = landmarks[:, :2]
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Check key regions for unusual patterns
        # This is a simplified check - in production, you might use a trained model
        
        # Define key regions to check (eyes, nose, mouth)
        if len(landmarks) >= 468:  # MediaPipe
            key_points = [
                landmarks[33],   # Left eye
                landmarks[263],  # Right eye
                landmarks[1],    # Nose
                landmarks[61],   # Left mouth
                landmarks[291],  # Right mouth
            ]
        elif len(landmarks) >= 68:  # dlib
            key_points = [
                landmarks[36],  # Left eye
                landmarks[45],  # Right eye
                landmarks[30],  # Nose
                landmarks[48],  # Left mouth
                landmarks[54],  # Right mouth
            ]
        else:
            return False
        
        # Check variance around key points
        # Low variance might indicate occlusion
        occlusion_count = 0
        for point in key_points:
            x, y = int(point[0]), int(point[1])
            
            # Extract small region around point
            region_size = 10
            y1 = max(0, y - region_size)
            y2 = min(gray.shape[0], y + region_size)
            x1 = max(0, x - region_size)
            x2 = min(gray.shape[1], x + region_size)
            
            if y2 > y1 and x2 > x1:
                region = gray[y1:y2, x1:x2]
                variance = np.var(region)
                
                # Very low variance might indicate occlusion
                if variance < 50:
                    occlusion_count += 1
        
        # If more than 2 key points have low variance, consider it occluded
        return occlusion_count > 2
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get filtering statistics.
        
        Returns:
            Dictionary with statistics
        """
        valid_frames = self.total_frames - self.filtered_frames
        filter_rate = self.filtered_frames / self.total_frames if self.total_frames > 0 else 0
        
        return {
            'total_frames': self.total_frames,
            'valid_frames': valid_frames,
            'filtered_frames': self.filtered_frames,
            'filter_rate': filter_rate,
            'filter_reasons': self.filter_reasons.copy()
        }
    
    def reset(self):
        """Reset statistics."""
        self.total_frames = 0
        self.filtered_frames = 0
        self.filter_reasons.clear()
    
    def _record_rejection(self, reason: str):
        """Record frame rejection for statistics."""
        self.filtered_frames += 1
        self.filter_reasons[reason] = self.filter_reasons.get(reason, 0) + 1
