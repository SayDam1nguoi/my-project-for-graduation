"""
Face Stabilizer Module

This module provides face stabilization functionality to reduce jitter and
maintain consistent face position across frames.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class FaceStabilizer:
    """
    Stabilizes face position across video frames.
    
    Uses landmark-based tracking to smooth face position and reduce jitter.
    Applies affine transformation to maintain consistent face position.
    """
    
    def __init__(
        self,
        window_size: int = 5,
        method: str = "affine",
        smoothing_factor: float = 0.7
    ):
        """
        Initialize FaceStabilizer.
        
        Args:
            window_size: Number of frames to use for position smoothing
            method: Transformation method ('affine' or 'similarity')
            smoothing_factor: Smoothing strength (0.0-1.0, higher = more smoothing)
        """
        self.window_size = window_size
        self.method = method
        self.smoothing_factor = smoothing_factor
        
        # History of face positions
        self.position_history: deque = deque(maxlen=window_size)
        self.landmark_history: deque = deque(maxlen=window_size)
        
        # Target position (center of frame)
        self.target_position: Optional[np.ndarray] = None
        
        logger.info(f"FaceStabilizer initialized: window_size={window_size}, "
                   f"method={method}, smoothing_factor={smoothing_factor}")
    
    def stabilize(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Stabilize face in frame using landmarks.
        
        Args:
            frame: Input frame
            landmarks: Facial landmarks array (shape: [N, 2] or [N, 3])
            
        Returns:
            Stabilized frame
        """
        # Ensure landmarks are 2D
        if landmarks.shape[1] > 2:
            landmarks = landmarks[:, :2]
        
        # Extract key points for stabilization
        key_points = self._extract_key_points(landmarks)
        
        # Calculate current face position
        current_position = np.mean(key_points, axis=0)
        
        # Update history
        self.position_history.append(current_position)
        self.landmark_history.append(key_points)
        
        # Calculate target position (smoothed)
        if len(self.position_history) < 2:
            # Not enough history, return original frame
            return frame
        
        target_position = self._calculate_smoothed_position()
        
        # Calculate transformation matrix
        transform_matrix = self._calculate_transform(key_points, target_position)
        
        # Apply transformation
        h, w = frame.shape[:2]
        stabilized = cv2.warpAffine(
            frame,
            transform_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return stabilized
    
    def _extract_key_points(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract key points for stabilization.
        
        Uses eye centers and nose tip for robust tracking.
        
        Args:
            landmarks: Full landmark array
            
        Returns:
            Key points array
        """
        if len(landmarks) >= 468:  # MediaPipe format
            # Key points: eyes and nose
            key_indices = [
                33,   # Left eye outer corner
                133,  # Left eye inner corner
                362,  # Right eye inner corner
                263,  # Right eye outer corner
                1,    # Nose tip
            ]
            key_points = landmarks[key_indices]
        elif len(landmarks) >= 68:  # dlib format
            # Key points: eyes and nose
            key_indices = [
                36,  # Left eye outer corner
                39,  # Left eye inner corner
                42,  # Right eye inner corner
                45,  # Right eye outer corner
                30,  # Nose tip
            ]
            key_points = landmarks[key_indices]
        else:
            # Use all available landmarks
            key_points = landmarks
        
        return key_points
    
    def _calculate_smoothed_position(self) -> np.ndarray:
        """
        Calculate smoothed target position from history.
        
        Uses weighted average with exponential decay.
        
        Returns:
            Smoothed position
        """
        if len(self.position_history) == 0:
            return np.array([0, 0])
        
        # Calculate weighted average
        positions = np.array(list(self.position_history))
        
        # Apply exponential weights (more recent = higher weight)
        weights = np.exp(np.linspace(-2, 0, len(positions)))
        weights = weights / weights.sum()
        
        # Apply smoothing factor
        weights = weights * self.smoothing_factor + (1 - self.smoothing_factor) / len(weights)
        weights = weights / weights.sum()
        
        smoothed = np.average(positions, axis=0, weights=weights)
        
        return smoothed
    
    def _calculate_transform(
        self,
        source_points: np.ndarray,
        target_position: np.ndarray
    ) -> np.ndarray:
        """
        Calculate transformation matrix to move face to target position.
        
        Args:
            source_points: Current face key points
            target_position: Target center position
            
        Returns:
            2x3 affine transformation matrix
        """
        # Calculate current center
        current_center = np.mean(source_points, axis=0)
        
        # Calculate translation
        translation = target_position - current_center
        
        if self.method == "affine":
            # Simple translation matrix
            transform_matrix = np.array([
                [1, 0, translation[0]],
                [0, 1, translation[1]]
            ], dtype=np.float32)
        
        elif self.method == "similarity":
            # Similarity transform (translation + rotation + scale)
            # Use previous frame's key points if available
            if len(self.landmark_history) >= 2:
                prev_points = self.landmark_history[-2]
                
                # Estimate similarity transform
                transform_matrix = cv2.estimateAffinePartial2D(
                    source_points,
                    prev_points + translation,
                    method=cv2.RANSAC
                )[0]
                
                if transform_matrix is None:
                    # Fallback to simple translation
                    transform_matrix = np.array([
                        [1, 0, translation[0]],
                        [0, 1, translation[1]]
                    ], dtype=np.float32)
            else:
                # Not enough history, use translation only
                transform_matrix = np.array([
                    [1, 0, translation[0]],
                    [0, 1, translation[1]]
                ], dtype=np.float32)
        else:
            raise ValueError(f"Unknown stabilization method: {self.method}")
        
        return transform_matrix
    
    def reset(self):
        """Reset stabilization history."""
        self.position_history.clear()
        self.landmark_history.clear()
        self.target_position = None
        logger.debug("FaceStabilizer reset")
