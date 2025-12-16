"""
Face Aligner Module

This module provides face alignment functionality including rotation correction,
face cropping, and standardization to match emotion classifier input format.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FaceAligner:
    """
    Aligns faces to canonical position.
    
    Performs:
    - Rotation correction to make eyes horizontal
    - Face cropping with standard proportions
    - Resizing to target size for emotion classifier
    - Perspective correction for extreme poses
    """
    
    def __init__(
        self,
        method: str = "landmarks",
        crop_margin: float = 0.2,
        target_size: Tuple[int, int] = (224, 224),
        correct_rotation: bool = True,
        correct_perspective: bool = True
    ):
        """
        Initialize FaceAligner.
        
        Args:
            method: Alignment method ('landmarks' or '3d_model')
            crop_margin: Margin around face as fraction of face size
            target_size: Target output size (width, height)
            correct_rotation: Enable rotation correction
            correct_perspective: Enable perspective correction
        """
        self.method = method
        self.crop_margin = crop_margin
        self.target_size = target_size
        self.correct_rotation = correct_rotation
        self.correct_perspective = correct_perspective
        
        logger.info(f"FaceAligner initialized: method={method}, "
                   f"target_size={target_size}, crop_margin={crop_margin}")
    
    def align_face(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Align face to canonical position.
        
        Args:
            frame: Input frame
            landmarks: Facial landmarks array (shape: [N, 2] or [N, 3])
            face_bbox: Optional face bounding box (x, y, w, h)
            
        Returns:
            Aligned face image
        """
        # Ensure landmarks are 2D
        if landmarks.shape[1] > 2:
            landmarks = landmarks[:, :2]
        
        # Extract eye positions
        left_eye, right_eye = self._get_eye_positions(landmarks)
        
        # Calculate rotation angle
        if self.correct_rotation:
            angle = self._calculate_rotation_angle(left_eye, right_eye)
        else:
            angle = 0.0
        
        # Calculate face center
        face_center = self._calculate_face_center(landmarks)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(
            tuple(face_center.astype(float)),
            angle,
            scale=1.0
        )
        
        # Rotate frame
        h, w = frame.shape[:2]
        rotated = cv2.warpAffine(
            frame,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # Rotate landmarks
        rotated_landmarks = self._rotate_landmarks(landmarks, rotation_matrix)
        
        # Crop face region
        cropped = self._crop_face(rotated, rotated_landmarks, face_bbox)
        
        # Resize to target size
        aligned = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_AREA)
        
        return aligned
    
    def _get_eye_positions(self, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract eye center positions from landmarks.
        
        Args:
            landmarks: Facial landmarks array
            
        Returns:
            Tuple of (left_eye_center, right_eye_center)
        """
        if len(landmarks) >= 468:  # MediaPipe format
            # Calculate eye centers from multiple points
            left_eye_indices = [33, 133, 160, 159, 158, 157, 173]
            right_eye_indices = [362, 263, 387, 386, 385, 384, 398]
            
            left_eye = np.mean(landmarks[left_eye_indices], axis=0)
            right_eye = np.mean(landmarks[right_eye_indices], axis=0)
        
        elif len(landmarks) >= 68:  # dlib format
            # Eye landmarks: 36-41 (left), 42-47 (right)
            left_eye = np.mean(landmarks[36:42], axis=0)
            right_eye = np.mean(landmarks[42:48], axis=0)
        
        else:
            # Fallback: use first two landmarks
            left_eye = landmarks[0]
            right_eye = landmarks[1] if len(landmarks) > 1 else landmarks[0]
        
        return left_eye, right_eye
    
    def _calculate_rotation_angle(
        self,
        left_eye: np.ndarray,
        right_eye: np.ndarray
    ) -> float:
        """
        Calculate rotation angle to make eyes horizontal.
        
        Args:
            left_eye: Left eye center position
            right_eye: Right eye center position
            
        Returns:
            Rotation angle in degrees
        """
        # Calculate angle between eyes
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        
        angle = np.degrees(np.arctan2(dy, dx))
        
        return angle
    
    def _calculate_face_center(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Calculate face center from landmarks.
        
        Args:
            landmarks: Facial landmarks array
            
        Returns:
            Face center position
        """
        # Use mean of all landmarks as center
        center = np.mean(landmarks, axis=0)
        return center
    
    def _rotate_landmarks(
        self,
        landmarks: np.ndarray,
        rotation_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Apply rotation matrix to landmarks.
        
        Args:
            landmarks: Original landmarks
            rotation_matrix: 2x3 rotation matrix
            
        Returns:
            Rotated landmarks
        """
        # Add homogeneous coordinate
        ones = np.ones((landmarks.shape[0], 1))
        landmarks_homo = np.hstack([landmarks, ones])
        
        # Apply rotation
        rotated = rotation_matrix @ landmarks_homo.T
        rotated = rotated.T
        
        return rotated
    
    def _crop_face(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Crop face region with margin.
        
        Args:
            frame: Input frame
            landmarks: Facial landmarks
            face_bbox: Optional face bounding box
            
        Returns:
            Cropped face image
        """
        h, w = frame.shape[:2]
        
        if face_bbox is not None:
            # Use provided bounding box
            x, y, fw, fh = face_bbox
        else:
            # Calculate bounding box from landmarks
            x_min = np.min(landmarks[:, 0])
            x_max = np.max(landmarks[:, 0])
            y_min = np.min(landmarks[:, 1])
            y_max = np.max(landmarks[:, 1])
            
            fw = x_max - x_min
            fh = y_max - y_min
            x = x_min
            y = y_min
        
        # Add margin
        margin_w = int(fw * self.crop_margin)
        margin_h = int(fh * self.crop_margin)
        
        x1 = max(0, int(x - margin_w))
        y1 = max(0, int(y - margin_h))
        x2 = min(w, int(x + fw + margin_w))
        y2 = min(h, int(y + fh + margin_h))
        
        # Crop
        cropped = frame[y1:y2, x1:x2]
        
        # Ensure minimum size
        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
            logger.warning("Cropped face too small, returning original frame")
            return frame
        
        return cropped
    
    def align_with_perspective(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Align face using perspective transformation.
        
        Useful for extreme pose angles.
        
        Args:
            frame: Input frame
            landmarks: Facial landmarks
            
        Returns:
            Aligned face image
        """
        # Define canonical face landmarks (frontal view)
        # These are normalized coordinates for a standard face
        canonical_landmarks = np.array([
            [0.3, 0.4],  # Left eye
            [0.7, 0.4],  # Right eye
            [0.5, 0.6],  # Nose
            [0.3, 0.8],  # Left mouth
            [0.7, 0.8],  # Right mouth
        ], dtype=np.float32)
        
        # Scale to target size
        canonical_landmarks[:, 0] *= self.target_size[0]
        canonical_landmarks[:, 1] *= self.target_size[1]
        
        # Extract corresponding landmarks from input
        if len(landmarks) >= 468:  # MediaPipe
            key_indices = [33, 263, 1, 61, 291]
        elif len(landmarks) >= 68:  # dlib
            key_indices = [36, 45, 30, 48, 54]
        else:
            # Not enough landmarks for perspective transform
            return self.align_face(frame, landmarks)
        
        source_landmarks = landmarks[key_indices].astype(np.float32)
        
        # Calculate perspective transform
        transform_matrix = cv2.getPerspectiveTransform(
            source_landmarks[:4],
            canonical_landmarks[:4]
        )
        
        # Apply transform
        aligned = cv2.warpPerspective(
            frame,
            transform_matrix,
            self.target_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return aligned
    
    def validate_alignment(
        self,
        aligned_face: np.ndarray
    ) -> bool:
        """
        Validate that aligned face meets quality criteria.
        
        Args:
            aligned_face: Aligned face image
            
        Returns:
            True if alignment is valid
        """
        # Check size
        if aligned_face.shape[:2] != self.target_size[::-1]:
            return False
        
        # Check if image is not blank
        if np.mean(aligned_face) < 10 or np.mean(aligned_face) > 245:
            return False
        
        # Check variance (should have some detail)
        if np.var(aligned_face) < 100:
            return False
        
        return True
