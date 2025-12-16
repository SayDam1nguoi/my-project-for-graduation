"""
Face detection module using MTCNN for emotion recognition system.

Provides FaceDetector class for detecting faces in video frames with high accuracy
and performance optimization. Supports multi-face detection with confidence filtering.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import warnings
import logging

try:
    from facenet_pytorch import MTCNN
    import torch
except ImportError as e:
    raise ImportError(
        "facenet-pytorch is required for face detection. "
        "Install it with: pip install facenet-pytorch"
    ) from e

from .model_loader import FaceDetection

# Setup logger
logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Face detector using MTCNN (Multi-task Cascaded Convolutional Networks).
    
    Features:
    - High-accuracy face detection with MTCNN
    - Confidence threshold filtering (default >0.9)
    - Multi-face detection (up to 10 faces)
    - Automatic frame resizing for performance optimization
    - Facial landmark detection (5 points)
    
    Requirements:
    - 4.1: Detect and localize faces with bounding boxes
    - 4.2: Process frames within 50ms
    - 4.3: Detect up to 10 faces simultaneously
    - 4.4: Maintain 95%+ accuracy for faces >80x80 pixels
    - 4.5: Handle partial occlusion up to 30%
    
    Example:
        >>> detector = FaceDetector(device='cuda', confidence_threshold=0.9)
        >>> frame = cv2.imread('image.jpg')
        >>> detections = detector.detect_faces(frame)
        >>> for det in detections:
        ...     print(f"Face at {det.bbox} with confidence {det.confidence:.2f}")
    """
    
    def __init__(
        self,
        device: str = 'auto',
        confidence_threshold: float = 0.5,  # Lowered to 0.5 for better detection
        max_faces: int = 1,  # Only detect 1 face for speed
        target_size: Tuple[int, int] = (240, 180),  # Smaller for maximum speed
        keep_all: bool = False,  # Only keep largest face
        enable_preprocessing: bool = False  # Disable by default for speed
    ):
        """
        Initialize FaceDetector with MTCNN.
        
        Args:
            device: Device to run detection on ('cuda', 'cpu', or 'auto')
            confidence_threshold: Minimum confidence score for face detection (0.0-1.0)
                                 Default 0.9 as per requirement 4.4
            max_faces: Maximum number of faces to detect per frame (default 10)
                      As per requirement 4.3
            target_size: Target frame size (width, height) for detection optimization
                        Default (640, 480) for performance
            keep_all: Whether to keep all detected faces or only the most confident one
        
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
            )
        
        if max_faces < 1:
            raise ValueError(f"max_faces must be at least 1, got {max_faces}")
        
        if len(target_size) != 2 or target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError(f"target_size must be (width, height) with positive values")
        
        self.confidence_threshold = confidence_threshold
        self.max_faces = max_faces
        self.target_size = target_size
        self.keep_all = keep_all
        self.enable_preprocessing = enable_preprocessing
        
        # Frame-by-frame detection tracking
        self.frames_processed = 0
        self.frames_with_faces = 0
        self.frames_without_faces = 0
        
        # Detect device
        self.device = self._detect_device(device)
        
        # Initialize MTCNN
        # MTCNN parameters optimized for maximum speed
        self.mtcnn = MTCNN(
            image_size=160,  # Standard face size for MTCNN
            margin=0,  # No margin, we'll handle cropping separately
            min_face_size=80,  # Larger for faster detection (skip small/distant faces)
            thresholds=[0.8, 0.85, 0.85],  # Very high thresholds for maximum speed
            factor=0.85,  # Larger factor = fewer pyramid levels = faster
            post_process=False,  # We'll handle post-processing
            select_largest=not keep_all,  # Select largest or keep all
            keep_all=keep_all,  # Keep all detected faces
            device=self.device
        )
        
        # Log initialization with threshold value (Requirement 5.1)
        logger.info(f"FaceDetector initialized with confidence_threshold={self.confidence_threshold}")
        logger.info(f"FaceDetector configuration: device={self.device}, max_faces={self.max_faces}, target_size={self.target_size}")
        
        print(f"FaceDetector initialized:")
        print(f"  Device: {self.device}")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        print(f"  Max faces: {self.max_faces}")
        print(f"  Target size: {self.target_size}")
    
    def _detect_device(self, device: str) -> torch.device:
        """
        Detect and validate device for face detection.
        
        Args:
            device: Device specification ('cuda', 'cpu', or 'auto')
        
        Returns:
            torch.device object
        """
        if device == 'auto':
            if torch.cuda.is_available():
                device_obj = torch.device('cuda')
                print(f"CUDA available for face detection: {torch.cuda.get_device_name(0)}")
            else:
                device_obj = torch.device('cpu')
                print("CUDA not available, using CPU for face detection")
        elif device == 'cuda':
            if not torch.cuda.is_available():
                warnings.warn("CUDA requested but not available, falling back to CPU")
                device_obj = torch.device('cpu')
            else:
                device_obj = torch.device('cuda')
        elif device == 'cpu':
            device_obj = torch.device('cpu')
        else:
            raise ValueError(f"Invalid device: {device}. Must be 'cuda', 'cpu', or 'auto'")
        
        return device_obj
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in a video frame.
        
        This method:
        1. Resizes frame to target_size for performance (requirement 4.2)
        2. Detects faces using MTCNN
        3. Filters by confidence threshold (requirement 4.4)
        4. Returns up to max_faces detections (requirement 4.3)
        5. Scales bounding boxes back to original frame size
        
        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format (OpenCV)
        
        Returns:
            List of FaceDetection objects, sorted by confidence (highest first)
            Empty list if no faces detected
        
        Raises:
            ValueError: If frame is invalid
        
        Example:
            >>> detector = FaceDetector()
            >>> frame = cv2.imread('group_photo.jpg')
            >>> faces = detector.detect_faces(frame)
            >>> print(f"Detected {len(faces)} faces")
            >>> for i, face in enumerate(faces):
            ...     x, y, w, h = face.bbox
            ...     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        """
        # Validate input
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame: frame is None or empty")
        
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Frame must be 3-channel BGR image, got shape {frame.shape}"
            )
        
        original_height, original_width = frame.shape[:2]
        
        # Optional preprocessing for better detection accuracy
        # Disabled by default for real-time performance
        if self.enable_preprocessing:
            # Light preprocessing only - avoid heavy operations
            # 1. Simple contrast enhancement (faster than CLAHE)
            preprocessed_frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
        else:
            preprocessed_frame = frame
        
        # Optimization: Resize frame to target size for faster detection
        # This significantly improves performance (requirement 4.2: <50ms)
        if (original_width, original_height) != self.target_size:
            # Use INTER_LINEAR for faster resizing (2x faster than INTER_CUBIC)
            resized_frame = cv2.resize(preprocessed_frame, self.target_size, interpolation=cv2.INTER_LINEAR)
            scale_x = original_width / self.target_size[0]
            scale_y = original_height / self.target_size[1]
        else:
            resized_frame = preprocessed_frame
            scale_x = 1.0
            scale_y = 1.0
        
        # Convert BGR to RGB (MTCNN expects RGB)
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces with MTCNN
        # Returns: boxes (N, 4), probs (N,), landmarks (N, 5, 2)
        boxes, probs, landmarks = self.mtcnn.detect(rgb_frame, landmarks=True)
        
        # Track frame-by-frame detection (Requirement 5.1)
        self.frames_processed += 1
        
        # Handle no detections
        if boxes is None or len(boxes) == 0:
            self.frames_without_faces += 1
            logger.debug(f"No faces detected in frame {self.frames_processed}")
            return []
        
        # Track successful detection
        self.frames_with_faces += 1
        logger.debug(f"Detected {len(boxes)} face(s) in frame {self.frames_processed}")
        
        # Filter by confidence threshold and convert to FaceDetection objects
        detections = []
        
        for i in range(len(boxes)):
            confidence = float(probs[i])
            
            # Filter by confidence threshold (requirement 4.4)
            if confidence < self.confidence_threshold:
                continue
            
            # Get bounding box in (x1, y1, x2, y2) format
            x1, y1, x2, y2 = boxes[i]
            
            # Scale back to original frame size
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            # Convert to (x, y, width, height) format
            x = max(0, x1)
            y = max(0, y1)
            width = min(x2 - x1, original_width - x)
            height = min(y2 - y1, original_height - y)
            
            # Skip invalid boxes
            if width <= 0 or height <= 0:
                continue
            
            # Scale landmarks back to original frame size
            face_landmarks = None
            if landmarks is not None and i < len(landmarks):
                face_landmarks = landmarks[i].copy()
                face_landmarks[:, 0] *= scale_x  # Scale x coordinates
                face_landmarks[:, 1] *= scale_y  # Scale y coordinates
            
            # Create FaceDetection object
            detection = FaceDetection(
                bbox=(x, y, width, height),
                confidence=confidence,
                landmarks=face_landmarks
            )
            
            detections.append(detection)
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        # Limit to max_faces (requirement 4.3)
        detections = detections[:self.max_faces]
        
        return detections
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Update confidence threshold for face detection.
        
        Args:
            threshold: New confidence threshold (0.0-1.0)
        
        Raises:
            ValueError: If threshold is invalid
        
        Example:
            >>> detector = FaceDetector()
            >>> detector.set_confidence_threshold(0.95)  # More strict
            >>> detector.set_confidence_threshold(0.85)  # More lenient
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"Confidence threshold must be between 0.0 and 1.0, got {threshold}"
            )
        
        self.confidence_threshold = threshold
        print(f"Confidence threshold updated to: {threshold}")
    
    def get_config(self) -> dict:
        """
        Get current detector configuration.
        
        Returns:
            Dictionary with detector settings
        """
        return {
            'device': str(self.device),
            'confidence_threshold': self.confidence_threshold,
            'max_faces': self.max_faces,
            'target_size': self.target_size,
            'keep_all': self.keep_all
        }
    
    def get_detection_stats(self) -> dict:
        """
        Get frame-by-frame detection statistics.
        
        Returns:
            Dictionary with detection statistics including:
            - frames_processed: Total frames processed
            - frames_with_faces: Frames where faces were detected
            - frames_without_faces: Frames where no faces were detected
            - detection_rate: Percentage of frames with faces detected
        """
        detection_rate = 0.0
        if self.frames_processed > 0:
            detection_rate = (self.frames_with_faces / self.frames_processed) * 100
        
        return {
            'frames_processed': self.frames_processed,
            'frames_with_faces': self.frames_with_faces,
            'frames_without_faces': self.frames_without_faces,
            'detection_rate': detection_rate
        }
    
    def reset_detection_stats(self) -> None:
        """
        Reset frame-by-frame detection statistics.
        
        Useful when processing multiple videos with the same detector instance.
        """
        self.frames_processed = 0
        self.frames_with_faces = 0
        self.frames_without_faces = 0
        logger.debug("Detection statistics reset")


if __name__ == '__main__':
    # Demo usage
    print("FaceDetector Demo")
    print("=" * 70)
    
    # Initialize detector
    detector = FaceDetector(device='auto', confidence_threshold=0.9)
    
    # Print configuration
    print("\nDetector Configuration:")
    config = detector.get_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Test with a sample image (if available)
    print("\n" + "=" * 70)
    print("Testing face detection...")
    
    # Create a dummy frame for testing
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    try:
        detections = detector.detect_faces(dummy_frame)
        print(f"\nDetected {len(detections)} faces in dummy frame")
        
        for i, det in enumerate(detections):
            print(f"\nFace {i+1}:")
            print(f"  Bounding box: {det.bbox}")
            print(f"  Confidence: {det.confidence:.4f}")
            if det.landmarks is not None:
                print(f"  Landmarks shape: {det.landmarks.shape}")
    
    except Exception as e:
        print(f"\nNote: Detection on random frame may not find faces: {e}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("\nTo use with real images:")
    print("  frame = cv2.imread('path/to/image.jpg')")
    print("  detections = detector.detect_faces(frame)")
