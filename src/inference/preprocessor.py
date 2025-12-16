"""
Face preprocessing module for emotion recognition inference.

Provides FacePreprocessor class for preparing detected faces for emotion
classification. Handles cropping, resizing, normalization, and tensor conversion.

Requirements: 5.1, 5.3, 5.4
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Union, Optional
import warnings

from .model_loader import FaceDetection


class FacePreprocessor:
    """
    Preprocess detected faces for emotion classification.
    
    Features:
    - Crop face from frame with configurable margin
    - Resize to target size (default 224x224)
    - Normalize using ImageNet statistics
    - Convert to PyTorch tensor
    - Batch preprocessing support
    
    Requirements:
    - 5.1: Prepare face images for emotion classification
    - 5.3: Provide confidence scores for predictions
    - 5.4: Process faces within 30ms
    
    Example:
        >>> preprocessor = FacePreprocessor(target_size=(224, 224), margin=0.2)
        >>> frame = cv2.imread('image.jpg')
        >>> detection = FaceDetection(bbox=(100, 100, 200, 200), confidence=0.95)
        >>> face_tensor = preprocessor.preprocess(frame, detection)
        >>> # face_tensor is ready for model inference
    """
    
    # ImageNet normalization statistics (same as training)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        margin: float = 0.2,
        normalize: bool = True,
        mean: Optional[list] = None,
        std: Optional[list] = None
    ):
        """
        Initialize FacePreprocessor.
        
        Args:
            target_size: Target face size (height, width) for model input
                        Default (224, 224) matches training
            margin: Margin to add around face bbox as fraction of bbox size
                   Default 0.2 (20% margin) to include context
            normalize: Whether to normalize pixel values
            mean: Normalization mean values (RGB). If None, uses ImageNet stats
            std: Normalization std values (RGB). If None, uses ImageNet stats
        
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if len(target_size) != 2 or target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError(
                f"target_size must be (height, width) with positive values, got {target_size}"
            )
        
        if not (0.0 <= margin <= 1.0):
            raise ValueError(f"margin must be between 0.0 and 1.0, got {margin}")
        
        self.target_size = target_size
        self.margin = margin
        self.normalize = normalize
        
        # Set normalization parameters
        self.mean = np.array(mean if mean is not None else self.IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(std if std is not None else self.IMAGENET_STD, dtype=np.float32)
        
        print(f"FacePreprocessor initialized:")
        print(f"  Target size: {self.target_size}")
        print(f"  Margin: {self.margin}")
        print(f"  Normalize: {self.normalize}")
        if self.normalize:
            print(f"  Mean: {self.mean}")
            print(f"  Std: {self.std}")
    
    def preprocess(
        self,
        frame: np.ndarray,
        detection: FaceDetection
    ) -> torch.Tensor:
        """
        Preprocess a single detected face for emotion classification.
        
        Pipeline:
        1. Crop face from frame with margin
        2. Resize to target size
        3. Convert BGR to RGB
        4. Normalize pixel values
        5. Convert to PyTorch tensor (C, H, W)
        
        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format (OpenCV)
            detection: FaceDetection object with bbox coordinates
        
        Returns:
            Preprocessed face as torch.Tensor of shape (C, H, W)
            Ready for model inference (add batch dimension before feeding to model)
        
        Raises:
            ValueError: If frame or detection is invalid
        
        Example:
            >>> preprocessor = FacePreprocessor()
            >>> frame = cv2.imread('image.jpg')
            >>> detection = FaceDetection(bbox=(100, 100, 200, 200), confidence=0.95)
            >>> face_tensor = preprocessor.preprocess(frame, detection)
            >>> print(face_tensor.shape)  # (3, 224, 224)
        """
        # Validate inputs
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame: frame is None or empty")
        
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Frame must be 3-channel BGR image, got shape {frame.shape}"
            )
        
        # Extract bbox coordinates
        x, y, width, height = detection.bbox
        
        # Calculate margin in pixels
        margin_x = int(width * self.margin)
        margin_y = int(height * self.margin)
        
        # Expand bbox with margin
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(frame.shape[1], x + width + margin_x)
        y2 = min(frame.shape[0], y + height + margin_y)
        
        # Crop face from frame
        face_crop = frame[y1:y2, x1:x2]
        
        # Validate crop
        if face_crop.size == 0:
            raise ValueError(
                f"Invalid face crop: bbox={detection.bbox}, "
                f"frame_shape={frame.shape}, crop_coords=({x1},{y1},{x2},{y2})"
            )
        
        # Resize to target size
        face_resized = cv2.resize(face_crop, self.target_size[::-1])  # OpenCV uses (width, height)
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to float and scale to [0, 1]
        face_float = face_rgb.astype(np.float32) / 255.0
        
        # Normalize if enabled
        if self.normalize:
            face_normalized = (face_float - self.mean) / self.std
        else:
            face_normalized = face_float
        
        # Convert to PyTorch tensor (H, W, C) -> (C, H, W)
        face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1)
        
        return face_tensor
    
    def preprocess_batch(
        self,
        frame: np.ndarray,
        detections: list[FaceDetection]
    ) -> torch.Tensor:
        """
        Preprocess multiple detected faces for batch inference.
        
        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format
            detections: List of FaceDetection objects
        
        Returns:
            Batch of preprocessed faces as torch.Tensor of shape (B, C, H, W)
            where B is the number of detections
        
        Example:
            >>> preprocessor = FacePreprocessor()
            >>> frame = cv2.imread('group_photo.jpg')
            >>> detections = detector.detect_faces(frame)
            >>> face_batch = preprocessor.preprocess_batch(frame, detections)
            >>> print(face_batch.shape)  # (num_faces, 3, 224, 224)
        """
        if not detections:
            # Return empty tensor with correct shape
            return torch.empty(0, 3, *self.target_size)
        
        # Preprocess each face
        face_tensors = []
        for detection in detections:
            try:
                face_tensor = self.preprocess(frame, detection)
                face_tensors.append(face_tensor)
            except ValueError as e:
                warnings.warn(f"Failed to preprocess face: {e}")
                continue
        
        if not face_tensors:
            # All preprocessing failed
            return torch.empty(0, 3, *self.target_size)
        
        # Stack into batch
        face_batch = torch.stack(face_tensors, dim=0)
        
        return face_batch
    
    def crop_face(
        self,
        frame: np.ndarray,
        detection: FaceDetection,
        return_rgb: bool = False
    ) -> np.ndarray:
        """
        Crop face from frame with margin (without normalization).
        
        Useful for visualization or saving face crops.
        
        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format
            detection: FaceDetection object with bbox coordinates
            return_rgb: If True, return RGB image; if False, return BGR
        
        Returns:
            Cropped and resized face as numpy array (H, W, C)
        
        Example:
            >>> preprocessor = FacePreprocessor()
            >>> frame = cv2.imread('image.jpg')
            >>> detection = FaceDetection(bbox=(100, 100, 200, 200), confidence=0.95)
            >>> face_crop = preprocessor.crop_face(frame, detection)
            >>> cv2.imwrite('face.jpg', face_crop)
        """
        # Extract bbox coordinates
        x, y, width, height = detection.bbox
        
        # Calculate margin in pixels
        margin_x = int(width * self.margin)
        margin_y = int(height * self.margin)
        
        # Expand bbox with margin
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(frame.shape[1], x + width + margin_x)
        y2 = min(frame.shape[0], y + height + margin_y)
        
        # Crop face from frame
        face_crop = frame[y1:y2, x1:x2]
        
        # Resize to target size
        face_resized = cv2.resize(face_crop, self.target_size[::-1])
        
        # Convert to RGB if requested
        if return_rgb:
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        return face_resized
    
    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Denormalize tensor back to image for visualization.
        
        Args:
            tensor: Normalized tensor of shape (C, H, W) or (B, C, H, W)
        
        Returns:
            Denormalized image as numpy array (H, W, C) or (B, H, W, C)
            in RGB format with values in [0, 255]
        
        Example:
            >>> preprocessor = FacePreprocessor()
            >>> face_tensor = preprocessor.preprocess(frame, detection)
            >>> face_image = preprocessor.denormalize(face_tensor)
            >>> cv2.imwrite('face.jpg', cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
        """
        # Handle batch dimension
        is_batch = tensor.dim() == 4
        
        if not is_batch:
            tensor = tensor.unsqueeze(0)
        
        # Move to CPU and convert to numpy
        tensor_np = tensor.cpu().numpy()
        
        # Denormalize
        if self.normalize:
            tensor_np = tensor_np * self.std.reshape(1, 3, 1, 1) + self.mean.reshape(1, 3, 1, 1)
        
        # Clip to [0, 1]
        tensor_np = np.clip(tensor_np, 0, 1)
        
        # Scale to [0, 255]
        tensor_np = (tensor_np * 255).astype(np.uint8)
        
        # Convert (B, C, H, W) -> (B, H, W, C)
        tensor_np = tensor_np.transpose(0, 2, 3, 1)
        
        # Remove batch dimension if input was single image
        if not is_batch:
            tensor_np = tensor_np[0]
        
        return tensor_np
    
    def get_config(self) -> dict:
        """
        Get current preprocessor configuration.
        
        Returns:
            Dictionary with preprocessor settings
        """
        return {
            'target_size': self.target_size,
            'margin': self.margin,
            'normalize': self.normalize,
            'mean': self.mean.tolist(),
            'std': self.std.tolist()
        }


if __name__ == '__main__':
    # Demo usage
    print("FacePreprocessor Demo")
    print("=" * 70)
    
    # Initialize preprocessor
    preprocessor = FacePreprocessor(target_size=(224, 224), margin=0.2)
    
    # Print configuration
    print("\nPreprocessor Configuration:")
    config = preprocessor.get_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Test with dummy data
    print("\n" + "=" * 70)
    print("Testing preprocessing...")
    
    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create dummy detection
    dummy_detection = FaceDetection(
        bbox=(100, 100, 200, 200),
        confidence=0.95,
        landmarks=None
    )
    
    try:
        # Test single face preprocessing
        face_tensor = preprocessor.preprocess(dummy_frame, dummy_detection)
        print(f"\nSingle face preprocessing:")
        print(f"  Input frame shape: {dummy_frame.shape}")
        print(f"  Output tensor shape: {face_tensor.shape}")
        print(f"  Tensor dtype: {face_tensor.dtype}")
        print(f"  Tensor range: [{face_tensor.min():.3f}, {face_tensor.max():.3f}]")
        
        # Test batch preprocessing
        dummy_detections = [
            FaceDetection(bbox=(100, 100, 200, 200), confidence=0.95),
            FaceDetection(bbox=(300, 150, 180, 180), confidence=0.92),
            FaceDetection(bbox=(50, 250, 150, 150), confidence=0.88)
        ]
        
        face_batch = preprocessor.preprocess_batch(dummy_frame, dummy_detections)
        print(f"\nBatch preprocessing:")
        print(f"  Number of faces: {len(dummy_detections)}")
        print(f"  Output batch shape: {face_batch.shape}")
        
        # Test denormalization
        denorm_image = preprocessor.denormalize(face_tensor)
        print(f"\nDenormalization:")
        print(f"  Denormalized image shape: {denorm_image.shape}")
        print(f"  Denormalized image range: [{denorm_image.min()}, {denorm_image.max()}]")
        
        # Test crop_face
        face_crop = preprocessor.crop_face(dummy_frame, dummy_detection, return_rgb=True)
        print(f"\nFace cropping:")
        print(f"  Cropped face shape: {face_crop.shape}")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("\nTo use with real images:")
    print("  frame = cv2.imread('path/to/image.jpg')")
    print("  detection = detector.detect_faces(frame)[0]")
    print("  face_tensor = preprocessor.preprocess(frame, detection)")
