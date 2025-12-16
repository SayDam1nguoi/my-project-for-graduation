"""
Model loader for emotion recognition inference.

Provides ModelLoader class for loading trained PyTorch models with automatic
device detection, model validation, and error handling. Also includes dataclasses
for inference results.

Requirements: 9.1, 9.2
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Tuple, Union
from dataclasses import dataclass
import numpy as np
import warnings


@dataclass
class FaceDetection:
    """
    Face detection result containing bounding box, confidence, and landmarks.
    
    Attributes:
        bbox: Bounding box coordinates (x, y, width, height)
        confidence: Detection confidence score (0.0 to 1.0)
        landmarks: Optional facial landmarks array (5 points: 2 eyes, nose, 2 mouth corners)
    """
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    landmarks: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate face detection data."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        if len(self.bbox) != 4:
            raise ValueError(f"Bbox must have 4 elements (x, y, w, h), got {len(self.bbox)}")
        
        if self.landmarks is not None and self.landmarks.shape != (5, 2):
            raise ValueError(f"Landmarks must have shape (5, 2), got {self.landmarks.shape}")


@dataclass
class EmotionPrediction:
    """
    Emotion prediction result containing predicted emotion, confidence, and probabilities.
    
    Attributes:
        emotion: Predicted emotion label (one of 7 basic emotions)
        confidence: Prediction confidence score (0.0 to 1.0)
        probabilities: Dictionary mapping emotion labels to their probabilities
    """
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    
    def __post_init__(self):
        """Validate emotion prediction data."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        # Validate probabilities sum to ~1.0
        prob_sum = sum(self.probabilities.values())
        if not (0.99 <= prob_sum <= 1.01):
            warnings.warn(f"Probabilities sum to {prob_sum:.4f}, expected ~1.0")
        
        # Validate all probabilities are in valid range
        for emotion, prob in self.probabilities.items():
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"Probability for {emotion} must be between 0.0 and 1.0, got {prob}")


class ModelLoader:
    """
    Load and manage trained emotion recognition models.
    
    Features:
    - Automatic device detection (CUDA/CPU)
    - Model validation and error handling
    - Support for multiple model architectures
    - Efficient model caching
    
    Requirements: 9.1 (GPU/CPU detection), 9.2 (CPU fallback)
    
    Example:
        >>> loader = ModelLoader(device='auto')
        >>> model = loader.load_model('models/efficientnet_b2_best.pth')
        >>> print(f"Model loaded on {loader.device}")
    """
    
    # Standard 7 emotions (must match training)
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize ModelLoader.
        
        Args:
            device: Device to load models on ('cuda', 'cpu', or 'auto')
                   'auto' will use CUDA if available, otherwise CPU
        
        Raises:
            ValueError: If device is invalid
        """
        self.device = self._detect_device(device)
        self._loaded_models: Dict[str, nn.Module] = {}
        
        print(f"ModelLoader initialized on device: {self.device}")
    
    def _detect_device(self, device: str) -> torch.device:
        """
        Detect and validate device for model inference.
        
        Args:
            device: Device specification ('cuda', 'cpu', or 'auto')
        
        Returns:
            torch.device object
        
        Raises:
            ValueError: If device is invalid
        """
        if device == 'auto':
            # Auto-detect: use CUDA if available, otherwise CPU
            if torch.cuda.is_available():
                device_obj = torch.device('cuda')
                print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                device_obj = torch.device('cpu')
                print("CUDA not available, using CPU")
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
    
    def load_model(
        self,
        model_path: Union[str, Path],
        model_class: Optional[type] = None,
        strict: bool = True
    ) -> nn.Module:
        """
        Load a trained PyTorch model from checkpoint file.
        
        Args:
            model_path: Path to model checkpoint (.pth or .pt file)
            model_class: Optional model class to instantiate. If None, will try to
                        load from checkpoint or use default BaseEmotionModel
            strict: Whether to strictly enforce that the keys in state_dict match
                   the keys returned by this module's state_dict() function
        
        Returns:
            Loaded model in evaluation mode
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
            ValueError: If model validation fails
        
        Example:
            >>> loader = ModelLoader()
            >>> model = loader.load_model('models/efficientnet_b2_best.pth')
            >>> # Model is ready for inference
        """
        model_path = Path(model_path)
        
        # Validate model file exists
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Check if model is already loaded (caching)
        cache_key = str(model_path.absolute())
        if cache_key in self._loaded_models:
            print(f"Using cached model: {model_path.name}")
            return self._loaded_models[cache_key]
        
        print(f"Loading model from: {model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False  # Allow loading full checkpoint with model architecture
            )
            
            # Extract model state dict and metadata
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
                model_config = checkpoint.get('model_config', {})
                epoch = checkpoint.get('epoch', 'unknown')
                
                print(f"Checkpoint info:")
                print(f"  Epoch: {epoch}")
                if 'val_accuracy' in checkpoint:
                    print(f"  Validation accuracy: {checkpoint['val_accuracy']:.4f}")
            else:
                # Checkpoint is just state dict
                state_dict = checkpoint
                model_config = {}
            
            # Create model instance
            if model_class is not None:
                # Use provided model class
                model = model_class(**model_config)
            elif 'model_class' in checkpoint:
                # Try to instantiate from checkpoint
                model = checkpoint['model_class'](**model_config)
            else:
                # Default: try to import and use BaseEmotionModel
                try:
                    from ..training.models import BaseEmotionModel
                    backbone = model_config.get('backbone', 'efficientnet_b2')
                    model = BaseEmotionModel(
                        backbone=backbone,
                        pretrained=False,  # Don't load ImageNet weights
                        num_classes=7
                    )
                except ImportError as e:
                    raise RuntimeError(
                        f"Could not import BaseEmotionModel. "
                        f"Please provide model_class parameter. Error: {e}"
                    )
            
            # Load state dict into model
            model.load_state_dict(state_dict, strict=strict)
            
            # Move model to device and set to eval mode
            model = model.to(self.device)
            model.eval()
            
            # Validate model
            self._validate_model(model)
            
            # Cache model
            self._loaded_models[cache_key] = model
            
            print(f"Model loaded successfully: {model_path.name}")
            print(f"  Device: {self.device}")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}") from e
    
    def _validate_model(self, model: nn.Module) -> None:
        """
        Validate loaded model structure and compatibility.
        
        Args:
            model: Loaded model to validate
        
        Raises:
            ValueError: If model validation fails
        """
        # Test forward pass with dummy input
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            # Validate output shape
            if output.shape[1] != len(self.EMOTIONS):
                raise ValueError(
                    f"Model output dimension {output.shape[1]} doesn't match "
                    f"expected number of emotions {len(self.EMOTIONS)}"
                )
            
            print(f"Model validation passed")
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {output.shape}")
            
        except Exception as e:
            raise ValueError(f"Model validation failed: {str(e)}") from e
    
    def predict(
        self,
        model: nn.Module,
        face_image: Union[torch.Tensor, np.ndarray]
    ) -> EmotionPrediction:
        """
        Predict emotion from face image.
        
        Args:
            model: Loaded emotion recognition model
            face_image: Face image as torch.Tensor (C, H, W) or numpy array (H, W, C)
                       Should be preprocessed (normalized, resized to 224x224)
        
        Returns:
            EmotionPrediction with predicted emotion, confidence, and probabilities
        
        Example:
            >>> loader = ModelLoader()
            >>> model = loader.load_model('models/model.pth')
            >>> face_tensor = preprocess_face(face_image)
            >>> prediction = loader.predict(model, face_tensor)
            >>> print(f"Emotion: {prediction.emotion} ({prediction.confidence:.2%})")
        """
        # Convert numpy array to tensor if needed
        if isinstance(face_image, np.ndarray):
            # Assume numpy array is (H, W, C) format
            face_image = torch.from_numpy(face_image).permute(2, 0, 1).float()
        
        # Add batch dimension if needed
        if face_image.dim() == 3:
            face_image = face_image.unsqueeze(0)
        
        # Move to device
        face_image = face_image.to(self.device)
        
        # Inference
        model.eval()
        with torch.no_grad():
            logits = model(face_image)
            probabilities = torch.softmax(logits, dim=1)
        
        # Get prediction
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        predicted_emotion = self.EMOTIONS[predicted_idx.item()]
        
        # Create probability dictionary
        prob_dict = {
            emotion: prob.item()
            for emotion, prob in zip(self.EMOTIONS, probabilities[0])
        }
        
        return EmotionPrediction(
            emotion=predicted_emotion,
            confidence=confidence.item(),
            probabilities=prob_dict
        )
    
    def predict_batch(
        self,
        model: nn.Module,
        face_images: Union[torch.Tensor, list]
    ) -> list[EmotionPrediction]:
        """
        Predict emotions for batch of face images.
        
        Args:
            model: Loaded emotion recognition model
            face_images: Batch of face images as torch.Tensor (B, C, H, W) or
                        list of tensors/arrays
        
        Returns:
            List of EmotionPrediction objects
        
        Example:
            >>> loader = ModelLoader()
            >>> model = loader.load_model('models/model.pth')
            >>> predictions = loader.predict_batch(model, face_batch)
            >>> for pred in predictions:
            ...     print(f"{pred.emotion}: {pred.confidence:.2%}")
        """
        # Convert list to batch tensor if needed
        if isinstance(face_images, list):
            # Process each image individually
            return [self.predict(model, img) for img in face_images]
        
        # Batch tensor processing
        face_images = face_images.to(self.device)
        
        model.eval()
        with torch.no_grad():
            logits = model(face_images)
            probabilities = torch.softmax(logits, dim=1)
        
        # Get predictions for each image in batch
        confidences, predicted_indices = torch.max(probabilities, dim=1)
        
        predictions = []
        for i in range(len(face_images)):
            predicted_emotion = self.EMOTIONS[predicted_indices[i].item()]
            
            prob_dict = {
                emotion: prob.item()
                for emotion, prob in zip(self.EMOTIONS, probabilities[i])
            }
            
            predictions.append(EmotionPrediction(
                emotion=predicted_emotion,
                confidence=confidences[i].item(),
                probabilities=prob_dict
            ))
        
        return predictions
    
    def clear_cache(self) -> None:
        """
        Clear cached models from memory.
        
        Useful for freeing up memory when models are no longer needed.
        """
        self._loaded_models.clear()
        print("Model cache cleared")
    
    def get_device_info(self) -> Dict[str, any]:
        """
        Get information about the current device.
        
        Returns:
            Dictionary with device information
        """
        info = {
            'device': str(self.device),
            'device_type': self.device.type,
        }
        
        if self.device.type == 'cuda':
            info.update({
                'cuda_available': True,
                'cuda_device_name': torch.cuda.get_device_name(0),
                'cuda_device_count': torch.cuda.device_count(),
                'cuda_memory_allocated': torch.cuda.memory_allocated(0),
                'cuda_memory_reserved': torch.cuda.memory_reserved(0),
            })
        else:
            info['cuda_available'] = False
        
        return info


if __name__ == '__main__':
    # Demo usage
    print("ModelLoader Demo")
    print("=" * 70)
    
    # Initialize loader
    loader = ModelLoader(device='auto')
    
    # Print device info
    print("\nDevice Information:")
    device_info = loader.get_device_info()
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    # Demo dataclasses
    print("\n" + "=" * 70)
    print("Testing dataclasses...")
    
    # Create sample face detection
    face_det = FaceDetection(
        bbox=(100, 100, 200, 200),
        confidence=0.95,
        landmarks=np.array([[120, 130], [180, 130], [150, 160], [130, 190], [170, 190]])
    )
    print(f"\nFaceDetection: {face_det}")
    
    # Create sample emotion prediction
    emotion_pred = EmotionPrediction(
        emotion='Happy',
        confidence=0.87,
        probabilities={
            'Angry': 0.02,
            'Disgust': 0.01,
            'Fear': 0.03,
            'Happy': 0.87,
            'Sad': 0.02,
            'Surprise': 0.03,
            'Neutral': 0.02
        }
    )
    print(f"\nEmotionPrediction: {emotion_pred}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
