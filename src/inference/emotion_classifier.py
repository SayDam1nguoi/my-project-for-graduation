"""
Emotion classification module for real-time inference.

Provides EmotionClassifier class for predicting emotions from preprocessed
face images using trained PyTorch models.

Requirements: 5.1, 5.3, 5.4
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, List, Optional
import time
import warnings

from .model_loader import ModelLoader, EmotionPrediction
from .preprocessor import FacePreprocessor


class EmotionClassifier:
    """
    Emotion classifier for real-time inference.
    
    Features:
    - Load trained emotion recognition models
    - Single face prediction
    - Batch prediction for multiple faces
    - Confidence scores and probability distributions
    - GPU acceleration support
    - Performance monitoring
    
    Requirements:
    - 5.1: Classify emotions into 7 basic emotions
    - 5.3: Provide confidence scores for predictions
    - 5.4: Process each face within 30ms
    
    Example:
        >>> classifier = EmotionClassifier('models/efficientnet_b2_best.pth')
        >>> face_tensor = preprocessor.preprocess(frame, detection)
        >>> prediction = classifier.predict(face_tensor)
        >>> print(f"Emotion: {prediction.emotion} ({prediction.confidence:.2%})")
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = 'auto',
        confidence_threshold: float = 0.6,
        emotion_bias: Optional[dict] = None
    ):
        """
        Initialize EmotionClassifier.
        
        Args:
            model_path: Path to trained model checkpoint (.pth file)
            device: Device to run inference on ('cuda', 'cpu', or 'auto')
            confidence_threshold: Minimum confidence for valid predictions
                                 Predictions below this are flagged as low confidence
                                 Default 0.6 as per requirement 5.5
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
            ValueError: If parameters are invalid
        
        Example:
            >>> # Load model on GPU if available
            >>> classifier = EmotionClassifier('models/model.pth', device='auto')
            
            >>> # Load model on CPU only
            >>> classifier = EmotionClassifier('models/model.pth', device='cpu')
        """
        # Validate parameters
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
            )
        
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.emotion_bias = emotion_bias or {}  # e.g., {'happy': 1.1, 'sad': 0.9}
        
        # Initialize model loader
        self.model_loader = ModelLoader(device=device)
        self.device = self.model_loader.device
        
        # Load model
        print(f"Loading emotion classification model from: {self.model_path}")
        self.model = self.model_loader.load_model(self.model_path)
        
        # Get emotion labels
        self.emotions = self.model_loader.EMOTIONS
        
        # Performance tracking
        self._inference_times = []
        self._max_history = 100  # Keep last 100 inference times
        
        print(f"EmotionClassifier initialized:")
        print(f"  Model: {self.model_path.name}")
        print(f"  Device: {self.device}")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        print(f"  Emotions: {self.emotions}")
    
    def predict(
        self,
        face_tensor: torch.Tensor,
        return_timing: bool = False
    ) -> Union[EmotionPrediction, tuple[EmotionPrediction, float]]:
        """
        Predict emotion from a single preprocessed face tensor.
        
        Args:
            face_tensor: Preprocessed face tensor of shape (C, H, W) or (1, C, H, W)
                        Should be normalized and ready for model input
            return_timing: If True, return (prediction, inference_time_ms)
        
        Returns:
            EmotionPrediction with emotion label, confidence, and probabilities
            If return_timing=True, returns tuple (prediction, inference_time_ms)
        
        Raises:
            ValueError: If face_tensor has invalid shape
        
        Example:
            >>> classifier = EmotionClassifier('models/model.pth')
            >>> face_tensor = preprocessor.preprocess(frame, detection)
            >>> prediction = classifier.predict(face_tensor)
            >>> print(f"Emotion: {prediction.emotion}")
            >>> print(f"Confidence: {prediction.confidence:.2%}")
            >>> for emotion, prob in prediction.probabilities.items():
            ...     print(f"  {emotion}: {prob:.2%}")
        """
        # Validate input shape
        if face_tensor.dim() == 3:
            # Add batch dimension (C, H, W) -> (1, C, H, W)
            face_tensor = face_tensor.unsqueeze(0)
        elif face_tensor.dim() != 4:
            raise ValueError(
                f"face_tensor must have shape (C, H, W) or (B, C, H, W), "
                f"got shape {face_tensor.shape}"
            )
        
        # Move to device
        face_tensor = face_tensor.to(self.device)
        
        # Inference with timing
        start_time = time.perf_counter()
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(face_tensor)
            probabilities = torch.softmax(logits, dim=1)
            
            # Apply emotion bias if specified
            if self.emotion_bias:
                probabilities = self._apply_emotion_bias(probabilities)
        
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000
        
        # Track inference time
        self._inference_times.append(inference_time_ms)
        if len(self._inference_times) > self._max_history:
            self._inference_times.pop(0)
        
        # Get prediction
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        predicted_emotion = self.emotions[predicted_idx.item()]
        
        # Create probability dictionary
        prob_dict = {
            emotion: prob.item()
            for emotion, prob in zip(self.emotions, probabilities[0])
        }
        
        # Create prediction object
        prediction = EmotionPrediction(
            emotion=predicted_emotion,
            confidence=confidence.item(),
            probabilities=prob_dict
        )
        
        # Warn if confidence is below threshold (requirement 5.5)
        # Note: Warning disabled for GUI to reduce console noise
        # if prediction.confidence < self.confidence_threshold:
        #     warnings.warn(
        #         f"Low confidence prediction: {prediction.emotion} "
        #         f"({prediction.confidence:.2%} < {self.confidence_threshold:.2%})",
        #         UserWarning
        #     )
        
        if return_timing:
            return prediction, inference_time_ms
        else:
            return prediction
    
    def _apply_emotion_bias(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Apply emotion bias to probability distribution.
        
        Multiplies probabilities by bias factors and renormalizes.
        
        Args:
            probabilities: Tensor of shape (B, num_emotions) with probabilities
        
        Returns:
            Adjusted probabilities (still sum to 1.0)
        """
        # Create bias tensor
        bias_factors = torch.ones_like(probabilities)
        
        for emotion, bias in self.emotion_bias.items():
            if emotion in self.emotions:
                emotion_idx = self.emotions.index(emotion)
                bias_factors[:, emotion_idx] = bias
        
        # Apply bias
        adjusted_probs = probabilities * bias_factors
        
        # Renormalize to sum to 1.0
        adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=1, keepdim=True)
        
        return adjusted_probs
    
    def predict_batch(
        self,
        face_tensors: torch.Tensor,
        return_timing: bool = False
    ) -> Union[List[EmotionPrediction], tuple[List[EmotionPrediction], float]]:
        """
        Predict emotions for a batch of preprocessed face tensors.
        
        More efficient than calling predict() multiple times for multiple faces.
        
        Args:
            face_tensors: Batch of preprocessed face tensors of shape (B, C, H, W)
            return_timing: If True, return (predictions, inference_time_ms)
        
        Returns:
            List of EmotionPrediction objects, one for each face
            If return_timing=True, returns tuple (predictions, inference_time_ms)
        
        Raises:
            ValueError: If face_tensors has invalid shape
        
        Example:
            >>> classifier = EmotionClassifier('models/model.pth')
            >>> face_batch = preprocessor.preprocess_batch(frame, detections)
            >>> predictions = classifier.predict_batch(face_batch)
            >>> for i, pred in enumerate(predictions):
            ...     print(f"Face {i+1}: {pred.emotion} ({pred.confidence:.2%})")
        """
        # Validate input shape
        if face_tensors.dim() != 4:
            raise ValueError(
                f"face_tensors must have shape (B, C, H, W), got shape {face_tensors.shape}"
            )
        
        # Handle empty batch
        if face_tensors.shape[0] == 0:
            if return_timing:
                return [], 0.0
            else:
                return []
        
        # Move to device
        face_tensors = face_tensors.to(self.device)
        
        # Inference with timing
        start_time = time.perf_counter()
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(face_tensors)
            probabilities = torch.softmax(logits, dim=1)
            
            # Apply emotion bias if specified
            if self.emotion_bias:
                probabilities = self._apply_emotion_bias(probabilities)
        
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000
        
        # Track inference time (average per face)
        avg_time_per_face = inference_time_ms / face_tensors.shape[0]
        self._inference_times.append(avg_time_per_face)
        if len(self._inference_times) > self._max_history:
            self._inference_times.pop(0)
        
        # Get predictions for each face
        confidences, predicted_indices = torch.max(probabilities, dim=1)
        
        predictions = []
        for i in range(len(face_tensors)):
            predicted_emotion = self.emotions[predicted_indices[i].item()]
            confidence = confidences[i].item()
            
            prob_dict = {
                emotion: prob.item()
                for emotion, prob in zip(self.emotions, probabilities[i])
            }
            
            prediction = EmotionPrediction(
                emotion=predicted_emotion,
                confidence=confidence,
                probabilities=prob_dict
            )
            
            # Warn if confidence is below threshold
            if prediction.confidence < self.confidence_threshold:
                warnings.warn(
                    f"Low confidence prediction for face {i+1}: {prediction.emotion} "
                    f"({prediction.confidence:.2%} < {self.confidence_threshold:.2%})",
                    UserWarning
                )
            
            predictions.append(prediction)
        
        if return_timing:
            return predictions, inference_time_ms
        else:
            return predictions
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Update confidence threshold for low confidence warnings.
        
        Args:
            threshold: New confidence threshold (0.0-1.0)
        
        Raises:
            ValueError: If threshold is invalid
        
        Example:
            >>> classifier = EmotionClassifier('models/model.pth')
            >>> classifier.set_confidence_threshold(0.7)  # More strict
            >>> classifier.set_confidence_threshold(0.5)  # More lenient
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"Confidence threshold must be between 0.0 and 1.0, got {threshold}"
            )
        
        self.confidence_threshold = threshold
        print(f"Confidence threshold updated to: {threshold}")
    
    def get_average_inference_time(self) -> float:
        """
        Get average inference time over recent predictions.
        
        Returns:
            Average inference time in milliseconds
            Returns 0.0 if no predictions have been made yet
        
        Example:
            >>> classifier = EmotionClassifier('models/model.pth')
            >>> # ... make some predictions ...
            >>> avg_time = classifier.get_average_inference_time()
            >>> print(f"Average inference time: {avg_time:.2f}ms")
        """
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times) / len(self._inference_times)
    
    def get_performance_stats(self) -> dict:
        """
        Get detailed performance statistics.
        
        Returns:
            Dictionary with performance metrics:
            - average_time_ms: Average inference time
            - min_time_ms: Minimum inference time
            - max_time_ms: Maximum inference time
            - num_predictions: Number of predictions tracked
            - meets_requirement: Whether average time meets 30ms requirement
        
        Example:
            >>> classifier = EmotionClassifier('models/model.pth')
            >>> # ... make some predictions ...
            >>> stats = classifier.get_performance_stats()
            >>> print(f"Average: {stats['average_time_ms']:.2f}ms")
            >>> print(f"Meets requirement: {stats['meets_requirement']}")
        """
        if not self._inference_times:
            return {
                'average_time_ms': 0.0,
                'min_time_ms': 0.0,
                'max_time_ms': 0.0,
                'num_predictions': 0,
                'meets_requirement': None
            }
        
        avg_time = sum(self._inference_times) / len(self._inference_times)
        
        return {
            'average_time_ms': avg_time,
            'min_time_ms': min(self._inference_times),
            'max_time_ms': max(self._inference_times),
            'num_predictions': len(self._inference_times),
            'meets_requirement': avg_time < 30.0  # Requirement 5.4: <30ms
        }
    
    def reset_performance_stats(self) -> None:
        """
        Reset performance tracking statistics.
        
        Useful when you want to measure performance for a specific session.
        
        Example:
            >>> classifier = EmotionClassifier('models/model.pth')
            >>> classifier.reset_performance_stats()
            >>> # ... make predictions ...
            >>> stats = classifier.get_performance_stats()
        """
        self._inference_times.clear()
        print("Performance statistics reset")
    
    def get_config(self) -> dict:
        """
        Get current classifier configuration.
        
        Returns:
            Dictionary with classifier settings
        """
        return {
            'model_path': str(self.model_path),
            'device': str(self.device),
            'confidence_threshold': self.confidence_threshold,
            'emotions': self.emotions,
            'num_parameters': sum(p.numel() for p in self.model.parameters())
        }


if __name__ == '__main__':
    # Demo usage
    print("EmotionClassifier Demo")
    print("=" * 70)
    
    # Check if model exists
    model_path = Path('models/efficientnet_b2_best.pth')
    
    if not model_path.exists():
        print(f"\nModel not found: {model_path}")
        print("Please train a model first using train.py")
        print("\nExample:")
        print("  python train.py --model efficientnet_b2 \\")
        print("                  --dataset data/processed/dataset.csv \\")
        print("                  --epochs 50 --batch-size 32")
        print("\n" + "=" * 70)
        exit(0)
    
    try:
        # Initialize classifier
        print(f"\nInitializing classifier with model: {model_path}")
        classifier = EmotionClassifier(model_path, device='auto')
        
        # Print configuration
        print("\nClassifier Configuration:")
        config = classifier.get_config()
        for key, value in config.items():
            if key == 'num_parameters':
                print(f"  {key}: {value:,}")
            else:
                print(f"  {key}: {value}")
        
        # Test with dummy data
        print("\n" + "=" * 70)
        print("Testing inference...")
        
        # Create dummy face tensor (normalized)
        import numpy as np
        dummy_face = torch.randn(1, 3, 224, 224)
        
        # Test single prediction
        print("\nSingle face prediction:")
        prediction, inference_time = classifier.predict(dummy_face, return_timing=True)
        print(f"  Emotion: {prediction.emotion}")
        print(f"  Confidence: {prediction.confidence:.2%}")
        print(f"  Inference time: {inference_time:.2f}ms")
        print(f"  Probabilities:")
        for emotion, prob in prediction.probabilities.items():
            print(f"    {emotion:10s}: {prob:.2%}")
        
        # Test batch prediction
        print("\nBatch prediction (3 faces):")
        dummy_batch = torch.randn(3, 3, 224, 224)
        predictions, batch_time = classifier.predict_batch(dummy_batch, return_timing=True)
        print(f"  Number of faces: {len(predictions)}")
        print(f"  Total inference time: {batch_time:.2f}ms")
        print(f"  Average per face: {batch_time/len(predictions):.2f}ms")
        for i, pred in enumerate(predictions):
            print(f"  Face {i+1}: {pred.emotion} ({pred.confidence:.2%})")
        
        # Performance statistics
        print("\nPerformance Statistics:")
        stats = classifier.get_performance_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        # Check if meets requirement
        if stats['meets_requirement']:
            print("\n✓ Meets requirement 5.4: Process each face within 30ms")
        else:
            print(f"\n✗ Does not meet requirement 5.4: {stats['average_time_ms']:.2f}ms > 30ms")
            print("  Consider using GPU or a smaller model for faster inference")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("\nTo use with real faces:")
    print("  from inference import FaceDetector, FacePreprocessor, EmotionClassifier")
    print("  ")
    print("  detector = FaceDetector()")
    print("  preprocessor = FacePreprocessor()")
    print("  classifier = EmotionClassifier('models/model.pth')")
    print("  ")
    print("  frame = cv2.imread('image.jpg')")
    print("  detections = detector.detect_faces(frame)")
    print("  for detection in detections:")
    print("      face_tensor = preprocessor.preprocess(frame, detection)")
    print("      prediction = classifier.predict(face_tensor)")
    print("      print(f'Emotion: {prediction.emotion}')")
