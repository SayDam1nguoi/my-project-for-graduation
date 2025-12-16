#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ONNX-based emotion classifier for faster inference.

Provides ONNXEmotionClassifier class that uses ONNX Runtime
for 50-60% faster inference compared to PyTorch on CPU.
"""

import numpy as np
import time
from pathlib import Path
from typing import Union, List, Optional

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available. Install with: pip install onnxruntime")

from .model_loader import EmotionPrediction


class ONNXEmotionClassifier:
    """
    ONNX-based emotion classifier for faster inference.
    
    Features:
    - 50-60% faster than PyTorch on CPU
    - Lower memory usage
    - Faster startup time
    - Cross-platform optimized
    
    Example:
        >>> classifier = ONNXEmotionClassifier('models/model.onnx')
        >>> face_tensor = preprocessor.preprocess(frame, detection)
        >>> prediction = classifier.predict(face_tensor)
        >>> print(f"Emotion: {prediction.emotion} ({prediction.confidence:.2%})")
    """
    
    # Emotion labels (must match training)
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    def __init__(
        self,
        model_path: Union[str, Path],
        confidence_threshold: float = 0.6,
        use_gpu: bool = False
    ):
        """
        Initialize ONNX emotion classifier.
        
        Args:
            model_path: Path to ONNX model file (.onnx)
            confidence_threshold: Minimum confidence for valid predictions
            use_gpu: Use GPU if available (requires onnxruntime-gpu)
        
        Raises:
            ImportError: If ONNX Runtime not installed
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX Runtime is required. Install with: pip install onnxruntime"
            )
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.confidence_threshold = confidence_threshold
        self.emotions = self.EMOTIONS
        
        # Setup execution providers
        providers = []
        if use_gpu:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        # Create ONNX Runtime session
        print(f"Loading ONNX model from: {self.model_path}")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Get input shape
        input_shape = self.session.get_inputs()[0].shape
        
        # Performance tracking
        self._inference_times = []
        self._max_history = 100
        
        print(f"ONNXEmotionClassifier initialized:")
        print(f"  Model: {self.model_path.name}")
        print(f"  Provider: {self.session.get_providers()[0]}")
        print(f"  Input shape: {input_shape}")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        print(f"  Emotions: {self.emotions}")
    
    def predict(
        self,
        face_tensor,
        return_timing: bool = False
    ) -> Union[EmotionPrediction, tuple]:
        """
        Predict emotion from a preprocessed face tensor.
        
        Args:
            face_tensor: Preprocessed face tensor (torch.Tensor or numpy.ndarray)
                        Shape: (C, H, W) or (1, C, H, W)
            return_timing: If True, return (prediction, inference_time_ms)
        
        Returns:
            EmotionPrediction with emotion label, confidence, and probabilities
            If return_timing=True, returns tuple (prediction, inference_time_ms)
        """
        # Convert to numpy if needed
        if hasattr(face_tensor, 'numpy'):
            # PyTorch tensor
            input_data = face_tensor.cpu().numpy()
        else:
            # Already numpy
            input_data = face_tensor
        
        # Ensure 4D shape (batch, channels, height, width)
        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        # Ensure float32
        input_data = input_data.astype(np.float32)
        
        # Run inference with timing
        start_time = time.perf_counter()
        
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )
        
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000
        
        # Track inference time
        self._inference_times.append(inference_time_ms)
        if len(self._inference_times) > self._max_history:
            self._inference_times.pop(0)
        
        # Process output
        logits = outputs[0][0]  # Shape: (num_classes,)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()
        
        # Get prediction
        predicted_idx = np.argmax(probabilities)
        predicted_emotion = self.emotions[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # Create probability dictionary
        prob_dict = {
            emotion: float(prob)
            for emotion, prob in zip(self.emotions, probabilities)
        }
        
        # Create prediction object
        prediction = EmotionPrediction(
            emotion=predicted_emotion,
            confidence=confidence,
            probabilities=prob_dict
        )
        
        if return_timing:
            return prediction, inference_time_ms
        else:
            return prediction
    
    def predict_batch(
        self,
        face_tensors,
        return_timing: bool = False
    ) -> Union[List[EmotionPrediction], tuple]:
        """
        Predict emotions for a batch of face tensors.
        
        Args:
            face_tensors: Batch of preprocessed face tensors
                         Shape: (B, C, H, W)
            return_timing: If True, return (predictions, inference_time_ms)
        
        Returns:
            List of EmotionPrediction objects
            If return_timing=True, returns tuple (predictions, inference_time_ms)
        """
        # Convert to numpy if needed
        if hasattr(face_tensors, 'numpy'):
            input_data = face_tensors.cpu().numpy()
        else:
            input_data = face_tensors
        
        # Ensure float32
        input_data = input_data.astype(np.float32)
        
        # Run inference with timing
        start_time = time.perf_counter()
        
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )
        
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000
        
        # Track average time per face
        avg_time_per_face = inference_time_ms / len(input_data)
        self._inference_times.append(avg_time_per_face)
        if len(self._inference_times) > self._max_history:
            self._inference_times.pop(0)
        
        # Process outputs
        logits_batch = outputs[0]  # Shape: (B, num_classes)
        
        predictions = []
        for logits in logits_batch:
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / exp_logits.sum()
            
            # Get prediction
            predicted_idx = np.argmax(probabilities)
            predicted_emotion = self.emotions[predicted_idx]
            confidence = float(probabilities[predicted_idx])
            
            # Create probability dictionary
            prob_dict = {
                emotion: float(prob)
                for emotion, prob in zip(self.emotions, probabilities)
            }
            
            # Create prediction object
            prediction = EmotionPrediction(
                emotion=predicted_emotion,
                confidence=confidence,
                probabilities=prob_dict
            )
            
            predictions.append(prediction)
        
        if return_timing:
            return predictions, inference_time_ms
        else:
            return predictions
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Update confidence threshold."""
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        self.confidence_threshold = threshold
        print(f"Confidence threshold updated to: {threshold}")
    
    def get_average_inference_time(self) -> float:
        """Get average inference time over recent predictions."""
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times) / len(self._inference_times)
    
    def get_performance_stats(self) -> dict:
        """Get detailed performance statistics."""
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
            'meets_requirement': avg_time < 30.0
        }
    
    def reset_performance_stats(self) -> None:
        """Reset performance tracking statistics."""
        self._inference_times.clear()
        print("Performance statistics reset")
    
    def get_config(self) -> dict:
        """Get current classifier configuration."""
        return {
            'model_path': str(self.model_path),
            'provider': self.session.get_providers()[0],
            'confidence_threshold': self.confidence_threshold,
            'emotions': self.emotions,
            'input_name': self.input_name,
            'output_name': self.output_name
        }


def is_onnx_available() -> bool:
    """Check if ONNX Runtime is available."""
    return ONNX_AVAILABLE


if __name__ == '__main__':
    print("ONNXEmotionClassifier Demo")
    print("=" * 70)
    
    if not ONNX_AVAILABLE:
        print("\nONNX Runtime not installed!")
        print("Install with: pip install onnxruntime")
        exit(1)
    
    # Check if ONNX model exists
    model_path = Path('models/efficientnet_b2_best.onnx')
    
    if not model_path.exists():
        print(f"\nONNX model not found: {model_path}")
        print("\nPlease export PyTorch model to ONNX first:")
        print("  python scripts/export_to_onnx.py")
        exit(0)
    
    try:
        # Initialize classifier
        print(f"\nInitializing classifier with model: {model_path}")
        classifier = ONNXEmotionClassifier(model_path)
        
        # Print configuration
        print("\nClassifier Configuration:")
        config = classifier.get_config()
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Test with dummy data
        print("\n" + "=" * 70)
        print("Testing inference...")
        
        # Create dummy face tensor
        import torch
        dummy_face = torch.randn(1, 3, 224, 224)
        
        # Test single prediction
        print("\nSingle face prediction:")
        prediction, inference_time = classifier.predict(dummy_face, return_timing=True)
        print(f"  Emotion: {prediction.emotion}")
        print(f"  Confidence: {prediction.confidence:.2%}")
        print(f"  Inference time: {inference_time:.2f}ms")
        
        # Test batch prediction
        print("\nBatch prediction (3 faces):")
        dummy_batch = torch.randn(3, 3, 224, 224)
        predictions, batch_time = classifier.predict_batch(dummy_batch, return_timing=True)
        print(f"  Number of faces: {len(predictions)}")
        print(f"  Total inference time: {batch_time:.2f}ms")
        print(f"  Average per face: {batch_time/len(predictions):.2f}ms")
        
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
            print("\n✓ Meets requirement: Process each face within 30ms")
        else:
            print(f"\n✗ Does not meet requirement: {stats['average_time_ms']:.2f}ms > 30ms")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Demo complete!")
