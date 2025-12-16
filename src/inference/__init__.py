"""
Inference module for facial emotion recognition.

This module provides components for loading trained models and performing
real-time emotion inference on face images.

Components:
- ModelLoader: Load and manage trained PyTorch models
- FaceDetector: Detect faces in video frames using MTCNN
- FacePreprocessor: Preprocess detected faces for emotion classification
- EmotionClassifier: Classify emotions from preprocessed face images
- VideoStreamHandler: Handle video input from camera or video files
- FaceDetection: Dataclass for face detection results
- EmotionPrediction: Dataclass for emotion prediction results
"""

from .model_loader import ModelLoader, FaceDetection, EmotionPrediction
from .face_detector import FaceDetector
from .preprocessor import FacePreprocessor
from .emotion_classifier import EmotionClassifier
from .video_stream import VideoStreamHandler
from .visualizer import VisualizationEngine, VisualizationConfig

__all__ = [
    'ModelLoader',
    'FaceDetector',
    'FacePreprocessor',
    'EmotionClassifier',
    'VideoStreamHandler',
    'VisualizationEngine',
    'VisualizationConfig',
    'FaceDetection',
    'EmotionPrediction',
]
