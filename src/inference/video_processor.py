# -*- coding: utf-8 -*-
"""
Video processors for emotion recognition.

Separate processors for camera (real-time) and video file (batch) processing
to optimize for different use cases.
"""

import cv2
import time
from typing import List, Tuple, Dict, Optional
from pathlib import Path

from .face_detector import FaceDetector
from .preprocessor import FacePreprocessor
from .emotion_classifier import EmotionClassifier
from .model_loader import FaceDetection, EmotionPrediction


class CameraProcessor:
    """
    Processor optimized for real-time camera input.
    
    Features:
    - Fast processing, skip frames if needed
    - Real-time display priority
    - Continuous processing
    """
    
    def __init__(
        self,
        detector: FaceDetector,
        preprocessor: FacePreprocessor,
        classifier: EmotionClassifier,
        detection_skip_frames: int = 1,  # Skip frames for face detection
        cache_results: bool = True  # Cache detection results
    ):
        """
        Initialize camera processor.
        
        Args:
            detector: Face detector instance
            preprocessor: Face preprocessor instance
            classifier: Emotion classifier instance
            detection_skip_frames: Skip N frames between face detections (0=every frame)
            cache_results: Whether to cache and reuse detection results
        """
        self.detector = detector
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.detection_skip_frames = detection_skip_frames
        self.cache_results = cache_results
        
        # Statistics
        self.emotion_counts = {}
        self.total_faces = 0
        self.frame_count = 0
        self.start_time = None
        
        # Caching for performance
        self._cached_detections = []
        self._cached_results = []
        self._cached_predictions = {}  # Cache predictions by face position
        self._detection_frame_counter = 0
        self._emotion_cache_counter = 0
        self._emotion_cache_interval = 5  # Cache emotion for 5 frames
        
        # Initialize emotion counts
        if classifier.emotions:
            self.emotion_counts = {emotion: 0 for emotion in classifier.emotions}
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.emotion_counts = {emotion: 0 for emotion in self.classifier.emotions}
        self.total_faces = 0
        self.frame_count = 0
        self.start_time = time.time()
    
    def process_frame(
        self,
        frame
    ) -> Tuple[List[Tuple[FaceDetection, EmotionPrediction]], float]:
        """
        Process a single frame from camera with smart caching.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            Tuple of (results, fps) where results is list of (detection, prediction)
        """
        if self.start_time is None:
            self.start_time = time.time()
        
        self.frame_count += 1
        self._detection_frame_counter += 1
        
        # Smart face detection - skip frames for performance
        should_detect = (self.detection_skip_frames == 0 or 
                        self._detection_frame_counter % (self.detection_skip_frames + 1) == 0)
        
        if should_detect or not self.cache_results:
            # Detect faces
            detections = self.detector.detect_faces(frame)
            self.total_faces += len(detections)
            
            # Cache detections
            if self.cache_results:
                self._cached_detections = detections
        else:
            # Reuse cached detections
            detections = self._cached_detections
        
        # Process each face with emotion caching
        results = []
        self._emotion_cache_counter += 1
        should_predict = (self._emotion_cache_counter % self._emotion_cache_interval == 0)
        
        for i, detection in enumerate(detections):
            try:
                # Create cache key based on face position
                x, y, w, h = detection.bbox
                cache_key = f"{x//10}_{y//10}_{w//10}_{h//10}"
                
                # Use cached prediction if available and not time to update
                if not should_predict and cache_key in self._cached_predictions:
                    prediction = self._cached_predictions[cache_key]
                else:
                    # Preprocess
                    face_tensor = self.preprocessor.preprocess(frame, detection)
                    
                    # Predict
                    prediction = self.classifier.predict(face_tensor)
                    
                    # Cache prediction
                    self._cached_predictions[cache_key] = prediction
                
                # Update statistics
                if prediction.confidence >= self.classifier.confidence_threshold:
                    self.emotion_counts[prediction.emotion] += 1
                
                results.append((detection, prediction))
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        # Calculate FPS
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        return results, fps
    
    def get_statistics(self) -> Dict:
        """
        Get current statistics.
        
        Returns:
            Dictionary with statistics
        """
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        return {
            'emotion_counts': self.emotion_counts.copy(),
            'total_faces': self.total_faces,
            'frame_count': self.frame_count,
            'elapsed_time': elapsed,
            'fps': self.frame_count / elapsed if elapsed > 0 else 0
        }


class VideoFileProcessor:
    """
    Processor optimized for video file input.
    
    Features:
    - Process every frame (no skipping)
    - Better accuracy with frame sampling
    - Temporal smoothing for consistent results
    - Detailed final report
    - Optional video preprocessing pipeline for enhanced accuracy
    """
    
    def __init__(
        self,
        detector: FaceDetector,
        preprocessor: FacePreprocessor,
        classifier: EmotionClassifier,
        frame_skip: int = 0,
        temporal_window: int = 5,
        enable_video_preprocessing: bool = False,
        enable_temporal_smoothing: bool = True
    ):
        """
        Initialize video file processor.
        
        Args:
            detector: Face detector instance
            preprocessor: Face preprocessor instance
            classifier: Emotion classifier instance
            frame_skip: Number of frames to skip (0 = process all frames)
            temporal_window: Number of frames for temporal smoothing
            enable_video_preprocessing: Enable video preprocessing pipeline
            enable_temporal_smoothing: Enable industry-standard temporal smoothing
        """
        self.detector = detector
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.frame_skip = frame_skip
        self.temporal_window = temporal_window
        self.enable_video_preprocessing = enable_video_preprocessing
        self.enable_temporal_smoothing = enable_temporal_smoothing
        
        # Initialize temporal processor (industry-standard technique)
        self.temporal_processor = None
        if enable_temporal_smoothing:
            try:
                from src.video_analysis.temporal_processor import AdaptiveTemporalProcessor
                self.temporal_processor = AdaptiveTemporalProcessor(
                    min_window_size=3,
                    max_window_size=temporal_window,
                    min_confidence=0.5,
                    smoothing_method='weighted_average'
                )
                print("✓ Temporal smoothing enabled (industry-standard)")
            except ImportError as e:
                print(f"Warning: Could not load temporal processor: {e}")
                self.enable_temporal_smoothing = False
        
        # Initialize video preprocessing pipeline if enabled
        self.video_preprocessing_pipeline = None
        if enable_video_preprocessing:
            try:
                from src.video_analysis.preprocessing import VideoPreprocessingPipeline
                self.video_preprocessing_pipeline = VideoPreprocessingPipeline()
                print("Video preprocessing pipeline enabled")
            except ImportError as e:
                print(f"Warning: Could not load video preprocessing pipeline: {e}")
                self.enable_video_preprocessing = False
        
        # Statistics
        self.emotion_counts = {}
        self.total_faces = 0
        self.frame_count = 0
        self.processed_frames = 0
        self.filtered_frames = 0
        self.start_time = None
        
        # Temporal smoothing
        self.face_history = {}  # Track faces across frames
        self.emotion_history = {}  # Store recent emotions for each face
        
        # Initialize emotion counts
        if classifier.emotions:
            self.emotion_counts = {emotion: 0 for emotion in classifier.emotions}
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.emotion_counts = {emotion: 0 for emotion in self.classifier.emotions}
        self.total_faces = 0
        self.frame_count = 0
        self.processed_frames = 0
        self.filtered_frames = 0
        self.start_time = time.time()
        self.face_history = {}
        self.emotion_history = {}
        
        # Reset preprocessing pipeline if enabled
        if self.video_preprocessing_pipeline is not None:
            self.video_preprocessing_pipeline.reset()
    
    def _should_process_frame(self) -> bool:
        """
        Determine if current frame should be processed.
        
        Returns:
            True if frame should be processed
        """
        if self.frame_skip == 0:
            return True
        return self.frame_count % (self.frame_skip + 1) == 0
    
    def _apply_temporal_smoothing(
        self,
        face_id: int,
        prediction: EmotionPrediction
    ) -> EmotionPrediction:
        """
        Apply temporal smoothing to prediction.
        
        Args:
            face_id: Unique face identifier
            prediction: Current prediction
        
        Returns:
            Smoothed prediction
        """
        # Initialize history for new face
        if face_id not in self.emotion_history:
            self.emotion_history[face_id] = []
        
        # Add current prediction
        self.emotion_history[face_id].append(prediction)
        
        # Keep only recent history
        if len(self.emotion_history[face_id]) > self.temporal_window:
            self.emotion_history[face_id].pop(0)
        
        # If not enough history, return current prediction
        if len(self.emotion_history[face_id]) < 3:
            return prediction
        
        # Average probabilities across temporal window
        avg_probs = {}
        for emotion in self.classifier.emotions:
            probs = [p.probabilities.get(emotion, 0) 
                    for p in self.emotion_history[face_id]]
            avg_probs[emotion] = sum(probs) / len(probs)
        
        # Find emotion with highest average probability
        smoothed_emotion = max(avg_probs.items(), key=lambda x: x[1])
        
        # Create smoothed prediction
        from .model_loader import EmotionPrediction
        return EmotionPrediction(
            emotion=smoothed_emotion[0],
            confidence=smoothed_emotion[1],
            probabilities=avg_probs
        )
    
    def process_frame(
        self,
        frame
    ) -> Tuple[List[Tuple[FaceDetection, EmotionPrediction]], float]:
        """
        Process a single frame from video file.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            Tuple of (results, fps) where results is list of (detection, prediction)
        """
        if self.start_time is None:
            self.start_time = time.time()
        
        self.frame_count += 1
        
        # Check if we should process this frame
        if not self._should_process_frame():
            return [], 0
        
        self.processed_frames += 1
        
        # Apply video preprocessing if enabled
        preprocessed_frame = frame
        if self.video_preprocessing_pipeline is not None:
            from src.video_analysis.preprocessing import FrameMetadata
            
            # Create frame metadata
            metadata = FrameMetadata(
                frame_number=self.frame_count,
                timestamp=self.frame_count / 30.0,  # Assume 30 fps
                fps=30.0,
                resolution=(frame.shape[1], frame.shape[0])
            )
            
            # Detect faces first for preprocessing
            detections = self.detector.detect_faces(frame)
            
            # Process frame through preprocessing pipeline
            if len(detections) > 0:
                # Use first face for preprocessing
                detection = detections[0]
                face_bbox = detection.bbox
                face_landmarks = detection.landmarks if hasattr(detection, 'landmarks') else None
                face_confidence = detection.confidence
                
                processed = self.video_preprocessing_pipeline.process_frame(
                    frame,
                    metadata,
                    face_bbox=face_bbox,
                    face_landmarks=face_landmarks,
                    face_confidence=face_confidence
                )
                
                # Check if frame is valid
                if not processed.is_valid:
                    self.filtered_frames += 1
                    return [], 0
                
                preprocessed_frame = processed.frame
            else:
                # No face detected, use original frame
                preprocessed_frame = frame
        else:
            # Detect faces without preprocessing
            detections = self.detector.detect_faces(frame)
        
        self.total_faces += len(detections)
        
        # Process each face
        results = []
        for idx, detection in enumerate(detections):
            try:
                # Preprocess face
                face_tensor = self.preprocessor.preprocess(preprocessed_frame, detection)
                
                # Predict emotion
                prediction = self.classifier.predict(face_tensor)
                
                # Apply industry-standard temporal smoothing
                if self.temporal_processor is not None:
                    smoothed_emotion, smoothed_confidence, smoothed_probs = \
                        self.temporal_processor.process(
                            emotion=prediction.emotion,
                            confidence=prediction.confidence,
                            probabilities=prediction.probabilities,
                            frame_number=self.frame_count
                        )
                    
                    # Update prediction with smoothed values
                    from .model_loader import EmotionPrediction
                    prediction = EmotionPrediction(
                        emotion=smoothed_emotion,
                        confidence=smoothed_confidence,
                        probabilities=smoothed_probs
                    )
                
                # Apply temporal smoothing from preprocessing pipeline if enabled (legacy)
                elif self.video_preprocessing_pipeline is not None:
                    smoothed_probs, smoothed_label, smoothed_conf = \
                        self.video_preprocessing_pipeline.smooth_emotion_prediction(
                            prediction.probabilities_array if hasattr(prediction, 'probabilities_array') 
                            else list(prediction.probabilities.values()),
                            prediction.emotion,
                            prediction.confidence
                        )
                    
                    # Update prediction with smoothed values
                    from .model_loader import EmotionPrediction
                    prediction = EmotionPrediction(
                        emotion=smoothed_label,
                        confidence=smoothed_conf,
                        probabilities=dict(zip(self.classifier.emotions, smoothed_probs))
                    )
                
                # Update statistics
                if prediction.confidence >= self.classifier.confidence_threshold:
                    self.emotion_counts[prediction.emotion] += 1
                
                results.append((detection, prediction))
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        # Calculate FPS
        elapsed = time.time() - self.start_time
        fps = self.processed_frames / elapsed if elapsed > 0 else 0
        
        return results, fps
    
    def get_statistics(self) -> Dict:
        """
        Get current statistics.
        
        Returns:
            Dictionary with statistics
        """
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        stats = {
            'emotion_counts': self.emotion_counts.copy(),
            'total_faces': self.total_faces,
            'frame_count': self.frame_count,
            'processed_frames': self.processed_frames,
            'filtered_frames': self.filtered_frames,
            'elapsed_time': elapsed,
            'fps': self.processed_frames / elapsed if elapsed > 0 else 0,
            'processing_rate': self.processed_frames / self.frame_count if self.frame_count > 0 else 0
        }
        
        # Add preprocessing statistics if enabled
        if self.video_preprocessing_pipeline is not None:
            preprocessing_stats = self.video_preprocessing_pipeline.get_statistics()
            stats['preprocessing'] = {
                'total_frames': preprocessing_stats.total_frames,
                'valid_frames': preprocessing_stats.valid_frames,
                'filtered_frames': preprocessing_stats.filtered_frames,
                'filter_rate': preprocessing_stats.filter_rate,
                'filter_reasons': preprocessing_stats.filter_reasons
            }
        
        return stats
    
    def get_dominant_emotion(self) -> Tuple[str, int, float]:
        """
        Get the dominant emotion from video analysis.
        
        Returns:
            Tuple of (emotion_name, count, percentage)
        """
        total = sum(self.emotion_counts.values())
        if total == 0:
            return ("Unknown", 0, 0.0)
        
        dominant = max(self.emotion_counts.items(), key=lambda x: x[1])
        percentage = (dominant[1] / total) * 100
        
        return (dominant[0], dominant[1], percentage)
