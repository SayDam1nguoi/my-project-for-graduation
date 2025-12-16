"""
Person Analyzer for Dual Person Comparison System.

This module analyzes a single person (primary or secondary) by integrating
face detection, emotion classification, and attention detection. It manages
historical data and calculates scores using ScoreCalculator.

Requirements: 1.1, 3.3, 3.4, 3.5, 4.1, 7.1, 7.3, 7.4, 7.5
"""

import time
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from collections import deque

from src.inference.face_detector import FaceDetector
from src.inference.preprocessor import FacePreprocessor
from src.inference.emotion_classifier import EmotionClassifier
from src.video_analysis.attention_detector import AttentionDetector
from src.video_analysis.score_calculator import ScoreCalculator
from src.video_analysis.dual_person_models import EmotionData, PersonResult


class PersonAnalyzer:
    """Analyze a single person for emotion and attention.
    
    Integrates face detection, emotion classification, and attention detection
    to provide comprehensive analysis of one person. Manages historical data
    and calculates emotion and attention scores.
    
    Attributes:
        person_id: Identifier for this person ("primary" or "secondary")
        face_detector: FaceDetector instance for detecting faces
        preprocessor: FacePreprocessor instance for preparing faces
        emotion_classifier: EmotionClassifier instance for emotion prediction
        attention_detector: AttentionDetector instance for attention analysis
        score_calculator: ScoreCalculator instance for score calculation
    """
    
    def __init__(
        self,
        person_id: str,
        face_detector: FaceDetector,
        preprocessor: FacePreprocessor,
        emotion_classifier: EmotionClassifier,
        attention_detector: AttentionDetector,
        score_calculator: ScoreCalculator = None,
        appearance_coordinator: Optional['AppearanceAssessmentCoordinator'] = None,
        low_confidence_threshold: float = 0.6,
        no_face_warning_duration: float = 5.0
    ):
        """Initialize PersonAnalyzer.
        
        Args:
            person_id: Identifier ("primary" or "secondary")
            face_detector: FaceDetector instance
            preprocessor: FacePreprocessor instance
            emotion_classifier: EmotionClassifier instance
            attention_detector: AttentionDetector instance
            score_calculator: ScoreCalculator instance (creates new if None)
            appearance_coordinator: Optional AppearanceAssessmentCoordinator for appearance assessment
            low_confidence_threshold: Threshold for low confidence warnings
            no_face_warning_duration: Duration (seconds) before showing no face warning
        """
        self.person_id = person_id
        self.face_detector = face_detector
        self.preprocessor = preprocessor
        self.emotion_classifier = emotion_classifier
        self.attention_detector = attention_detector
        self.appearance_coordinator = appearance_coordinator
        
        # Create score calculator if not provided
        if score_calculator is None:
            self.score_calculator = ScoreCalculator()
        else:
            self.score_calculator = score_calculator
        
        self.low_confidence_threshold = low_confidence_threshold
        self.no_face_warning_duration = no_face_warning_duration
        
        # History storage
        self.emotion_history: List[EmotionData] = []
        self.attention_history: List[float] = []
        self.attention_timestamps: List[float] = []
        
        # Frame tracking
        self.frame_count = 0
        
        # No face detection tracking
        self.no_face_start_time: Optional[float] = None
        self.last_face_time: Optional[float] = None
        
        # Statistics
        self.total_frames = 0
        self.face_detected_frames = 0
        self.low_confidence_frames = 0
        
        print(f"PersonAnalyzer initialized for {person_id}")
    
    def process_frame(self, frame: np.ndarray) -> PersonResult:
        """Process one frame for this person.
        
        Pipeline:
        1. Detect face in frame
        2. If face detected:
           - Classify emotion
           - Calculate attention
           - Update histories
           - Calculate scores
        3. Return PersonResult with all data
        
        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format
        
        Returns:
            PersonResult with detection, emotion, attention, and scores
        """
        self.frame_count += 1
        self.total_frames += 1
        current_time = time.time()
        
        # Initialize result
        result = PersonResult(
            person_id=self.person_id,
            frame_number=self.frame_count,
            timestamp=current_time,
            face_detected=False
        )
        
        # Detect faces (with caching support)
        try:
            # Check cache first if enabled
            if hasattr(self, 'enable_caching') and self.enable_caching:
                detections = self._get_cached_detections(frame)
                if detections is None:
                    detections = self.face_detector.detect_faces(frame)
                    self._cache_detections(detections)
            else:
                detections = self.face_detector.detect_faces(frame)
        except Exception as e:
            print(f"[{self.person_id}] Face detection error: {e}")
            detections = []
        
        # Handle no face detected
        if not detections:
            self._handle_no_face(result, current_time)
            return result
        
        # Use the largest face (most confident) detection
        # For secondary user (screen capture), this ensures we track the main person
        detection = self._select_best_detection(detections)
        
        # Update face detection tracking
        self.face_detected_frames += 1
        self.last_face_time = current_time
        self.no_face_start_time = None
        
        # Update result with detection info
        result.face_detected = True
        result.face_bbox = detection.bbox
        result.face_landmarks = detection.landmarks
        result.no_face_warning = False
        
        # Process emotion
        try:
            emotion_result = self._process_emotion(frame, detection, current_time)
            result.emotion = emotion_result['emotion']
            result.emotion_confidence = emotion_result['confidence']
            result.emotion_probabilities = emotion_result['probabilities']
            result.low_confidence = emotion_result['low_confidence']
            
            # Track low confidence
            if result.low_confidence:
                self.low_confidence_frames += 1
        except Exception as e:
            print(f"[{self.person_id}] Emotion processing error: {e}")
        
        # Process attention
        try:
            if detection.landmarks is not None:
                attention_result = self._process_attention(
                    detection.landmarks,
                    frame.shape[:2]
                )
                result.attention_score = attention_result['score']
                result.attention_level = attention_result['level']
                result.attention_details = attention_result['details']
        except Exception as e:
            print(f"[{self.person_id}] Attention processing error: {e}")
        
        # Process appearance assessment if enabled
        try:
            if self.appearance_coordinator is not None and result.face_detected:
                appearance_assessment = self.appearance_coordinator.assess_appearance(
                    frame,
                    detection.bbox
                )
                result.appearance = appearance_assessment
        except Exception as e:
            print(f"[{self.person_id}] Appearance assessment error: {e}")
        
        # Calculate scores
        result.emotion_score = self.calculate_emotion_score()
        result.attention_score_avg = self.calculate_attention_score()
        
        return result
    
    def _handle_no_face(self, result: PersonResult, current_time: float):
        """Handle case when no face is detected.
        
        Args:
            result: PersonResult to update
            current_time: Current timestamp
        """
        # Start tracking no face duration
        if self.no_face_start_time is None:
            self.no_face_start_time = current_time
        
        # Check if warning should be shown
        no_face_duration = current_time - self.no_face_start_time
        if no_face_duration >= self.no_face_warning_duration:
            result.no_face_warning = True
        
        # Set scores to current averages (don't update with new data)
        result.emotion_score = self.calculate_emotion_score()
        result.attention_score_avg = self.calculate_attention_score()
    
    def _process_emotion(
        self,
        frame: np.ndarray,
        detection,
        current_time: float
    ) -> Dict:
        """Process emotion for detected face.
        
        Args:
            frame: Input frame
            detection: FaceDetection object
            current_time: Current timestamp
        
        Returns:
            Dictionary with emotion results
        """
        # Preprocess face
        face_tensor = self.preprocessor.preprocess(frame, detection)
        
        # Classify emotion
        prediction = self.emotion_classifier.predict(face_tensor)
        
        # Check confidence
        low_confidence = prediction.confidence < self.low_confidence_threshold
        
        # Add to history only if confidence is acceptable
        if not low_confidence:
            emotion_data = EmotionData(
                timestamp=current_time,
                emotion=prediction.emotion,
                confidence=prediction.confidence,
                frame_number=self.frame_count
            )
            self.emotion_history.append(emotion_data)
        
        return {
            'emotion': prediction.emotion,
            'confidence': prediction.confidence,
            'probabilities': prediction.probabilities,
            'low_confidence': low_confidence
        }
    
    def _process_attention(
        self,
        landmarks: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> Dict:
        """Process attention for detected face.
        
        Args:
            landmarks: Facial landmarks
            frame_shape: Frame shape (height, width)
        
        Returns:
            Dictionary with attention results
        """
        # Calculate attention score - có landmarks = có khuôn mặt
        face_detected = landmarks is not None
        score, details = self.attention_detector.calculate_attention_score(
            landmarks,
            frame_shape,
            face_detected=face_detected
        )
        
        # Get attention level
        level = self.attention_detector.get_attention_level(score)
        
        # Add to history
        current_time = time.time()
        self.attention_history.append(score)
        self.attention_timestamps.append(current_time)
        
        return {
            'score': score,
            'level': level,
            'details': details
        }
    
    def get_emotion_history(self, window_seconds: int = 60) -> List[EmotionData]:
        """Get emotion history for specified window.
        
        Args:
            window_seconds: Window size in seconds
        
        Returns:
            List of EmotionData within the window
        """
        if not self.emotion_history:
            return []
        
        current_time = time.time()
        window_start = current_time - window_seconds
        
        return [
            data for data in self.emotion_history
            if data.timestamp >= window_start
        ]
    
    def get_attention_history(self, window_seconds: int = 60) -> List[float]:
        """Get attention scores for specified window.
        
        Args:
            window_seconds: Window size in seconds
        
        Returns:
            List of attention scores within the window
        """
        if not self.attention_history:
            return []
        
        current_time = time.time()
        window_start = current_time - window_seconds
        
        windowed_scores = []
        for score, timestamp in zip(self.attention_history, self.attention_timestamps):
            if timestamp >= window_start:
                windowed_scores.append(score)
        
        return windowed_scores
    
    def calculate_emotion_score(self) -> float:
        """Calculate emotion score (0-100) from history.
        
        Uses ScoreCalculator to compute weighted emotion score
        based on recent emotion detections.
        
        Returns:
            Emotion score in range 0-100
        """
        return self.score_calculator.calculate_emotion_score(self.emotion_history)
    
    def calculate_attention_score(self) -> float:
        """Calculate attention score (0-100) from history.
        
        Uses ScoreCalculator to compute average attention score
        based on recent attention measurements.
        
        Returns:
            Attention score in range 0-100
        """
        return self.score_calculator.calculate_attention_score(
            self.attention_history,
            self.attention_timestamps
        )
    
    def get_statistics(self) -> Dict:
        """Get statistics for this person.
        
        Returns:
            Dictionary with analysis statistics
        """
        face_detection_rate = 0.0
        if self.total_frames > 0:
            face_detection_rate = self.face_detected_frames / self.total_frames
        
        low_confidence_rate = 0.0
        if self.face_detected_frames > 0:
            low_confidence_rate = self.low_confidence_frames / self.face_detected_frames
        
        return {
            'person_id': self.person_id,
            'total_frames': self.total_frames,
            'face_detected_frames': self.face_detected_frames,
            'face_detection_rate': face_detection_rate,
            'low_confidence_frames': self.low_confidence_frames,
            'low_confidence_rate': low_confidence_rate,
            'emotion_history_size': len(self.emotion_history),
            'attention_history_size': len(self.attention_history),
            'current_emotion_score': self.calculate_emotion_score(),
            'current_attention_score': self.calculate_attention_score()
        }
    
    def reset(self):
        """Reset analyzer state.
        
        Clears all history and statistics. Useful when starting
        a new session or switching modes.
        """
        # Clear histories
        self.emotion_history.clear()
        self.attention_history.clear()
        self.attention_timestamps.clear()
        
        # Reset counters
        self.frame_count = 0
        self.total_frames = 0
        self.face_detected_frames = 0
        self.low_confidence_frames = 0
        
        # Reset tracking
        self.no_face_start_time = None
        self.last_face_time = None
        
        # Reset score calculator
        self.score_calculator.reset()
        
        # Reset attention detector
        self.attention_detector.reset()
        
        print(f"PersonAnalyzer reset for {self.person_id}")
    
    def _select_best_detection(self, detections: List) -> object:
        """Select the best face detection from multiple detections.
        
        For primary user: Use the most confident detection
        For secondary user: Use the largest face (main person on screen)
        
        Args:
            detections: List of face detections
        
        Returns:
            Best detection object
        """
        if len(detections) == 1:
            return detections[0]
        
        # For secondary user (screen capture), prefer largest face
        if self.person_id == "secondary":
            largest_detection = max(
                detections,
                key=lambda d: self._calculate_bbox_area(d.bbox)
            )
            return largest_detection
        
        # For primary user, use first (most confident) detection
        return detections[0]
    
    def _calculate_bbox_area(self, bbox: Tuple[int, int, int, int]) -> int:
        """Calculate bounding box area.
        
        Args:
            bbox: Bounding box (x, y, w, h)
        
        Returns:
            Area in pixels
        """
        x, y, w, h = bbox
        return w * h
    
    def _get_cached_detections(self, frame: np.ndarray) -> Optional[List]:
        """Get cached face detections if available.
        
        Args:
            frame: Input frame
        
        Returns:
            Cached detections or None if cache miss
        """
        if not hasattr(self, 'detection_cache'):
            return None
        
        current_time = time.time()
        
        # Check if cache is still valid
        if hasattr(self, 'detection_cache_time'):
            if current_time - self.detection_cache_time < self.cache_ttl:
                if hasattr(self, 'cache_stats'):
                    self.cache_stats['hits'] += 1
                return self.detection_cache
        
        if hasattr(self, 'cache_stats'):
            self.cache_stats['misses'] += 1
        return None
    
    def _cache_detections(self, detections: List):
        """Cache face detections.
        
        Args:
            detections: Detection results to cache
        """
        self.detection_cache = detections
        self.detection_cache_time = time.time()
    
    def enable_detection_caching(self, cache_ttl: float = 0.1):
        """Enable face detection caching.
        
        Args:
            cache_ttl: Cache time-to-live in seconds
        """
        self.enable_caching = True
        self.cache_ttl = cache_ttl
        self.detection_cache = None
        self.detection_cache_time = 0.0
        self.cache_stats = {'hits': 0, 'misses': 0}
        print(f"[{self.person_id}] Face detection caching enabled (TTL: {cache_ttl}s)")
    
    def disable_detection_caching(self):
        """Disable face detection caching."""
        self.enable_caching = False
        print(f"[{self.person_id}] Face detection caching disabled")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache hit/miss stats
        """
        if not hasattr(self, 'cache_stats'):
            return {'hits': 0, 'misses': 0, 'hit_rate': 0.0}
        
        total = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total if total > 0 else 0.0
        
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': hit_rate
        }
