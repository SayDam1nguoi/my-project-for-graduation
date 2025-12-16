"""
Data Models for Video Preprocessing Pipeline

This module defines data classes for frame metadata, face data, processed frames,
and analysis results.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import numpy as np


@dataclass
class FrameMetadata:
    """Metadata for a video frame."""
    frame_number: int
    timestamp: float
    fps: float
    resolution: Tuple[int, int]  # (width, height)
    
    def __repr__(self) -> str:
        return (f"FrameMetadata(frame={self.frame_number}, "
                f"time={self.timestamp:.2f}s, fps={self.fps})")


@dataclass
class FaceData:
    """Data for a detected face in a frame."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    landmarks: np.ndarray  # Facial landmarks array
    confidence: float
    pose_angles: Optional[Tuple[float, float, float]] = None  # (yaw, pitch, roll)
    
    def __repr__(self) -> str:
        return (f"FaceData(bbox={self.bbox}, confidence={self.confidence:.2f}, "
                f"landmarks_count={len(self.landmarks)})")


@dataclass
class ProcessedFrame:
    """Result of processing a single frame."""
    frame: np.ndarray
    frame_metadata: FrameMetadata
    face_data: Optional[FaceData]
    is_valid: bool
    rejection_reason: Optional[str] = None
    preprocessing_time_ms: float = 0.0
    applied_steps: List[str] = field(default_factory=list)
    
    def __repr__(self) -> str:
        status = "valid" if self.is_valid else f"rejected ({self.rejection_reason})"
        return (f"ProcessedFrame(frame={self.frame_metadata.frame_number}, "
                f"status={status}, time={self.preprocessing_time_ms:.1f}ms)")


@dataclass
class EmotionPrediction:
    """Emotion prediction for a frame."""
    frame_number: int
    timestamp: float
    emotion_label: str
    confidence: float
    probabilities: np.ndarray  # Probabilities for all emotion classes
    is_smoothed: bool = False
    
    def __repr__(self) -> str:
        return (f"EmotionPrediction(frame={self.frame_number}, "
                f"emotion={self.emotion_label}, confidence={self.confidence:.2f})")


@dataclass
class PipelineStatistics:
    """Statistics for preprocessing pipeline execution."""
    total_frames: int
    valid_frames: int
    filtered_frames: int
    filter_reasons: Dict[str, int] = field(default_factory=dict)
    avg_processing_time_ms: float = 0.0
    fps: float = 0.0
    step_timings: Dict[str, float] = field(default_factory=dict)
    
    @property
    def filter_rate(self) -> float:
        """Calculate frame filter rate."""
        if self.total_frames == 0:
            return 0.0
        return self.filtered_frames / self.total_frames
    
    @property
    def valid_rate(self) -> float:
        """Calculate valid frame rate."""
        if self.total_frames == 0:
            return 0.0
        return self.valid_frames / self.total_frames
    
    def __repr__(self) -> str:
        return (f"PipelineStatistics(total={self.total_frames}, "
                f"valid={self.valid_frames}, filtered={self.filtered_frames}, "
                f"fps={self.fps:.1f})")


@dataclass
class VideoAnalysisResult:
    """Complete result of video analysis."""
    video_path: str
    total_frames: int
    processed_frames: int
    filtered_frames: int
    emotion_predictions: List[EmotionPrediction]
    dominant_emotion: str
    statistics: PipelineStatistics
    processing_time: float
    emotion_distribution: Dict[str, float] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (f"VideoAnalysisResult(video={self.video_path}, "
                f"frames={self.processed_frames}/{self.total_frames}, "
                f"dominant={self.dominant_emotion}, time={self.processing_time:.1f}s)")
    
    def get_emotion_timeline(self) -> List[Tuple[float, str, float]]:
        """
        Get emotion timeline as list of (timestamp, emotion, confidence).
        
        Returns:
            List of tuples (timestamp, emotion_label, confidence)
        """
        return [
            (pred.timestamp, pred.emotion_label, pred.confidence)
            for pred in self.emotion_predictions
        ]
    
    def calculate_emotion_distribution(self) -> Dict[str, float]:
        """
        Calculate distribution of emotions across video.
        
        Returns:
            Dictionary mapping emotion labels to their frequency
        """
        if not self.emotion_predictions:
            return {}
        
        emotion_counts: Dict[str, int] = {}
        for pred in self.emotion_predictions:
            emotion_counts[pred.emotion_label] = emotion_counts.get(pred.emotion_label, 0) + 1
        
        total = len(self.emotion_predictions)
        distribution = {
            emotion: count / total
            for emotion, count in emotion_counts.items()
        }
        
        return distribution


class PreprocessingError(Exception):
    """Base exception for preprocessing errors."""
    pass


class InvalidFrameError(PreprocessingError):
    """Frame cannot be processed."""
    pass


class InsufficientQualityError(PreprocessingError):
    """Too many frames filtered, cannot proceed."""
    pass


class ConfigurationError(PreprocessingError):
    """Invalid configuration."""
    pass
