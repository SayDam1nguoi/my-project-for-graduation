"""
Video Preprocessing Pipeline Module

This module provides a comprehensive preprocessing pipeline for video emotion detection,
including normalization, quality filtering, stabilization, lighting correction, alignment,
and temporal smoothing.
"""

from .config import PreprocessingConfig, load_preprocessing_config
from .pipeline import VideoPreprocessingPipeline
from .normalizer import VideoNormalizer
from .quality_filter import FrameQualityFilter
from .stabilizer import FaceStabilizer
from .lighting_corrector import LightingCorrector
from .aligner import FaceAligner
from .temporal_smoother import TemporalSmoother
from .data_models import (
    FrameMetadata, FaceData, ProcessedFrame, EmotionPrediction,
    PipelineStatistics, VideoAnalysisResult,
    PreprocessingError, InvalidFrameError, InsufficientQualityError
)
from .utils import (
    calculate_angle_between_points, calculate_distance, expand_bbox,
    ensure_bbox_in_bounds, blend_images, normalize_landmarks,
    denormalize_landmarks, calculate_iou, draw_landmarks, draw_bbox,
    format_time, validate_frame, validate_landmarks
)

__all__ = [
    # Configuration
    'PreprocessingConfig',
    'load_preprocessing_config',
    
    # Main Pipeline
    'VideoPreprocessingPipeline',
    
    # Preprocessing Modules
    'VideoNormalizer',
    'FrameQualityFilter',
    'FaceStabilizer',
    'LightingCorrector',
    'FaceAligner',
    'TemporalSmoother',
    
    # Data Models
    'FrameMetadata',
    'FaceData',
    'ProcessedFrame',
    'EmotionPrediction',
    'PipelineStatistics',
    'VideoAnalysisResult',
    
    # Exceptions
    'PreprocessingError',
    'InvalidFrameError',
    'InsufficientQualityError',
    
    # Utilities
    'calculate_angle_between_points',
    'calculate_distance',
    'expand_bbox',
    'ensure_bbox_in_bounds',
    'blend_images',
    'normalize_landmarks',
    'denormalize_landmarks',
    'calculate_iou',
    'draw_landmarks',
    'draw_bbox',
    'format_time',
    'validate_frame',
    'validate_landmarks',
]
