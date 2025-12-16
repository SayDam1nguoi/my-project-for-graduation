# -*- coding: utf-8 -*-
"""
Video Analysis Module - HireVue-Style Analysis

This module provides advanced video analysis features for emotion detection,
including Facial Action Units, Emotion Intensity, and Micro-Expression detection.

These components are ONLY used for video mode, NOT for camera mode.
"""

from .facial_action_units import FacialActionUnitDetector
from .emotion_intensity import EmotionIntensityAnalyzer
from .micro_expression import MicroExpressionDetector
from .video_preprocessor import VideoPreprocessor
from .dual_person_models import (
    EmotionData,
    PersonResult,
    DualPersonResult,
    ComparisonResult,
    SessionReport
)
from .score_calculator import ScoreCalculator
from .comparison_engine import ComparisonEngine
from .video_validator import VideoValidator, ValidationResult, AudioTrackInfo, QualityMetrics
from .audio_extractor import AudioExtractor, AudioExtractionError
from .language_detector import (
    LanguageDetector,
    LanguageDetectionResult,
    LanguageSegment,
    LanguageDetectionError
)
from .speaker_diarization import (
    SpeakerDiarization,
    SpeakerSegment
)

__all__ = [
    'FacialActionUnitDetector',
    'EmotionIntensityAnalyzer',
    'MicroExpressionDetector',
    'VideoPreprocessor',
    'EmotionData',
    'PersonResult',
    'DualPersonResult',
    'ComparisonResult',
    'SessionReport',
    'ScoreCalculator',
    'ComparisonEngine',
    'VideoValidator',
    'ValidationResult',
    'AudioTrackInfo',
    'QualityMetrics',
    'AudioExtractor',
    'AudioExtractionError',
    'LanguageDetector',
    'LanguageDetectionResult',
    'LanguageSegment',
    'LanguageDetectionError',
    'SpeakerDiarization',
    'SpeakerSegment',
]

__version__ = '1.0.0'
