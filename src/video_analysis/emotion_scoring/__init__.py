"""
Recruitment Emotion Scoring System

This module provides AI-powered emotion analysis and scoring for video interviews.
It evaluates 12 distinct emotional criteria to provide objective candidate assessments.
"""

from .models import (
    EmotionReport,
    CriterionScore,
    FacialData,
    MicroExpression,
    ScoringConfig,
    KeyMoment,
    BoundingBox,
    ConversationContext,
    EmotionState,
)

from .config import (
    load_config_from_yaml,
    save_config_to_yaml,
    load_default_config,
    get_default_config_path,
    validate_config,
)

from .facial_data_extractor import FacialDataExtractor
from .emotion_criteria_analyzer import EmotionCriteriaAnalyzer
from .emotion_scoring_engine import EmotionScoringEngine
from .report_generator import ReportGenerator
from .evidence_retrieval import EvidenceRetrievalSystem, VideoClipMetadata

__all__ = [
    'EmotionReport',
    'CriterionScore',
    'FacialData',
    'MicroExpression',
    'ScoringConfig',
    'KeyMoment',
    'BoundingBox',
    'ConversationContext',
    'EmotionState',
    'FacialDataExtractor',
    'EmotionCriteriaAnalyzer',
    'EmotionScoringEngine',
    'ReportGenerator',
    'EvidenceRetrievalSystem',
    'VideoClipMetadata',
    'load_config_from_yaml',
    'save_config_to_yaml',
    'load_default_config',
    'get_default_config_path',
    'validate_config',
]
