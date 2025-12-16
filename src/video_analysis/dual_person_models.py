"""
Data models for dual person comparison system.

This module contains dataclasses for storing and managing data related to
dual person emotion and attention comparison.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, TYPE_CHECKING
from datetime import datetime
import numpy as np

if TYPE_CHECKING:
    from src.video_analysis.appearance.models import AppearanceAssessment


@dataclass
class EmotionData:
    """Single emotion detection data point.
    
    Stores a single emotion detection result with timestamp for historical tracking.
    Used in rolling window calculations for emotion scores.
    """
    timestamp: float
    emotion: str
    confidence: float
    frame_number: int


@dataclass
class PersonResult:
    """Result from analyzing one person in one frame.
    
    Contains all detection, emotion, and attention results for a single person
    in a single frame. Used by PersonAnalyzer to return comprehensive results.
    """
    person_id: str  # "primary" or "secondary"
    frame_number: int
    timestamp: float
    
    # Detection
    face_detected: bool
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    face_landmarks: Optional[np.ndarray] = None
    
    # Emotion
    emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None
    emotion_probabilities: Optional[Dict[str, float]] = None
    
    # Attention
    attention_score: Optional[float] = None
    attention_level: Optional[str] = None
    attention_details: Optional[Dict] = None
    
    # Scores (0-100)
    emotion_score: float = 0.0
    attention_score_avg: float = 0.0
    
    # Appearance (optional)
    appearance: Optional['AppearanceAssessment'] = None
    
    # Flags
    low_confidence: bool = False
    no_face_warning: bool = False


@dataclass
class DualPersonResult:
    """Combined result from both persons.
    
    Aggregates results from both primary and secondary users for a single frame.
    Includes comparison results when available.
    """
    frame_number: int
    timestamp: float
    
    primary: PersonResult
    secondary: PersonResult
    
    comparison: Optional['ComparisonResult'] = None


@dataclass
class ComparisonResult:
    """Comparison between two persons.
    
    Contains comparison metrics for emotion and attention scores between
    primary and secondary users. Updated at configurable intervals.
    """
    timestamp: float
    
    # Emotion comparison
    primary_emotion_score: float
    secondary_emotion_score: float
    emotion_winner: str  # "primary", "secondary", or "tie"
    emotion_difference: float  # percentage
    significant_emotion_gap: bool
    
    # Attention comparison
    primary_attention_score: float
    secondary_attention_score: float
    attention_winner: str  # "primary", "secondary", or "tie"
    attention_difference: float  # percentage
    significant_attention_gap: bool


@dataclass
class SessionReport:
    """Complete session report for export.
    
    Comprehensive report containing all statistics, timelines, and comparison
    history for an entire dual person comparison session. Used for JSON export.
    """
    session_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    
    # Primary user stats
    primary_stats: Dict = field(default_factory=dict)
    primary_emotion_timeline: List[EmotionData] = field(default_factory=list)
    primary_attention_timeline: List[float] = field(default_factory=list)
    
    # Secondary user stats
    secondary_stats: Dict = field(default_factory=dict)
    secondary_emotion_timeline: List[EmotionData] = field(default_factory=list)
    secondary_attention_timeline: List[float] = field(default_factory=list)
    
    # Comparison history
    comparison_history: List[ComparisonResult] = field(default_factory=list)
    
    # Summary
    overall_emotion_winner: str = ""
    overall_attention_winner: str = ""
    average_emotion_difference: float = 0.0
    average_attention_difference: float = 0.0
