"""
Data models for Recruitment Emotion Scoring System.

This module contains dataclasses for storing and managing emotion scoring data,
including reports, criterion scores, facial data, and configuration.

Requirements: 13.2, 14.1
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import json
import numpy as np


@dataclass
class BoundingBox:
    """Bounding box coordinates for face detection."""
    x: int
    y: int
    width: int
    height: int
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'BoundingBox':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class MicroExpression:
    """
    Micro-expression detection data.
    
    Represents a brief (<500ms) emotional expression that may indicate
    suppressed or hidden emotions.
    
    Attributes:
        start_frame: Starting frame number
        end_frame: Ending frame number
        duration: Duration in seconds
        expression_type: Type of micro-expression (e.g., 'fear', 'disgust')
        intensity: Intensity level (0-1)
        confidence: Detection confidence (0-1)
    """
    start_frame: int
    end_frame: int
    duration: float
    expression_type: str
    intensity: float
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MicroExpression':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PostureData:
    """
    Posture analysis data for confidence assessment.
    
    Contains indicators of body posture and positioning that contribute
    to confidence scoring.
    
    Attributes:
        head_tilt: Head tilt angle in degrees (0 = upright, positive = tilted)
        shoulder_alignment: Shoulder alignment score (0-1, 1 = well-aligned)
        body_orientation: Body orientation relative to camera (0-1, 1 = facing camera)
        posture_confidence: Overall posture confidence score (0-1)
    """
    head_tilt: float
    shoulder_alignment: float
    body_orientation: float
    posture_confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PostureData':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class AttentionData:
    """
    Attention indicators for engagement assessment.
    
    Contains indicators of attention and engagement level during the interview.
    
    Attributes:
        eye_contact_score: Eye contact quality score (0-1)
        head_movement_frequency: Frequency of head movements (movements per second)
        facial_responsiveness: Facial expression change rate (0-1)
        attention_score: Overall attention score (0-1)
    """
    eye_contact_score: float
    head_movement_frequency: float
    facial_responsiveness: float
    attention_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AttentionData':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class FacialData:
    """
    Facial analysis data for a single frame.
    
    Contains comprehensive facial feature data including landmarks,
    emotions, action units, gaze direction, and head pose.
    
    Attributes:
        frame_number: Frame number in video
        timestamp: Timestamp in seconds
        face_bbox: Face bounding box
        landmarks: Facial landmarks as numpy array (serialized as list)
        emotion_probabilities: Probability distribution over emotions
        action_units: Facial action unit activations
        gaze_direction: Gaze direction as (x, y) tuple
        head_pose: Head pose as (pitch, yaw, roll) tuple in degrees
        posture_data: Posture indicators for confidence (optional)
        attention_data: Attention indicators for engagement (optional)
    """
    frame_number: int
    timestamp: float
    face_bbox: BoundingBox
    landmarks: np.ndarray
    emotion_probabilities: Dict[str, float]
    action_units: Dict[str, float]
    gaze_direction: Tuple[float, float]
    head_pose: Tuple[float, float, float]
    posture_data: Optional['PostureData'] = None
    attention_data: Optional['AttentionData'] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            'frame_number': self.frame_number,
            'timestamp': self.timestamp,
            'face_bbox': self.face_bbox.to_dict(),
            'landmarks': self.landmarks.tolist() if isinstance(self.landmarks, np.ndarray) else self.landmarks,
            'emotion_probabilities': self.emotion_probabilities,
            'action_units': self.action_units,
            'gaze_direction': list(self.gaze_direction),
            'head_pose': list(self.head_pose),
            'posture_data': self.posture_data.to_dict() if self.posture_data else None,
            'attention_data': self.attention_data.to_dict() if self.attention_data else None,
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FacialData':
        """Create from dictionary."""
        return cls(
            frame_number=data['frame_number'],
            timestamp=data['timestamp'],
            face_bbox=BoundingBox.from_dict(data['face_bbox']),
            landmarks=np.array(data['landmarks']),
            emotion_probabilities=data['emotion_probabilities'],
            action_units=data['action_units'],
            gaze_direction=tuple(data['gaze_direction']),
            head_pose=tuple(data['head_pose']),
            posture_data=PostureData.from_dict(data['posture_data']) if data.get('posture_data') else None,
            attention_data=AttentionData.from_dict(data['attention_data']) if data.get('attention_data') else None,
        )


@dataclass
class KeyMoment:
    """
    Key moment in the video interview.
    
    Represents a significant timestamp with associated emotional data
    that provides evidence for scoring decisions.
    
    Attributes:
        timestamp: Timestamp in seconds
        frame_number: Frame number
        description: Description of the moment
        criterion: Associated criterion (e.g., 'calmness', 'confidence')
        emotion: Detected emotion at this moment
        intensity: Emotion intensity (0-1)
        video_clip_path: Optional path to extracted video clip
    """
    timestamp: float
    frame_number: int
    description: str
    criterion: str
    emotion: str
    intensity: float
    video_clip_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KeyMoment':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CriterionScore:
    """
    Score for a single emotion criterion.
    
    Contains the score, confidence, explanation, and supporting evidence
    for one of the 4 core emotional criteria.
    
    Attributes:
        criterion_name: Name of the criterion (e.g., 'confidence', 'positivity', 'professionalism', 'engagement')
        score: Score value (0-10)
        confidence: Confidence in the score (0-1)
        explanation: Human-readable explanation of the score
        evidence_timestamps: List of timestamps providing evidence
        supporting_data: Additional data supporting the score
        weight: Weight of this criterion in total score calculation (0-1)
    """
    criterion_name: str
    score: float
    confidence: float
    explanation: str
    evidence_timestamps: List[float] = field(default_factory=list)
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    weight: float = 0.0
    
    def __post_init__(self):
        """Validate score range."""
        if not 0 <= self.score <= 10:
            raise ValueError(f"Score must be between 0 and 10, got {self.score}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if not 0 <= self.weight <= 1:
            raise ValueError(f"Weight must be between 0 and 1, got {self.weight}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CriterionScore':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class EmotionReport:
    """
    Comprehensive emotion scoring report for a video interview.
    
    Contains all criterion scores, total score, key moments, and metadata
    for a complete candidate assessment.
    
    Attributes:
        candidate_id: Unique identifier for the candidate
        video_path: Path to the analyzed video file
        timestamp: Report generation timestamp
        criterion_scores: Dictionary mapping criterion names to scores
        total_score: Weighted average total score (0-10)
        key_moments: List of significant moments in the interview
        metadata: Additional metadata (video duration, frame count, etc.)
    """
    candidate_id: str
    video_path: str
    timestamp: datetime
    criterion_scores: Dict[str, CriterionScore]
    total_score: float
    key_moments: List[KeyMoment] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate total score range."""
        if not 0 <= self.total_score <= 10:
            raise ValueError(f"Total score must be between 0 and 10, got {self.total_score}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'candidate_id': self.candidate_id,
            'video_path': self.video_path,
            'timestamp': self.timestamp.isoformat(),
            'criterion_scores': {
                name: score.to_dict() 
                for name, score in self.criterion_scores.items()
            },
            'total_score': self.total_score,
            'key_moments': [moment.to_dict() for moment in self.key_moments],
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionReport':
        """Create from dictionary."""
        return cls(
            candidate_id=data['candidate_id'],
            video_path=data['video_path'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            criterion_scores={
                name: CriterionScore.from_dict(score_data)
                for name, score_data in data['criterion_scores'].items()
            },
            total_score=data['total_score'],
            key_moments=[KeyMoment.from_dict(m) for m in data['key_moments']],
            metadata=data['metadata'],
        )
    
    def to_json(self, indent: int = 2) -> str:
        """
        Serialize to JSON string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EmotionReport':
        """
        Deserialize from JSON string.
        
        Args:
            json_str: JSON string
            
        Returns:
            EmotionReport instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_criterion_score(self, criterion_name: str) -> Optional[CriterionScore]:
        """
        Get score for a specific criterion.
        
        Args:
            criterion_name: Name of the criterion
            
        Returns:
            CriterionScore if found, None otherwise
        """
        return self.criterion_scores.get(criterion_name)
    
    def get_all_scores(self) -> Dict[str, float]:
        """
        Get all criterion scores as a simple dictionary.
        
        Returns:
            Dictionary mapping criterion names to score values
        """
        return {name: score.score for name, score in self.criterion_scores.items()}


@dataclass
class EmotionState:
    """
    Emotional state at a specific point in time.
    
    Represents the emotional state of a candidate at a particular frame,
    including the dominant emotion, intensity, and timestamp.
    
    Attributes:
        frame_number: Frame number in video
        timestamp: Timestamp in seconds
        emotion: Dominant emotion name
        intensity: Emotion intensity (0-1)
        emotion_probabilities: Full probability distribution over emotions
    """
    frame_number: int
    timestamp: float
    emotion: str
    intensity: float
    emotion_probabilities: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionState':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ConversationContext:
    """
    Context information about the conversation/interview.
    
    Provides contextual information about the interview that can be used
    to evaluate appropriateness of emotional responses.
    
    Attributes:
        question_timestamps: List of timestamps when questions were asked
        question_types: Types of questions (e.g., 'technical', 'behavioral', 'casual')
        expected_emotions: Expected emotional responses for each question type
        interviewer_tone: Tone of the interviewer (e.g., 'formal', 'casual', 'friendly')
        interview_phase: Phase of interview (e.g., 'introduction', 'technical', 'closing')
    """
    question_timestamps: List[float] = field(default_factory=list)
    question_types: List[str] = field(default_factory=list)
    expected_emotions: Dict[str, List[str]] = field(default_factory=dict)
    interviewer_tone: str = 'formal'
    interview_phase: str = 'general'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationContext':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def default(cls) -> 'ConversationContext':
        """
        Create default conversation context.
        
        Returns:
            Default ConversationContext with standard interview expectations
        """
        return cls(
            question_timestamps=[],
            question_types=[],
            expected_emotions={
                'technical': ['neutral', 'happy'],
                'behavioral': ['happy', 'neutral'],
                'casual': ['happy'],
                'challenging': ['neutral', 'surprise'],
            },
            interviewer_tone='formal',
            interview_phase='general',
        )


@dataclass
class ScoringConfig:
    """
    Configuration for emotion scoring system.
    
    Defines weights, thresholds, and feature toggles for the scoring engine.
    
    Attributes:
        criterion_weights: Weight for each criterion (must sum to 1.0)
        score_thresholds: (low, high) thresholds for each criterion
        enable_micro_expression_analysis: Enable micro-expression detection
        enable_gaze_tracking: Enable gaze tracking
        frame_sampling_rate: Process every Nth frame (1 = all frames)
    """
    criterion_weights: Dict[str, float]
    score_thresholds: Dict[str, Tuple[float, float]]
    enable_micro_expression_analysis: bool = True
    enable_gaze_tracking: bool = True
    frame_sampling_rate: int = 1
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate weights sum to 1.0 (with small tolerance for floating point)
        weight_sum = sum(self.criterion_weights.values())
        if not (0.99 <= weight_sum <= 1.01):
            raise ValueError(
                f"Criterion weights must sum to 1.0, got {weight_sum:.4f}"
            )
        
        # Validate all weights are non-negative
        for criterion, weight in self.criterion_weights.items():
            if weight < 0:
                raise ValueError(
                    f"Weight for {criterion} must be non-negative, got {weight}"
                )
        
        # Validate thresholds
        for criterion, (low, high) in self.score_thresholds.items():
            if not (0 <= low <= high <= 10):
                raise ValueError(
                    f"Invalid thresholds for {criterion}: ({low}, {high}). "
                    f"Must satisfy 0 <= low <= high <= 10"
                )
        
        # Validate frame sampling rate
        if self.frame_sampling_rate < 1:
            raise ValueError(
                f"Frame sampling rate must be >= 1, got {self.frame_sampling_rate}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'criterion_weights': self.criterion_weights,
            'score_thresholds': {
                name: list(threshold) 
                for name, threshold in self.score_thresholds.items()
            },
            'enable_micro_expression_analysis': self.enable_micro_expression_analysis,
            'enable_gaze_tracking': self.enable_gaze_tracking,
            'frame_sampling_rate': self.frame_sampling_rate,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScoringConfig':
        """Create from dictionary."""
        return cls(
            criterion_weights=data['criterion_weights'],
            score_thresholds={
                name: tuple(threshold)
                for name, threshold in data['score_thresholds'].items()
            },
            enable_micro_expression_analysis=data.get('enable_micro_expression_analysis', True),
            enable_gaze_tracking=data.get('enable_gaze_tracking', True),
            frame_sampling_rate=data.get('frame_sampling_rate', 1),
        )
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ScoringConfig':
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def default(cls) -> 'ScoringConfig':
        """
        Create default configuration with 4-criteria system and specified weights.
        
        Returns:
            Default ScoringConfig instance with 4 core criteria
        """
        # 4 core criteria with specified weights
        criterion_weights = {
            'confidence': 0.325,      # 32.5% - Voice tone, posture, eye contact, facial expressions
            'positivity': 0.275,      # 27.5% - Smiles, warm expressions, friendly demeanor
            'professionalism': 0.225, # 22.5% - Composure, emotional control, appropriate behavior
            'engagement': 0.175,      # 17.5% - Eye contact, head movements, facial responsiveness, attention
        }
        
        # Standard thresholds: low=4.0, high=7.0
        # Scores 0-4: Low, 4-7: Medium, 7-10: High
        score_thresholds = {
            'confidence': (4.0, 7.0),
            'positivity': (4.0, 7.0),
            'professionalism': (4.0, 7.0),
            'engagement': (4.0, 7.0),
        }
        
        return cls(
            criterion_weights=criterion_weights,
            score_thresholds=score_thresholds,
            enable_micro_expression_analysis=True,
            enable_gaze_tracking=True,
            frame_sampling_rate=1,
        )
