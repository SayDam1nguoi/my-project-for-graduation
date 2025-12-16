"""Quality analysis data models."""

from dataclasses import dataclass, field
from typing import List
from datetime import datetime


@dataclass
class AudioQualityReport:
    """Audio quality analysis report."""
    
    clarity_score: float  # 0-1, based on SNR and spectral clarity
    snr: float  # Signal-to-noise ratio in dB
    rms_level: float  # Root mean square level
    zero_crossing_rate: float  # Zero crossing rate
    spectral_centroid: float  # Spectral centroid in Hz
    is_silent: bool  # Whether audio is silent
    has_speech: bool  # Whether audio contains speech
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate audio quality report data."""
        if not 0.0 <= self.clarity_score <= 1.0:
            raise ValueError("Clarity score must be between 0 and 1")


@dataclass
class TranscriptionQualityReport:
    """Transcription quality analysis report."""
    
    fluency_score: float  # 0-1, based on speech rate and pauses
    confidence_avg: float  # Average confidence score
    confidence_min: float  # Minimum confidence score
    confidence_max: float  # Maximum confidence score
    speech_rate: float  # Words per minute
    hallucination_risk: float  # 0-1, likelihood of hallucinations
    segment_count: int  # Number of segments
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate transcription quality report data."""
        if not 0.0 <= self.fluency_score <= 1.0:
            raise ValueError("Fluency score must be between 0 and 1")
        if not 0.0 <= self.confidence_avg <= 1.0:
            raise ValueError("Average confidence must be between 0 and 1")
        if not 0.0 <= self.hallucination_risk <= 1.0:
            raise ValueError("Hallucination risk must be between 0 and 1")


@dataclass
class QualityReport:
    """Combined quality analysis report."""
    
    audio_quality: AudioQualityReport
    transcription_quality: TranscriptionQualityReport
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def overall_score(self) -> float:
        """Calculate overall quality score (0-1)."""
        return (
            self.audio_quality.clarity_score * 0.4 +
            self.transcription_quality.fluency_score * 0.3 +
            self.transcription_quality.confidence_avg * 0.3
        )
