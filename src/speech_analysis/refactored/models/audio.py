"""Audio-related data models."""

from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np


@dataclass
class AudioDevice:
    """Audio input device information."""
    
    index: int
    name: str
    max_input_channels: int
    default_sample_rate: float
    is_default: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioChunk:
    """Audio data chunk with metadata."""
    
    data: np.ndarray
    sample_rate: int
    timestamp: float
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate audio chunk data."""
        if not isinstance(self.data, np.ndarray):
            raise TypeError("Audio data must be a numpy array")
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.duration <= 0:
            raise ValueError("Duration must be positive")


@dataclass
class SpeechSegment:
    """Speech segment detected by VAD."""
    
    audio: np.ndarray
    start_time: float
    end_time: float
    confidence: float
    is_speech: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end_time - self.start_time
    
    def __post_init__(self):
        """Validate speech segment data."""
        if not isinstance(self.audio, np.ndarray):
            raise TypeError("Audio data must be a numpy array")
        if self.start_time < 0:
            raise ValueError("Start time must be non-negative")
        if self.end_time <= self.start_time:
            raise ValueError("End time must be greater than start time")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
