"""Transcription-related data models."""

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class TranscriptionSegment:
    """Single transcription segment with timing and confidence."""
    
    text: str
    start_time: float
    end_time: float
    confidence: float
    language: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end_time - self.start_time
    
    @property
    def word_count(self) -> int:
        """Get word count in segment."""
        return len(self.text.split())
    
    def __post_init__(self):
        """Validate transcription segment data."""
        if self.start_time < 0:
            raise ValueError("Start time must be non-negative")
        if self.end_time <= self.start_time:
            raise ValueError("End time must be greater than start time")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class TranscriptionResult:
    """Complete transcription result with metadata."""
    
    text: str
    segments: List[TranscriptionSegment]
    language: str
    confidence: float
    processing_time: float
    engine_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def segment_count(self) -> int:
        """Get number of segments."""
        return len(self.segments)
    
    @property
    def total_duration(self) -> float:
        """Get total duration of all segments."""
        if not self.segments:
            return 0.0
        return self.segments[-1].end_time - self.segments[0].start_time
    
    @property
    def average_confidence(self) -> float:
        """Get average confidence across all segments."""
        if not self.segments:
            return 0.0
        return sum(seg.confidence for seg in self.segments) / len(self.segments)
    
    def __post_init__(self):
        """Validate transcription result data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        if self.processing_time < 0:
            raise ValueError("Processing time must be non-negative")
        if not self.engine_name:
            raise ValueError("Engine name must be specified")
