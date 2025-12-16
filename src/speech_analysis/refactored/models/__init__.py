"""Data models for the refactored STT system."""

from .audio import AudioChunk, SpeechSegment, AudioDevice
from .transcription import TranscriptionSegment, TranscriptionResult
from .quality import QualityReport, AudioQualityReport, TranscriptionQualityReport
from .performance import PerformanceMetrics, ResourceStatus
from .config import (
    STTConfig,
    AudioConfig,
    EngineConfig,
    ProcessingConfig,
    PerformanceConfig,
    QualityConfig,
)

__all__ = [
    "AudioChunk",
    "SpeechSegment",
    "AudioDevice",
    "TranscriptionSegment",
    "TranscriptionResult",
    "QualityReport",
    "AudioQualityReport",
    "TranscriptionQualityReport",
    "PerformanceMetrics",
    "ResourceStatus",
    "STTConfig",
    "AudioConfig",
    "EngineConfig",
    "ProcessingConfig",
    "PerformanceConfig",
    "QualityConfig",
]
