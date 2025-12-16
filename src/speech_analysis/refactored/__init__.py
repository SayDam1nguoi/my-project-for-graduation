"""
Refactored Speech-to-Text System

This package contains the refactored STT system with improved architecture,
modularity, and performance. It is designed to achieve 95%+ accuracy for
Vietnamese speech while eliminating hallucinations and maintaining real-time
performance.

Key Features:
- Modular pipeline architecture
- Multiple STT engine support (Whisper, faster-whisper, Vosk)
- Advanced hallucination filtering
- Real-time performance monitoring
- Comprehensive error handling
"""

__version__ = "2.0.0"
__author__ = "Speech Analysis Team"

# Core interfaces
from .interfaces.audio import IAudioCapture, IAudioPreprocessor, IVADDetector
from .interfaces.engine import ISTTEngine, ISTTEngineManager
from .interfaces.processing import (
    IHallucinationFilter,
    ITextPostProcessor,
    IQualityAnalyzer,
)
from .interfaces.monitoring import IPerformanceMonitor
from .interfaces.config import IConfigManager

# Data models
from .models.audio import AudioChunk, SpeechSegment
from .models.transcription import TranscriptionSegment, TranscriptionResult
from .models.quality import QualityReport, AudioQualityReport, TranscriptionQualityReport
from .models.performance import PerformanceMetrics, ResourceStatus
from .models.config import STTConfig, AudioConfig, EngineConfig, ProcessingConfig

__all__ = [
    # Interfaces
    "IAudioCapture",
    "IAudioPreprocessor",
    "IVADDetector",
    "ISTTEngine",
    "ISTTEngineManager",
    "IHallucinationFilter",
    "ITextPostProcessor",
    "IQualityAnalyzer",
    "IPerformanceMonitor",
    "IConfigManager",
    # Models
    "AudioChunk",
    "SpeechSegment",
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
]
