"""Core interfaces for the refactored STT system."""

from .audio import IAudioCapture, IAudioPreprocessor, IVADDetector
from .engine import ISTTEngine, ISTTEngineManager
from .processing import IHallucinationFilter, ITextPostProcessor, IQualityAnalyzer
from .monitoring import IPerformanceMonitor
from .config import IConfigManager

__all__ = [
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
]
