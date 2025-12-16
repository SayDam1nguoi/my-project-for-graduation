"""Test that all imports work correctly."""

import pytest


def test_package_imports():
    """Test that the main package imports successfully."""
    import src.speech_analysis.refactored as stt
    
    assert hasattr(stt, '__version__')
    assert stt.__version__ == "2.0.0"


def test_interface_imports():
    """Test that all interfaces can be imported."""
    from src.speech_analysis.refactored import (
        IAudioCapture,
        IAudioPreprocessor,
        IVADDetector,
        ISTTEngine,
        ISTTEngineManager,
        IHallucinationFilter,
        ITextPostProcessor,
        IQualityAnalyzer,
        IPerformanceMonitor,
        IConfigManager,
    )
    
    # Verify they are all classes
    assert IAudioCapture is not None
    assert IAudioPreprocessor is not None
    assert IVADDetector is not None
    assert ISTTEngine is not None
    assert ISTTEngineManager is not None
    assert IHallucinationFilter is not None
    assert ITextPostProcessor is not None
    assert IQualityAnalyzer is not None
    assert IPerformanceMonitor is not None
    assert IConfigManager is not None


def test_model_imports():
    """Test that all models can be imported."""
    from src.speech_analysis.refactored import (
        AudioChunk,
        SpeechSegment,
        TranscriptionSegment,
        TranscriptionResult,
        QualityReport,
        AudioQualityReport,
        TranscriptionQualityReport,
        PerformanceMetrics,
        ResourceStatus,
        STTConfig,
        AudioConfig,
        EngineConfig,
        ProcessingConfig,
    )
    
    # Verify they are all classes
    assert AudioChunk is not None
    assert SpeechSegment is not None
    assert TranscriptionSegment is not None
    assert TranscriptionResult is not None
    assert QualityReport is not None
    assert AudioQualityReport is not None
    assert TranscriptionQualityReport is not None
    assert PerformanceMetrics is not None
    assert ResourceStatus is not None
    assert STTConfig is not None
    assert AudioConfig is not None
    assert EngineConfig is not None
    assert ProcessingConfig is not None


def test_exception_imports():
    """Test that exceptions can be imported."""
    from src.speech_analysis.refactored.exceptions import (
        STTException,
        AudioCaptureException,
        MicrophoneNotFoundError,
        MicrophoneInUseError,
        AudioProcessingException,
        EngineException,
        ModelLoadError,
        EngineNotAvailableError,
        TranscriptionException,
        ConfigurationException,
        InvalidConfigError,
        PerformanceException,
        ResourceLimitExceeded,
    )
    
    # Verify they are all exception classes
    assert issubclass(STTException, Exception)
    assert issubclass(AudioCaptureException, STTException)
    assert issubclass(MicrophoneNotFoundError, AudioCaptureException)
    assert issubclass(EngineException, STTException)
    assert issubclass(ModelLoadError, EngineException)


def test_all_exports():
    """Test that __all__ exports are correct."""
    import src.speech_analysis.refactored as stt
    
    # Check that __all__ is defined
    assert hasattr(stt, '__all__')
    
    # Check that all exported names are accessible
    for name in stt.__all__:
        assert hasattr(stt, name), f"Exported name '{name}' not found in module"
