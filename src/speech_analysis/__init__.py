"""
Speech Analysis Module

This module provides speech analysis capabilities including:
- Audio capture from microphone
- Speech quality analysis (clarity and fluency)
- Speech-to-text conversion (Vietnamese support)
- Transcript display and storage

Main Components:
- AudioCapture: Captures audio from microphone
- SpeechQualityAnalyzer: Analyzes speech clarity and fluency
- SpeechToTextEngine: Converts speech to text
- TranscriptWindow: Displays transcripts in GUI
- TextStorage: Saves transcripts to files
- SpeechAnalysisCoordinator: Coordinates all components
"""

# Version information
__version__ = "1.0.0"
__author__ = "Emotion Scanner Team"

# Import main classes (will be implemented in subsequent tasks)
# These imports will be uncommented as each component is implemented

from .audio_capture import AudioCapture, AudioConfig
from .quality_analyzer import SpeechQualityAnalyzer, QualityReport, QualityConfig
from .speech_to_text import (
    SpeechToTextEngine,
    STTConfig,
    TranscriptionResult,
    TranscriptionSegment,
    VoskSTTEngine,
    create_stt_engine
)
from .transcript_window import TranscriptWindow
from .text_storage import TextStorage, TranscriptMetadata, TranscriptInfo
from .coordinator import (
    SpeechAnalysisCoordinator,
    SpeechEvent,
    TranscriptionEvent,
    QualityEvent,
    SessionData,
    EventType
)
from .config import (
    load_config,
    save_config,
    SpeechAnalysisConfig,
    AudioConfig as ConfigAudioConfig,
    STTConfig as ConfigSTTConfig,
    QualityConfig as ConfigQualityConfig,
    StorageConfig,
    PerformanceConfig,
    LoggingConfig
)
from .logger import (
    setup_logger,
    get_logger,
    shutdown_logger,
    set_log_level,
    debug,
    info,
    warning,
    error,
    critical,
    exception,
    LogLevelContext
)
from .exceptions import (
    SpeechAnalysisError,
    MicrophoneNotFoundError,
    MicrophoneInUseError,
    AudioDriverError,
    ModelLoadError,
    OutOfMemoryError,
    AudioQualityError,
    TranscriptionError,
    FileStorageError
)
from .vietnamese_stt_optimizer import (
    VietnameseSTTOptimizer,
    VietnameseDiacriticValidator,
    VietnamesePostCorrector,
    create_vietnamese_optimizer
)

# Public API
__all__ = [
    # Core components (to be uncommented as implemented)
    'AudioCapture',
    'AudioConfig',
    'SpeechQualityAnalyzer',
    'QualityReport',
    'QualityConfig',
    'SpeechToTextEngine',
    'STTConfig',
    'TranscriptionResult',
    'TranscriptionSegment',
    'VoskSTTEngine',
    'create_stt_engine',
    'TranscriptWindow',
    'TextStorage',
    'TranscriptMetadata',
    'TranscriptInfo',
    'SpeechAnalysisCoordinator',
    'SpeechEvent',
    'TranscriptionEvent',
    'QualityEvent',
    'SessionData',
    'EventType',
    # Configuration
    'load_config',
    'save_config',
    'SpeechAnalysisConfig',
    'ConfigAudioConfig',
    'ConfigSTTConfig',
    'ConfigQualityConfig',
    'StorageConfig',
    'PerformanceConfig',
    'LoggingConfig',
    # Logging
    'setup_logger',
    'get_logger',
    'shutdown_logger',
    'set_log_level',
    'debug',
    'info',
    'warning',
    'error',
    'critical',
    'exception',
    'LogLevelContext',
    # Exceptions
    'SpeechAnalysisError',
    'MicrophoneNotFoundError',
    'MicrophoneInUseError',
    'AudioDriverError',
    'ModelLoadError',
    'OutOfMemoryError',
    'AudioQualityError',
    'TranscriptionError',
    'FileStorageError',
    # Vietnamese Optimization
    'VietnameseSTTOptimizer',
    'VietnameseDiacriticValidator',
    'VietnamesePostCorrector',
    'create_vietnamese_optimizer',
]

# Module-level configuration
DEFAULT_CONFIG_PATH = "config/speech_config.yaml"

def get_version():
    """Return the version of the speech analysis module."""
    return __version__

def check_dependencies():
    """
    Check if all required dependencies are installed.
    
    Returns:
        dict: Dictionary with dependency names as keys and availability as values
    """
    dependencies = {
        'pyaudio': False,
        'vosk': False,
        'librosa': False,
        'soundfile': False,
        'numpy': False,
        'scipy': False,
    }
    
    # Check PyAudio
    try:
        import pyaudio
        dependencies['pyaudio'] = True
    except ImportError:
        pass
    
    # Check Vosk
    try:
        import vosk
        dependencies['vosk'] = True
    except ImportError:
        pass
    
    # Check librosa
    try:
        import librosa
        dependencies['librosa'] = True
    except ImportError:
        pass
    
    # Check soundfile
    try:
        import soundfile
        dependencies['soundfile'] = True
    except ImportError:
        pass
    
    # Check numpy
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        pass
    
    # Check scipy
    try:
        import scipy
        dependencies['scipy'] = True
    except ImportError:
        pass
    
    return dependencies

def is_available():
    """
    Check if speech analysis is available (all core dependencies installed).
    
    Returns:
        bool: True if speech analysis can be used, False otherwise
    """
    deps = check_dependencies()
    # Core dependencies: pyaudio, numpy, scipy, vosk
    core_available = deps['pyaudio'] and deps['numpy'] and deps['scipy']
    stt_available = deps['vosk']
    
    return core_available and stt_available

# Module initialization message
def _init_message():
    """Print initialization message with dependency status."""
    if is_available():
        return "Speech Analysis module initialized successfully"
    else:
        missing = [name for name, available in check_dependencies().items() if not available]
        return f"Speech Analysis module loaded with missing dependencies: {', '.join(missing)}"

# Note: Uncomment this when dependencies are installed
# print(_init_message())
