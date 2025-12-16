"""Custom exceptions for the refactored STT system."""


class STTException(Exception):
    """Base exception for STT system."""
    pass


class AudioCaptureException(STTException):
    """Exception related to audio capture."""
    pass


class MicrophoneNotFoundError(AudioCaptureException):
    """Microphone device not found."""
    pass


class MicrophoneInUseError(AudioCaptureException):
    """Microphone is already in use."""
    pass


class AudioDriverError(AudioCaptureException):
    """Audio driver error."""
    pass


class AudioProcessingException(STTException):
    """Exception during audio processing."""
    pass


class EngineException(STTException):
    """Exception related to STT engines."""
    pass


class ModelLoadError(EngineException):
    """Failed to load STT model."""
    pass


class EngineNotAvailableError(EngineException):
    """STT engine is not available."""
    pass


class TranscriptionException(STTException):
    """Exception during transcription."""
    pass


class ConfigurationException(STTException):
    """Exception related to configuration."""
    pass


class InvalidConfigError(ConfigurationException):
    """Configuration is invalid."""
    pass


class PerformanceException(STTException):
    """Exception related to performance issues."""
    pass


class ResourceLimitExceeded(PerformanceException):
    """Resource limit exceeded (CPU, memory, etc.)."""
    pass
