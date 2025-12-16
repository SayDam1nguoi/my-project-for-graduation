"""Custom exceptions for audio recording system."""


class AudioRecordingError(Exception):
    """Base exception for audio recording errors."""
    pass


class DeviceError(AudioRecordingError):
    """Device-related errors.
    
    Raised when:
    - No audio devices are available
    - Device is disconnected during recording
    - Permission denied for microphone access
    - Device is busy (used by another application)
    """
    pass


class FileSystemError(AudioRecordingError):
    """File system-related errors.
    
    Raised when:
    - Cannot create recordings directory
    - Disk is full
    - Permission denied for file write
    - Invalid file path
    """
    pass


class AudioFormatError(AudioRecordingError):
    """Audio format-related errors.
    
    Raised when:
    - Sample rate is not supported
    - Invalid audio format
    - Bit depth is not supported
    """
    pass


class StateError(AudioRecordingError):
    """Invalid state transition errors.
    
    Raised when:
    - Invalid state transition (e.g., pause when not recording)
    - Already recording
    - Not recording when trying to stop
    """
    pass
