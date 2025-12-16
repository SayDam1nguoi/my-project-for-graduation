"""
Custom exceptions for speech analysis module.

This module defines custom exceptions for various error conditions
that can occur during speech analysis operations.
"""


class SpeechAnalysisError(Exception):
    """Base exception for all speech analysis errors."""
    pass


class MicrophoneNotFoundError(SpeechAnalysisError):
    """Raised when no microphone device is found."""
    
    def __init__(self, message="No microphone device found"):
        self.message = message
        super().__init__(self.message)


class MicrophoneInUseError(SpeechAnalysisError):
    """Raised when microphone is already in use by another application."""
    
    def __init__(self, message="Microphone is already in use"):
        self.message = message
        super().__init__(self.message)


class AudioDriverError(SpeechAnalysisError):
    """Raised when there's an error with the audio driver."""
    
    def __init__(self, message="Audio driver error occurred"):
        self.message = message
        super().__init__(self.message)


class ModelLoadError(SpeechAnalysisError):
    """Raised when a speech recognition model fails to load."""
    
    def __init__(self, message="Failed to load speech recognition model"):
        self.message = message
        super().__init__(self.message)


class OutOfMemoryError(SpeechAnalysisError):
    """Raised when system runs out of memory during processing."""
    
    def __init__(self, message="Out of memory"):
        self.message = message
        super().__init__(self.message)


class AudioQualityError(SpeechAnalysisError):
    """Raised when audio quality is too low for processing."""
    
    def __init__(self, message="Audio quality is too low"):
        self.message = message
        super().__init__(self.message)


class TranscriptionError(SpeechAnalysisError):
    """Raised when speech-to-text transcription fails."""
    
    def __init__(self, message="Transcription failed"):
        self.message = message
        super().__init__(self.message)


class FileStorageError(SpeechAnalysisError):
    """Raised when file storage operations fail."""
    
    def __init__(self, message="File storage operation failed"):
        self.message = message
        super().__init__(self.message)
