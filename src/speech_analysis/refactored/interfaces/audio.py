"""Audio pipeline interfaces."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import numpy as np

from ..models.audio import AudioChunk, AudioDevice, SpeechSegment


class IAudioCapture(ABC):
    """Interface for audio capture from microphone."""
    
    @abstractmethod
    def start_capture(self, device_index: Optional[int] = None) -> bool:
        """
        Start capturing audio from microphone.
        
        Args:
            device_index: Optional device index to use. If None, uses default.
            
        Returns:
            True if capture started successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def stop_capture(self) -> None:
        """Stop capturing and cleanup resources."""
        pass
    
    @abstractmethod
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[AudioChunk]:
        """
        Get next audio chunk from queue.
        
        Args:
            timeout: Maximum time to wait for chunk in seconds.
            
        Returns:
            AudioChunk if available, None if timeout or stopped.
        """
        pass
    
    @abstractmethod
    def is_capturing(self) -> bool:
        """
        Check if currently capturing.
        
        Returns:
            True if capturing, False otherwise.
        """
        pass
    
    @abstractmethod
    def list_devices(self) -> List[AudioDevice]:
        """
        List available audio input devices.
        
        Returns:
            List of available audio devices.
        """
        pass
    
    @abstractmethod
    def get_current_device(self) -> Optional[AudioDevice]:
        """
        Get currently active audio device.
        
        Returns:
            Current AudioDevice if capturing, None otherwise.
        """
        pass


class IAudioPreprocessor(ABC):
    """Interface for audio preprocessing pipeline."""
    
    @abstractmethod
    def preprocess(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply preprocessing pipeline to audio.
        
        Args:
            audio: Input audio data.
            sample_rate: Sample rate of audio.
            
        Returns:
            Preprocessed audio data.
        """
        pass
    
    @abstractmethod
    def add_step(self, step_name: str, step_func: callable, **kwargs) -> None:
        """
        Add preprocessing step to pipeline.
        
        Args:
            step_name: Name of the preprocessing step.
            step_func: Function to apply for this step.
            **kwargs: Additional arguments for the step function.
        """
        pass
    
    @abstractmethod
    def remove_step(self, step_name: str) -> None:
        """
        Remove preprocessing step from pipeline.
        
        Args:
            step_name: Name of the step to remove.
        """
        pass
    
    @abstractmethod
    def get_pipeline_info(self) -> List[Dict[str, Any]]:
        """
        Get information about current pipeline.
        
        Returns:
            List of dictionaries containing step information.
        """
        pass
    
    @abstractmethod
    def enable_step(self, step_name: str) -> None:
        """
        Enable a preprocessing step.
        
        Args:
            step_name: Name of the step to enable.
        """
        pass
    
    @abstractmethod
    def disable_step(self, step_name: str) -> None:
        """
        Disable a preprocessing step.
        
        Args:
            step_name: Name of the step to disable.
        """
        pass


class IVADDetector(ABC):
    """Interface for Voice Activity Detection."""
    
    @abstractmethod
    def detect_speech(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> List[SpeechSegment]:
        """
        Detect speech segments in audio.
        
        Args:
            audio: Input audio data.
            sample_rate: Sample rate of audio.
            
        Returns:
            List of detected speech segments.
        """
        pass
    
    @abstractmethod
    def is_speech(self, audio: np.ndarray, sample_rate: int) -> bool:
        """
        Check if audio contains speech.
        
        Args:
            audio: Input audio data.
            sample_rate: Sample rate of audio.
            
        Returns:
            True if audio contains speech, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_speech_probability(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> float:
        """
        Get probability that audio contains speech.
        
        Args:
            audio: Input audio data.
            sample_rate: Sample rate of audio.
            
        Returns:
            Speech probability (0-1).
        """
        pass
    
    @abstractmethod
    def set_sensitivity(self, sensitivity: float) -> None:
        """
        Set VAD sensitivity.
        
        Args:
            sensitivity: Sensitivity level (0-1). Higher = more sensitive.
        """
        pass
