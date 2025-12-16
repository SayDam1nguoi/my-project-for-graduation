"""STT Engine interfaces."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Iterator
import numpy as np

from ..models.transcription import TranscriptionResult
from ..models.config import EngineConfig


class EngineInfo:
    """Information about an STT engine."""
    
    def __init__(
        self,
        name: str,
        version: str,
        supported_languages: List[str],
        supports_streaming: bool,
        supports_timestamps: bool,
        supports_confidence: bool,
        max_audio_length: Optional[float] = None,
    ):
        self.name = name
        self.version = version
        self.supported_languages = supported_languages
        self.supports_streaming = supports_streaming
        self.supports_timestamps = supports_timestamps
        self.supports_confidence = supports_confidence
        self.max_audio_length = max_audio_length


class ISTTEngine(ABC):
    """Interface for Speech-to-Text engines."""
    
    @abstractmethod
    def initialize(self, config: EngineConfig) -> bool:
        """
        Initialize engine with configuration.
        
        Args:
            config: Engine configuration.
            
        Returns:
            True if initialization successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def transcribe_chunk(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> TranscriptionResult:
        """
        Transcribe a single audio chunk.
        
        Args:
            audio: Audio data to transcribe.
            sample_rate: Sample rate of audio.
            
        Returns:
            Transcription result.
        """
        pass
    
    @abstractmethod
    def transcribe_stream(
        self,
        audio_stream: Iterator[np.ndarray],
        sample_rate: int,
    ) -> Iterator[TranscriptionResult]:
        """
        Transcribe audio stream in real-time.
        
        Args:
            audio_stream: Iterator of audio chunks.
            sample_rate: Sample rate of audio.
            
        Yields:
            Transcription results as they become available.
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if engine is available and ready.
        
        Returns:
            True if engine is ready, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_engine_info(self) -> EngineInfo:
        """
        Get engine information and capabilities.
        
        Returns:
            Engine information.
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources and unload models."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get engine name.
        
        Returns:
            Engine name.
        """
        pass


class ISTTEngineManager(ABC):
    """Interface for managing multiple STT engines."""
    
    @abstractmethod
    def register_engine(self, engine: ISTTEngine, priority: int = 0) -> None:
        """
        Register an STT engine with priority.
        
        Args:
            engine: Engine to register.
            priority: Engine priority (higher = preferred). Default is 0.
        """
        pass
    
    @abstractmethod
    def unregister_engine(self, engine_name: str) -> None:
        """
        Unregister an STT engine.
        
        Args:
            engine_name: Name of engine to unregister.
        """
        pass
    
    @abstractmethod
    def transcribe(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> TranscriptionResult:
        """
        Transcribe audio using best available engine.
        
        Args:
            audio: Audio data to transcribe.
            sample_rate: Sample rate of audio.
            
        Returns:
            Transcription result.
            
        Raises:
            RuntimeError: If all engines fail.
        """
        pass
    
    @abstractmethod
    def get_active_engine(self) -> Optional[ISTTEngine]:
        """
        Get currently active engine.
        
        Returns:
            Active engine if available, None otherwise.
        """
        pass
    
    @abstractmethod
    def switch_engine(self, engine_name: str) -> bool:
        """
        Switch to specified engine.
        
        Args:
            engine_name: Name of engine to switch to.
            
        Returns:
            True if switch successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_available_engines(self) -> List[str]:
        """
        List all available engines.
        
        Returns:
            List of engine names.
        """
        pass
    
    @abstractmethod
    def get_engine_by_name(self, engine_name: str) -> Optional[ISTTEngine]:
        """
        Get engine by name.
        
        Args:
            engine_name: Name of engine to retrieve.
            
        Returns:
            Engine if found, None otherwise.
        """
        pass
    
    @abstractmethod
    def get_engine_health(self, engine_name: str) -> Dict[str, Any]:
        """
        Get health status of an engine.
        
        Args:
            engine_name: Name of engine to check.
            
        Returns:
            Dictionary containing health information.
        """
        pass
