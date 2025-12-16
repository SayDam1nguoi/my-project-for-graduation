"""Base STT Engine implementation."""

import logging
from abc import ABC
from typing import Iterator, Optional
import numpy as np

from ..interfaces.engine import ISTTEngine, EngineInfo
from ..models.config import EngineConfig
from ..models.transcription import TranscriptionResult
from ..exceptions import EngineException, ModelLoadError


logger = logging.getLogger(__name__)


class BaseSTTEngine(ISTTEngine, ABC):
    """
    Base implementation for STT engines.
    
    Provides common functionality for all engines:
    - Initialization tracking
    - Configuration management
    - Basic error handling
    - Logging
    """
    
    def __init__(self, name: str):
        """
        Initialize base engine.
        
        Args:
            name: Engine name.
        """
        self._name = name
        self._config: Optional[EngineConfig] = None
        self._initialized = False
        self._available = False
        
        logger.info(f"Created {name} engine")
    
    def initialize(self, config: EngineConfig) -> bool:
        """
        Initialize engine with configuration.
        
        Args:
            config: Engine configuration.
            
        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            logger.info(f"Initializing {self._name} engine")
            self._config = config
            
            # Call subclass initialization
            success = self._initialize_impl(config)
            
            if success:
                self._initialized = True
                self._available = True
                logger.info(f"{self._name} engine initialized successfully")
            else:
                logger.error(f"{self._name} engine initialization failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error initializing {self._name} engine: {e}")
            self._initialized = False
            self._available = False
            return False
    
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
            
        Raises:
            EngineException: If engine not initialized or transcription fails.
        """
        if not self._initialized:
            raise EngineException(f"{self._name} engine not initialized")
        
        if not self._available:
            raise EngineException(f"{self._name} engine not available")
        
        try:
            # Call subclass implementation
            return self._transcribe_chunk_impl(audio, sample_rate)
            
        except Exception as e:
            logger.error(f"Transcription error in {self._name}: {e}")
            raise EngineException(
                f"Transcription failed in {self._name}: {str(e)}"
            ) from e
    
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
            
        Raises:
            EngineException: If engine not initialized or transcription fails.
        """
        if not self._initialized:
            raise EngineException(f"{self._name} engine not initialized")
        
        if not self._available:
            raise EngineException(f"{self._name} engine not available")
        
        try:
            # Call subclass implementation
            yield from self._transcribe_stream_impl(audio_stream, sample_rate)
            
        except Exception as e:
            logger.error(f"Stream transcription error in {self._name}: {e}")
            raise EngineException(
                f"Stream transcription failed in {self._name}: {str(e)}"
            ) from e
    
    def is_available(self) -> bool:
        """
        Check if engine is available and ready.
        
        Returns:
            True if engine is ready, False otherwise.
        """
        return self._initialized and self._available
    
    def get_engine_info(self) -> EngineInfo:
        """
        Get engine information and capabilities.
        
        Returns:
            Engine information.
        """
        # Call subclass implementation
        return self._get_engine_info_impl()
    
    def cleanup(self) -> None:
        """Cleanup resources and unload models."""
        try:
            logger.info(f"Cleaning up {self._name} engine")
            
            # Call subclass cleanup
            self._cleanup_impl()
            
            self._initialized = False
            self._available = False
            
            logger.info(f"{self._name} engine cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up {self._name} engine: {e}")
    
    def get_name(self) -> str:
        """
        Get engine name.
        
        Returns:
            Engine name.
        """
        return self._name
    
    def get_config(self) -> Optional[EngineConfig]:
        """
        Get current configuration.
        
        Returns:
            Engine configuration if initialized, None otherwise.
        """
        return self._config
    
    # Abstract methods to be implemented by subclasses
    
    def _initialize_impl(self, config: EngineConfig) -> bool:
        """
        Subclass-specific initialization.
        
        Args:
            config: Engine configuration.
            
        Returns:
            True if successful, False otherwise.
        """
        raise NotImplementedError
    
    def _transcribe_chunk_impl(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> TranscriptionResult:
        """
        Subclass-specific chunk transcription.
        
        Args:
            audio: Audio data to transcribe.
            sample_rate: Sample rate of audio.
            
        Returns:
            Transcription result.
        """
        raise NotImplementedError
    
    def _transcribe_stream_impl(
        self,
        audio_stream: Iterator[np.ndarray],
        sample_rate: int,
    ) -> Iterator[TranscriptionResult]:
        """
        Subclass-specific stream transcription.
        
        Args:
            audio_stream: Iterator of audio chunks.
            sample_rate: Sample rate of audio.
            
        Yields:
            Transcription results.
        """
        raise NotImplementedError
    
    def _get_engine_info_impl(self) -> EngineInfo:
        """
        Subclass-specific engine info.
        
        Returns:
            Engine information.
        """
        raise NotImplementedError
    
    def _cleanup_impl(self) -> None:
        """Subclass-specific cleanup."""
        pass
