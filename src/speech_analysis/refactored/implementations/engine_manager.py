"""STT Engine Manager implementation."""

import logging
import time
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np

from ..interfaces.engine import ISTTEngine, ISTTEngineManager
from ..models.transcription import TranscriptionResult
from ..exceptions import (
    EngineException,
    EngineNotAvailableError,
    TranscriptionException,
)


logger = logging.getLogger(__name__)


class EngineHealthStatus:
    """Track health status of an engine."""
    
    def __init__(self, engine_name: str):
        self.engine_name = engine_name
        self.is_healthy = True
        self.last_success_time: Optional[float] = None
        self.last_failure_time: Optional[float] = None
        self.consecutive_failures = 0
        self.total_successes = 0
        self.total_failures = 0
        self.average_processing_time = 0.0
        self.last_error: Optional[str] = None
    
    def record_success(self, processing_time: float) -> None:
        """Record successful transcription."""
        self.last_success_time = time.time()
        self.consecutive_failures = 0
        self.total_successes += 1
        self.is_healthy = True
        
        # Update average processing time (exponential moving average)
        alpha = 0.3
        if self.average_processing_time == 0.0:
            self.average_processing_time = processing_time
        else:
            self.average_processing_time = (
                alpha * processing_time + 
                (1 - alpha) * self.average_processing_time
            )
    
    def record_failure(self, error: str) -> None:
        """Record failed transcription."""
        self.last_failure_time = time.time()
        self.consecutive_failures += 1
        self.total_failures += 1
        self.last_error = error
        
        # Mark as unhealthy after 3 consecutive failures
        if self.consecutive_failures >= 3:
            self.is_healthy = False
    
    def get_success_rate(self) -> float:
        """Get success rate (0-1)."""
        total = self.total_successes + self.total_failures
        if total == 0:
            return 1.0
        return self.total_successes / total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "engine_name": self.engine_name,
            "is_healthy": self.is_healthy,
            "last_success_time": self.last_success_time,
            "last_failure_time": self.last_failure_time,
            "consecutive_failures": self.consecutive_failures,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "success_rate": self.get_success_rate(),
            "average_processing_time": self.average_processing_time,
            "last_error": self.last_error,
        }


class STTEngineManager(ISTTEngineManager):
    """
    Manages multiple STT engines with priority-based selection and fallback.
    
    Features:
    - Priority-based engine selection
    - Automatic fallback on failure
    - Engine health monitoring
    - Runtime engine switching
    """
    
    def __init__(self):
        """Initialize engine manager."""
        self._engines: Dict[str, ISTTEngine] = {}
        self._priorities: Dict[str, int] = {}
        self._health_status: Dict[str, EngineHealthStatus] = {}
        self._active_engine_name: Optional[str] = None
        self._fallback_enabled = True
        
        logger.info("STTEngineManager initialized")
    
    def register_engine(self, engine: ISTTEngine, priority: int = 0) -> None:
        """
        Register an STT engine with priority.
        
        Args:
            engine: Engine to register.
            priority: Engine priority (higher = preferred). Default is 0.
        """
        engine_name = engine.get_name()
        
        if engine_name in self._engines:
            logger.warning(f"Engine {engine_name} already registered, replacing")
        
        self._engines[engine_name] = engine
        self._priorities[engine_name] = priority
        self._health_status[engine_name] = EngineHealthStatus(engine_name)
        
        # Set as active if it's the first engine or has higher priority
        if (self._active_engine_name is None or 
            priority > self._priorities.get(self._active_engine_name, -1)):
            self._active_engine_name = engine_name
        
        logger.info(
            f"Registered engine: {engine_name} with priority {priority}"
        )
    
    def unregister_engine(self, engine_name: str) -> None:
        """
        Unregister an STT engine.
        
        Args:
            engine_name: Name of engine to unregister.
        """
        if engine_name not in self._engines:
            logger.warning(f"Engine {engine_name} not found")
            return
        
        # Cleanup engine
        try:
            self._engines[engine_name].cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up engine {engine_name}: {e}")
        
        # Remove from tracking
        del self._engines[engine_name]
        del self._priorities[engine_name]
        del self._health_status[engine_name]
        
        # Update active engine if needed
        if self._active_engine_name == engine_name:
            self._active_engine_name = self._get_best_available_engine()
        
        logger.info(f"Unregistered engine: {engine_name}")
    
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
        if not self._engines:
            raise EngineNotAvailableError("No engines registered")
        
        # Get engines in priority order
        engines_to_try = self._get_engines_by_priority()
        
        # Try each engine until one succeeds
        last_error = None
        for engine_name in engines_to_try:
            engine = self._engines[engine_name]
            health = self._health_status[engine_name]
            
            # Skip unhealthy engines unless it's the last option
            if not health.is_healthy and len(engines_to_try) > 1:
                logger.warning(
                    f"Skipping unhealthy engine: {engine_name} "
                    f"(failures: {health.consecutive_failures})"
                )
                continue
            
            # Check if engine is available
            if not engine.is_available():
                logger.warning(f"Engine {engine_name} not available")
                health.record_failure("Engine not available")
                continue
            
            # Try transcription
            try:
                logger.debug(f"Attempting transcription with {engine_name}")
                start_time = time.time()
                
                result = engine.transcribe_chunk(audio, sample_rate)
                
                processing_time = time.time() - start_time
                health.record_success(processing_time)
                
                # Update active engine
                self._active_engine_name = engine_name
                
                logger.info(
                    f"Transcription successful with {engine_name} "
                    f"(time: {processing_time:.2f}s)"
                )
                
                return result
                
            except Exception as e:
                error_msg = f"Transcription failed: {str(e)}"
                logger.error(f"Engine {engine_name} failed: {error_msg}")
                health.record_failure(error_msg)
                last_error = e
                
                # If fallback is disabled, raise immediately
                if not self._fallback_enabled:
                    raise TranscriptionException(
                        f"Engine {engine_name} failed: {error_msg}"
                    ) from e
                
                # Continue to next engine
                continue
        
        # All engines failed
        error_msg = "All engines failed to transcribe audio"
        logger.error(error_msg)
        if last_error:
            raise TranscriptionException(error_msg) from last_error
        else:
            raise TranscriptionException(error_msg)
    
    def get_active_engine(self) -> Optional[ISTTEngine]:
        """
        Get currently active engine.
        
        Returns:
            Active engine if available, None otherwise.
        """
        if self._active_engine_name is None:
            return None
        return self._engines.get(self._active_engine_name)
    
    def switch_engine(self, engine_name: str) -> bool:
        """
        Switch to specified engine.
        
        Args:
            engine_name: Name of engine to switch to.
            
        Returns:
            True if switch successful, False otherwise.
        """
        if engine_name not in self._engines:
            logger.error(f"Engine {engine_name} not found")
            return False
        
        engine = self._engines[engine_name]
        
        # Check if engine is available
        if not engine.is_available():
            logger.error(f"Engine {engine_name} not available")
            return False
        
        self._active_engine_name = engine_name
        logger.info(f"Switched to engine: {engine_name}")
        return True
    
    def get_available_engines(self) -> List[str]:
        """
        List all available engines.
        
        Returns:
            List of engine names.
        """
        return list(self._engines.keys())
    
    def get_engine_by_name(self, engine_name: str) -> Optional[ISTTEngine]:
        """
        Get engine by name.
        
        Args:
            engine_name: Name of engine to retrieve.
            
        Returns:
            Engine if found, None otherwise.
        """
        return self._engines.get(engine_name)
    
    def get_engine_health(self, engine_name: str) -> Dict[str, Any]:
        """
        Get health status of an engine.
        
        Args:
            engine_name: Name of engine to check.
            
        Returns:
            Dictionary containing health information.
        """
        if engine_name not in self._health_status:
            return {
                "error": f"Engine {engine_name} not found"
            }
        
        health = self._health_status[engine_name]
        return health.to_dict()
    
    def get_all_engine_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Get health status of all engines.
        
        Returns:
            Dictionary mapping engine names to health information.
        """
        return {
            name: health.to_dict()
            for name, health in self._health_status.items()
        }
    
    def set_fallback_enabled(self, enabled: bool) -> None:
        """
        Enable or disable automatic fallback.
        
        Args:
            enabled: True to enable fallback, False to disable.
        """
        self._fallback_enabled = enabled
        logger.info(f"Fallback {'enabled' if enabled else 'disabled'}")
    
    def _get_engines_by_priority(self) -> List[str]:
        """
        Get list of engine names sorted by priority (highest first).
        
        Returns:
            List of engine names in priority order.
        """
        return sorted(
            self._engines.keys(),
            key=lambda name: self._priorities[name],
            reverse=True
        )
    
    def _get_best_available_engine(self) -> Optional[str]:
        """
        Get the best available engine based on priority and health.
        
        Returns:
            Engine name if found, None otherwise.
        """
        engines_by_priority = self._get_engines_by_priority()
        
        for engine_name in engines_by_priority:
            engine = self._engines[engine_name]
            health = self._health_status[engine_name]
            
            if engine.is_available() and health.is_healthy:
                return engine_name
        
        # If no healthy engine found, return highest priority engine
        if engines_by_priority:
            return engines_by_priority[0]
        
        return None
    
    def cleanup(self) -> None:
        """Cleanup all engines."""
        logger.info("Cleaning up all engines")
        
        for engine_name, engine in self._engines.items():
            try:
                engine.cleanup()
                logger.debug(f"Cleaned up engine: {engine_name}")
            except Exception as e:
                logger.error(f"Error cleaning up engine {engine_name}: {e}")
        
        self._engines.clear()
        self._priorities.clear()
        self._health_status.clear()
        self._active_engine_name = None
