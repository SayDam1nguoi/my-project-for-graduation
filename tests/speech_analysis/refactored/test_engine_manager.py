"""Tests for STT Engine Manager."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
import time

from src.speech_analysis.refactored.implementations.engine_manager import (
    STTEngineManager,
    EngineHealthStatus,
)
from src.speech_analysis.refactored.interfaces.engine import ISTTEngine, EngineInfo
from src.speech_analysis.refactored.models.transcription import (
    TranscriptionResult,
    TranscriptionSegment,
)
from src.speech_analysis.refactored.models.config import EngineConfig, EngineType
from src.speech_analysis.refactored.exceptions import (
    EngineNotAvailableError,
    TranscriptionException,
)


class MockSTTEngine(ISTTEngine):
    """Mock STT engine for testing."""
    
    def __init__(self, name: str, should_fail: bool = False):
        self.name = name
        self.should_fail = should_fail
        self._initialized = False
        self._available = True
        self.transcribe_count = 0
    
    def initialize(self, config: EngineConfig) -> bool:
        self._initialized = True
        return True
    
    def transcribe_chunk(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> TranscriptionResult:
        self.transcribe_count += 1
        
        if self.should_fail:
            raise Exception(f"{self.name} transcription failed")
        
        segment = TranscriptionSegment(
            text=f"Transcribed by {self.name}",
            start_time=0.0,
            end_time=1.0,
            confidence=0.9,
            language="vi",
        )
        
        return TranscriptionResult(
            text=segment.text,
            segments=[segment],
            language="vi",
            confidence=0.9,
            processing_time=0.1,
            engine_name=self.name,
        )
    
    def transcribe_stream(self, audio_stream, sample_rate):
        for audio in audio_stream:
            yield self.transcribe_chunk(audio, sample_rate)
    
    def is_available(self) -> bool:
        return self._available
    
    def get_engine_info(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            version="1.0.0",
            supported_languages=["vi", "en"],
            supports_streaming=True,
            supports_timestamps=True,
            supports_confidence=True,
        )
    
    def cleanup(self) -> None:
        self._initialized = False
    
    def get_name(self) -> str:
        return self.name


class TestEngineHealthStatus:
    """Test EngineHealthStatus class."""
    
    def test_initial_state(self):
        """Test initial health status."""
        health = EngineHealthStatus("test_engine")
        
        assert health.engine_name == "test_engine"
        assert health.is_healthy is True
        assert health.consecutive_failures == 0
        assert health.total_successes == 0
        assert health.total_failures == 0
        assert health.get_success_rate() == 1.0
    
    def test_record_success(self):
        """Test recording successful transcription."""
        health = EngineHealthStatus("test_engine")
        
        health.record_success(0.5)
        
        assert health.is_healthy is True
        assert health.consecutive_failures == 0
        assert health.total_successes == 1
        assert health.last_success_time is not None
        assert health.average_processing_time == 0.5
    
    def test_record_failure(self):
        """Test recording failed transcription."""
        health = EngineHealthStatus("test_engine")
        
        health.record_failure("Test error")
        
        assert health.consecutive_failures == 1
        assert health.total_failures == 1
        assert health.last_error == "Test error"
        assert health.is_healthy is True  # Still healthy after 1 failure
    
    def test_unhealthy_after_consecutive_failures(self):
        """Test engine becomes unhealthy after 3 consecutive failures."""
        health = EngineHealthStatus("test_engine")
        
        health.record_failure("Error 1")
        health.record_failure("Error 2")
        assert health.is_healthy is True
        
        health.record_failure("Error 3")
        assert health.is_healthy is False
        assert health.consecutive_failures == 3
    
    def test_success_resets_consecutive_failures(self):
        """Test success resets consecutive failure count."""
        health = EngineHealthStatus("test_engine")
        
        health.record_failure("Error 1")
        health.record_failure("Error 2")
        assert health.consecutive_failures == 2
        
        health.record_success(0.5)
        assert health.consecutive_failures == 0
        assert health.is_healthy is True
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        health = EngineHealthStatus("test_engine")
        
        health.record_success(0.5)
        health.record_success(0.5)
        health.record_failure("Error")
        
        assert health.get_success_rate() == 2.0 / 3.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        health = EngineHealthStatus("test_engine")
        health.record_success(0.5)
        
        data = health.to_dict()
        
        assert data["engine_name"] == "test_engine"
        assert data["is_healthy"] is True
        assert data["total_successes"] == 1
        assert data["success_rate"] == 1.0


class TestSTTEngineManager:
    """Test STTEngineManager class."""
    
    def test_initialization(self):
        """Test engine manager initialization."""
        manager = STTEngineManager()
        
        assert manager.get_available_engines() == []
        assert manager.get_active_engine() is None
    
    def test_register_single_engine(self):
        """Test registering a single engine."""
        manager = STTEngineManager()
        engine = MockSTTEngine("engine1")
        
        manager.register_engine(engine, priority=10)
        
        assert "engine1" in manager.get_available_engines()
        assert manager.get_active_engine() == engine
    
    def test_register_multiple_engines_priority(self):
        """Test registering multiple engines with different priorities."""
        manager = STTEngineManager()
        engine1 = MockSTTEngine("engine1")
        engine2 = MockSTTEngine("engine2")
        engine3 = MockSTTEngine("engine3")
        
        manager.register_engine(engine1, priority=5)
        manager.register_engine(engine2, priority=10)
        manager.register_engine(engine3, priority=3)
        
        # Highest priority engine should be active
        assert manager.get_active_engine() == engine2
        assert len(manager.get_available_engines()) == 3
    
    def test_unregister_engine(self):
        """Test unregistering an engine."""
        manager = STTEngineManager()
        engine = MockSTTEngine("engine1")
        
        manager.register_engine(engine, priority=10)
        assert "engine1" in manager.get_available_engines()
        
        manager.unregister_engine("engine1")
        assert "engine1" not in manager.get_available_engines()
        assert manager.get_active_engine() is None
    
    def test_transcribe_with_single_engine(self, sample_audio_16khz):
        """Test transcription with single engine."""
        manager = STTEngineManager()
        engine = MockSTTEngine("engine1")
        manager.register_engine(engine, priority=10)
        
        audio, sample_rate = sample_audio_16khz
        result = manager.transcribe(audio, sample_rate)
        
        assert result.text == "Transcribed by engine1"
        assert result.engine_name == "engine1"
        assert engine.transcribe_count == 1
    
    def test_transcribe_no_engines_registered(self, sample_audio_16khz):
        """Test transcription fails when no engines registered."""
        manager = STTEngineManager()
        audio, sample_rate = sample_audio_16khz
        
        with pytest.raises(EngineNotAvailableError):
            manager.transcribe(audio, sample_rate)
    
    def test_fallback_to_next_engine(self, sample_audio_16khz):
        """Test fallback to next engine when first fails."""
        manager = STTEngineManager()
        engine1 = MockSTTEngine("engine1", should_fail=True)
        engine2 = MockSTTEngine("engine2", should_fail=False)
        
        manager.register_engine(engine1, priority=10)
        manager.register_engine(engine2, priority=5)
        
        audio, sample_rate = sample_audio_16khz
        result = manager.transcribe(audio, sample_rate)
        
        # Should use engine2 after engine1 fails
        assert result.engine_name == "engine2"
        assert engine1.transcribe_count == 1
        assert engine2.transcribe_count == 1
    
    def test_all_engines_fail(self, sample_audio_16khz):
        """Test when all engines fail."""
        manager = STTEngineManager()
        engine1 = MockSTTEngine("engine1", should_fail=True)
        engine2 = MockSTTEngine("engine2", should_fail=True)
        
        manager.register_engine(engine1, priority=10)
        manager.register_engine(engine2, priority=5)
        
        audio, sample_rate = sample_audio_16khz
        
        with pytest.raises(TranscriptionException):
            manager.transcribe(audio, sample_rate)
    
    def test_switch_engine(self):
        """Test switching between engines."""
        manager = STTEngineManager()
        engine1 = MockSTTEngine("engine1")
        engine2 = MockSTTEngine("engine2")
        
        manager.register_engine(engine1, priority=10)
        manager.register_engine(engine2, priority=5)
        
        assert manager.get_active_engine() == engine1
        
        success = manager.switch_engine("engine2")
        assert success is True
        assert manager.get_active_engine() == engine2
    
    def test_switch_to_nonexistent_engine(self):
        """Test switching to non-existent engine fails."""
        manager = STTEngineManager()
        engine = MockSTTEngine("engine1")
        manager.register_engine(engine, priority=10)
        
        success = manager.switch_engine("nonexistent")
        assert success is False
        assert manager.get_active_engine() == engine
    
    def test_switch_to_unavailable_engine(self):
        """Test switching to unavailable engine fails."""
        manager = STTEngineManager()
        engine1 = MockSTTEngine("engine1")
        engine2 = MockSTTEngine("engine2")
        engine2._available = False
        
        manager.register_engine(engine1, priority=10)
        manager.register_engine(engine2, priority=5)
        
        success = manager.switch_engine("engine2")
        assert success is False
        assert manager.get_active_engine() == engine1
    
    def test_get_engine_by_name(self):
        """Test getting engine by name."""
        manager = STTEngineManager()
        engine = MockSTTEngine("engine1")
        manager.register_engine(engine, priority=10)
        
        retrieved = manager.get_engine_by_name("engine1")
        assert retrieved == engine
        
        not_found = manager.get_engine_by_name("nonexistent")
        assert not_found is None
    
    def test_get_engine_health(self, sample_audio_16khz):
        """Test getting engine health status."""
        manager = STTEngineManager()
        engine = MockSTTEngine("engine1")
        manager.register_engine(engine, priority=10)
        
        # Perform successful transcription
        audio, sample_rate = sample_audio_16khz
        manager.transcribe(audio, sample_rate)
        
        health = manager.get_engine_health("engine1")
        
        assert health["engine_name"] == "engine1"
        assert health["is_healthy"] is True
        assert health["total_successes"] == 1
        assert health["total_failures"] == 0
    
    def test_get_all_engine_health(self, sample_audio_16khz):
        """Test getting health status of all engines."""
        manager = STTEngineManager()
        engine1 = MockSTTEngine("engine1")
        engine2 = MockSTTEngine("engine2")
        
        manager.register_engine(engine1, priority=10)
        manager.register_engine(engine2, priority=5)
        
        audio, sample_rate = sample_audio_16khz
        manager.transcribe(audio, sample_rate)
        
        all_health = manager.get_all_engine_health()
        
        assert "engine1" in all_health
        assert "engine2" in all_health
        assert all_health["engine1"]["total_successes"] == 1
    
    def test_health_tracking_on_failure(self, sample_audio_16khz):
        """Test health tracking when engine fails."""
        manager = STTEngineManager()
        engine1 = MockSTTEngine("engine1", should_fail=True)
        engine2 = MockSTTEngine("engine2", should_fail=False)
        
        manager.register_engine(engine1, priority=10)
        manager.register_engine(engine2, priority=5)
        
        audio, sample_rate = sample_audio_16khz
        manager.transcribe(audio, sample_rate)
        
        health1 = manager.get_engine_health("engine1")
        health2 = manager.get_engine_health("engine2")
        
        assert health1["total_failures"] == 1
        assert health2["total_successes"] == 1
    
    def test_skip_unhealthy_engine(self, sample_audio_16khz):
        """Test that unhealthy engines are skipped."""
        manager = STTEngineManager()
        engine1 = MockSTTEngine("engine1", should_fail=True)
        engine2 = MockSTTEngine("engine2", should_fail=False)
        
        manager.register_engine(engine1, priority=10)
        manager.register_engine(engine2, priority=5)
        
        audio, sample_rate = sample_audio_16khz
        
        # Make engine1 unhealthy by failing 3 times
        for _ in range(3):
            try:
                manager.transcribe(audio, sample_rate)
            except:
                pass
        
        # Now engine1 should be unhealthy and skipped
        engine1.should_fail = False  # Fix engine1
        result = manager.transcribe(audio, sample_rate)
        
        # Should still use engine2 because engine1 is unhealthy
        assert result.engine_name == "engine2"
    
    def test_fallback_disabled(self, sample_audio_16khz):
        """Test that fallback can be disabled."""
        manager = STTEngineManager()
        engine1 = MockSTTEngine("engine1", should_fail=True)
        engine2 = MockSTTEngine("engine2", should_fail=False)
        
        manager.register_engine(engine1, priority=10)
        manager.register_engine(engine2, priority=5)
        manager.set_fallback_enabled(False)
        
        audio, sample_rate = sample_audio_16khz
        
        # Should fail immediately without trying engine2
        with pytest.raises(TranscriptionException):
            manager.transcribe(audio, sample_rate)
        
        assert engine1.transcribe_count == 1
        assert engine2.transcribe_count == 0
    
    def test_cleanup_all_engines(self):
        """Test cleanup of all engines."""
        manager = STTEngineManager()
        engine1 = MockSTTEngine("engine1")
        engine2 = MockSTTEngine("engine2")
        
        manager.register_engine(engine1, priority=10)
        manager.register_engine(engine2, priority=5)
        
        manager.cleanup()
        
        assert len(manager.get_available_engines()) == 0
        assert manager.get_active_engine() is None
    
    def test_replace_existing_engine(self):
        """Test replacing an already registered engine."""
        manager = STTEngineManager()
        engine1 = MockSTTEngine("engine1")
        engine2 = MockSTTEngine("engine1")  # Same name
        
        manager.register_engine(engine1, priority=10)
        manager.register_engine(engine2, priority=5)
        
        # Should have replaced engine1 with engine2
        assert manager.get_engine_by_name("engine1") == engine2
        assert len(manager.get_available_engines()) == 1
