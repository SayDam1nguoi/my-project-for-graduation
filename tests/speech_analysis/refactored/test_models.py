"""Unit tests for data models."""

import pytest
import numpy as np
from datetime import datetime

from src.speech_analysis.refactored.models.audio import (
    AudioChunk,
    AudioDevice,
    SpeechSegment,
)
from src.speech_analysis.refactored.models.transcription import (
    TranscriptionSegment,
    TranscriptionResult,
)
from src.speech_analysis.refactored.models.config import (
    STTConfig,
    AudioConfig,
    EngineConfig,
    EngineType,
    DeviceType,
)
from src.speech_analysis.refactored.models.performance import (
    PerformanceMetrics,
    ResourceStatus,
)


class TestAudioModels:
    """Tests for audio-related models."""
    
    def test_audio_chunk_creation(self):
        """Test AudioChunk creation with valid data."""
        audio_data = np.random.randn(16000).astype(np.float32)
        chunk = AudioChunk(
            data=audio_data,
            sample_rate=16000,
            timestamp=0.0,
            duration=1.0,
        )
        
        assert chunk.sample_rate == 16000
        assert chunk.duration == 1.0
        assert len(chunk.data) == 16000
    
    def test_audio_chunk_validation(self):
        """Test AudioChunk validation."""
        audio_data = np.random.randn(16000).astype(np.float32)
        
        # Invalid sample rate
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            AudioChunk(
                data=audio_data,
                sample_rate=-1,
                timestamp=0.0,
                duration=1.0,
            )
        
        # Invalid duration
        with pytest.raises(ValueError, match="Duration must be positive"):
            AudioChunk(
                data=audio_data,
                sample_rate=16000,
                timestamp=0.0,
                duration=-1.0,
            )
    
    def test_speech_segment_creation(self):
        """Test SpeechSegment creation."""
        audio_data = np.random.randn(16000).astype(np.float32)
        segment = SpeechSegment(
            audio=audio_data,
            start_time=0.0,
            end_time=1.0,
            confidence=0.95,
            is_speech=True,
        )
        
        assert segment.duration == 1.0
        assert segment.confidence == 0.95
        assert segment.is_speech is True
    
    def test_speech_segment_validation(self):
        """Test SpeechSegment validation."""
        audio_data = np.random.randn(16000).astype(np.float32)
        
        # Invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            SpeechSegment(
                audio=audio_data,
                start_time=0.0,
                end_time=1.0,
                confidence=1.5,
                is_speech=True,
            )
        
        # Invalid time range
        with pytest.raises(ValueError, match="End time must be greater than start time"):
            SpeechSegment(
                audio=audio_data,
                start_time=1.0,
                end_time=0.5,
                confidence=0.9,
                is_speech=True,
            )


class TestTranscriptionModels:
    """Tests for transcription-related models."""
    
    def test_transcription_segment_creation(self):
        """Test TranscriptionSegment creation."""
        segment = TranscriptionSegment(
            text="Xin chào",
            start_time=0.0,
            end_time=1.0,
            confidence=0.95,
            language="vi",
        )
        
        assert segment.text == "Xin chào"
        assert segment.duration == 1.0
        assert segment.word_count == 2
    
    def test_transcription_result_creation(self, sample_transcription_segments):
        """Test TranscriptionResult creation."""
        result = TranscriptionResult(
            text="Xin chào. Tôi là trợ lý AI. Rất vui được gặp bạn.",
            segments=sample_transcription_segments,
            language="vi",
            confidence=0.92,
            processing_time=1.5,
            engine_name="whisper",
        )
        
        assert result.segment_count == 3
        assert result.total_duration == 5.0
        assert 0.88 <= result.average_confidence <= 0.95


class TestConfigModels:
    """Tests for configuration models."""
    
    def test_audio_config_creation(self):
        """Test AudioConfig creation."""
        config = AudioConfig(
            sample_rate=16000,
            chunk_duration=3.0,
            overlap_duration=0.5,
        )
        
        assert config.sample_rate == 16000
        assert config.chunk_duration == 3.0
    
    def test_audio_config_validation(self):
        """Test AudioConfig validation."""
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            AudioConfig(sample_rate=-1)
    
    def test_engine_config_creation(self):
        """Test EngineConfig creation."""
        config = EngineConfig(
            engine_type=EngineType.FASTER_WHISPER,
            model_size="base",
            device=DeviceType.CPU,
        )
        
        assert config.engine_type == EngineType.FASTER_WHISPER
        assert config.model_size == "base"
    
    def test_stt_config_profiles(self):
        """Test STTConfig profile creation."""
        # Test accuracy profile
        accuracy_config = STTConfig.create_profile("accuracy")
        assert accuracy_config.engine.engine_type == EngineType.WHISPER
        assert accuracy_config.engine.model_size == "large-v3"
        
        # Test balanced profile
        balanced_config = STTConfig.create_profile("balanced")
        assert balanced_config.engine.engine_type == EngineType.FASTER_WHISPER
        assert balanced_config.engine.model_size == "base"
        
        # Test speed profile
        speed_config = STTConfig.create_profile("speed")
        assert speed_config.engine.model_size == "tiny"
        
        # Test invalid profile
        with pytest.raises(ValueError, match="Unknown profile"):
            STTConfig.create_profile("invalid")


class TestPerformanceModels:
    """Tests for performance models."""
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation."""
        metrics = PerformanceMetrics(
            rtf=1.2,
            cpu_usage=65.0,
            memory_usage_mb=800.0,
            latency_ms=150.0,
            throughput_ratio=0.83,
            timestamp=datetime.now(),
        )
        
        assert metrics.rtf == 1.2
        assert not metrics.is_realtime  # RTF >= 1.0
        assert metrics.cpu_status == ResourceStatus.WARNING
        assert metrics.memory_status == ResourceStatus.NORMAL
    
    def test_performance_metrics_validation(self):
        """Test PerformanceMetrics validation."""
        with pytest.raises(ValueError, match="RTF must be non-negative"):
            PerformanceMetrics(
                rtf=-1.0,
                cpu_usage=50.0,
                memory_usage_mb=500.0,
                latency_ms=100.0,
                throughput_ratio=1.0,
                timestamp=datetime.now(),
            )
        
        with pytest.raises(ValueError, match="CPU usage must be between 0 and 100"):
            PerformanceMetrics(
                rtf=1.0,
                cpu_usage=150.0,
                memory_usage_mb=500.0,
                latency_ms=100.0,
                throughput_ratio=1.0,
                timestamp=datetime.now(),
            )
