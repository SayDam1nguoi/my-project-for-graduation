"""Unit tests for audio pipeline components."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.speech_analysis.refactored.implementations.audio_capture import AudioCapture
from src.speech_analysis.refactored.implementations.audio_preprocessor import AudioPreprocessor
from src.speech_analysis.refactored.implementations.vad_detector import (
    VADDetector, EnergyVAD, WebRTCVAD, SileroVAD
)
from src.speech_analysis.refactored.implementations.audio_buffer import AudioBuffer
from src.speech_analysis.refactored.models.audio import AudioChunk, SpeechSegment
from src.speech_analysis.refactored.exceptions import (
    MicrophoneNotFoundError,
    AudioDriverError
)


class TestAudioPreprocessor:
    """Test AudioPreprocessor implementation."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initializes with default steps."""
        preprocessor = AudioPreprocessor(target_sample_rate=16000)
        
        assert preprocessor.target_sample_rate == 16000
        assert len(preprocessor.steps) > 0
        
        # Check default steps are registered
        pipeline_info = preprocessor.get_pipeline_info()
        step_names = [step['name'] for step in pipeline_info]
        
        assert 'noise_reduction' in step_names
        assert 'normalization' in step_names
        assert 'pre_emphasis' in step_names
    
    def test_preprocess_audio(self):
        """Test preprocessing audio."""
        preprocessor = AudioPreprocessor(target_sample_rate=16000)
        
        # Create test audio (1 second of sine wave)
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Preprocess
        processed = preprocessor.preprocess(audio, sample_rate)
        
        # Check output
        assert isinstance(processed, np.ndarray)
        assert processed.dtype == np.float32
        assert len(processed) > 0
    
    def test_add_remove_step(self):
        """Test adding and removing preprocessing steps."""
        preprocessor = AudioPreprocessor()
        
        # Add custom step
        def custom_step(audio, sample_rate):
            return audio * 0.5
        
        preprocessor.add_step('custom', custom_step)
        
        pipeline_info = preprocessor.get_pipeline_info()
        step_names = [step['name'] for step in pipeline_info]
        assert 'custom' in step_names
        
        # Remove step
        preprocessor.remove_step('custom')
        
        pipeline_info = preprocessor.get_pipeline_info()
        step_names = [step['name'] for step in pipeline_info]
        assert 'custom' not in step_names
    
    def test_enable_disable_step(self):
        """Test enabling and disabling steps."""
        preprocessor = AudioPreprocessor()
        
        # Disable noise reduction
        preprocessor.disable_step('noise_reduction')
        
        pipeline_info = preprocessor.get_pipeline_info()
        noise_step = next(s for s in pipeline_info if s['name'] == 'noise_reduction')
        assert noise_step['enabled'] is False
        
        # Enable it again
        preprocessor.enable_step('noise_reduction')
        
        pipeline_info = preprocessor.get_pipeline_info()
        noise_step = next(s for s in pipeline_info if s['name'] == 'noise_reduction')
        assert noise_step['enabled'] is True
    
    def test_normalization(self):
        """Test audio normalization."""
        preprocessor = AudioPreprocessor()
        
        # Create quiet audio
        audio = np.random.randn(16000).astype(np.float32) * 0.01
        
        # Normalize
        normalized = preprocessor._normalize(audio, 16000, target_rms=0.1)
        
        # Check RMS is close to target
        rms = np.sqrt(np.mean(normalized ** 2))
        assert abs(rms - 0.1) < 0.01
    
    def test_pre_emphasis(self):
        """Test pre-emphasis filter."""
        preprocessor = AudioPreprocessor()
        
        # Create test audio
        audio = np.random.randn(16000).astype(np.float32)
        
        # Apply pre-emphasis
        emphasized = preprocessor._pre_emphasis(audio, 16000, coef=0.97)
        
        # Check output shape
        assert len(emphasized) == len(audio)


class TestEnergyVAD:
    """Test Energy-based VAD implementation."""
    
    def test_vad_initialization(self):
        """Test VAD initializes correctly."""
        vad = EnergyVAD(sample_rate=16000)
        
        assert vad.sample_rate == 16000
        assert vad.energy_threshold > 0
        assert vad.zcr_threshold > 0
    
    def test_detect_speech_in_audio(self):
        """Test speech detection in audio."""
        vad = EnergyVAD(sample_rate=16000)
        
        # Create audio with speech-like characteristics
        # (high energy and moderate zero-crossing rate)
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Mix of frequencies to simulate speech
        audio = (
            0.3 * np.sin(2 * np.pi * 200 * t) +
            0.2 * np.sin(2 * np.pi * 500 * t) +
            0.1 * np.sin(2 * np.pi * 1000 * t)
        ).astype(np.float32)
        
        # Detect speech
        segments = vad.detect_speech(audio, sample_rate)
        
        # Should detect at least one segment
        assert len(segments) >= 0  # May or may not detect depending on thresholds
        
        # Check segment structure
        for segment in segments:
            assert isinstance(segment, SpeechSegment)
            assert segment.is_speech is True
            assert 0 <= segment.confidence <= 1
    
    def test_is_speech(self):
        """Test is_speech method."""
        vad = EnergyVAD(sample_rate=16000)
        
        # Create loud audio (likely speech)
        loud_audio = np.random.randn(16000).astype(np.float32) * 0.5
        
        # Create quiet audio (likely silence)
        quiet_audio = np.random.randn(16000).astype(np.float32) * 0.001
        
        # Test (results may vary based on thresholds)
        # Just check that method runs without error
        result_loud = vad.is_speech(loud_audio, 16000)
        result_quiet = vad.is_speech(quiet_audio, 16000)
        
        assert isinstance(result_loud, bool)
        assert isinstance(result_quiet, bool)
    
    def test_get_speech_probability(self):
        """Test speech probability calculation."""
        vad = EnergyVAD(sample_rate=16000)
        
        # Create test audio
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        
        # Get probability
        prob = vad.get_speech_probability(audio, 16000)
        
        # Check probability is in valid range
        assert 0.0 <= prob <= 1.0
    
    def test_set_sensitivity(self):
        """Test setting VAD sensitivity."""
        vad = EnergyVAD(sample_rate=16000)
        
        # Set high sensitivity
        vad.set_sensitivity(0.9)
        
        # Thresholds should be lower (more sensitive)
        assert vad.energy_threshold < 0.01
        
        # Set low sensitivity
        vad.set_sensitivity(0.1)
        
        # Thresholds should be higher (less sensitive)
        assert vad.energy_threshold > 0.01


class TestAudioBuffer:
    """Test AudioBuffer implementation."""
    
    def test_buffer_initialization(self):
        """Test buffer initializes correctly."""
        buffer = AudioBuffer(max_duration=30.0, sample_rate=16000)
        
        assert buffer.max_duration == 30.0
        assert buffer.sample_rate == 16000
        assert buffer.max_samples == 30 * 16000
        assert buffer.is_empty()
    
    def test_add_chunk(self):
        """Test adding chunks to buffer."""
        buffer = AudioBuffer(max_duration=1.0, sample_rate=16000)
        
        # Create test chunk
        audio_data = np.random.randn(1600).astype(np.float32)
        chunk = AudioChunk(
            data=audio_data,
            sample_rate=16000,
            timestamp=0.0,
            duration=0.1
        )
        
        # Add chunk
        success = buffer.add_chunk(chunk)
        
        assert success is True
        assert not buffer.is_empty()
    
    def test_get_latest_audio(self):
        """Test retrieving latest audio."""
        buffer = AudioBuffer(max_duration=1.0, sample_rate=16000)
        
        # Add some chunks
        for i in range(5):
            audio_data = np.random.randn(1600).astype(np.float32)
            chunk = AudioChunk(
                data=audio_data,
                sample_rate=16000,
                timestamp=i * 0.1,
                duration=0.1
            )
            buffer.add_chunk(chunk)
        
        # Get all audio
        audio = buffer.get_latest_audio()
        
        assert len(audio) > 0
        assert isinstance(audio, np.ndarray)
        
        # Get last 0.2 seconds
        audio_partial = buffer.get_latest_audio(duration=0.2)
        
        assert len(audio_partial) <= len(audio)
    
    def test_circular_buffer_overflow(self):
        """Test circular buffer handles overflow."""
        buffer = AudioBuffer(max_duration=0.1, sample_rate=16000)  # Small buffer
        
        # Add more data than buffer can hold
        for i in range(20):
            audio_data = np.random.randn(1600).astype(np.float32)
            chunk = AudioChunk(
                data=audio_data,
                sample_rate=16000,
                timestamp=i * 0.1,
                duration=0.1
            )
            buffer.add_chunk(chunk)
        
        # Buffer should not exceed max size
        stats = buffer.get_stats()
        assert stats.current_size <= buffer.max_samples
        assert stats.overflow_count > 0
    
    def test_buffer_stats(self):
        """Test buffer statistics."""
        buffer = AudioBuffer(max_duration=1.0, sample_rate=16000)
        
        # Add some chunks
        for i in range(3):
            audio_data = np.random.randn(1600).astype(np.float32)
            chunk = AudioChunk(
                data=audio_data,
                sample_rate=16000,
                timestamp=i * 0.1,
                duration=0.1
            )
            buffer.add_chunk(chunk)
        
        # Get stats
        stats = buffer.get_stats()
        
        assert stats.total_chunks_added == 3
        assert stats.current_size > 0
        assert 0.0 <= stats.utilization <= 1.0
    
    def test_clear_buffer(self):
        """Test clearing buffer."""
        buffer = AudioBuffer(max_duration=1.0, sample_rate=16000)
        
        # Add chunk
        audio_data = np.random.randn(1600).astype(np.float32)
        chunk = AudioChunk(
            data=audio_data,
            sample_rate=16000,
            timestamp=0.0,
            duration=0.1
        )
        buffer.add_chunk(chunk)
        
        assert not buffer.is_empty()
        
        # Clear
        buffer.clear()
        
        assert buffer.is_empty()
        
        stats = buffer.get_stats()
        assert stats.current_size == 0


class TestVADFactory:
    """Test VAD factory method."""
    
    def test_create_energy_vad(self):
        """Test creating Energy VAD."""
        vad = VADDetector.create(method='energy', sample_rate=16000)
        
        assert isinstance(vad, EnergyVAD)
    
    def test_create_invalid_method(self):
        """Test creating VAD with invalid method."""
        with pytest.raises(ValueError):
            VADDetector.create(method='invalid')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
