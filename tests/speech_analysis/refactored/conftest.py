"""Pytest configuration and fixtures for refactored STT tests."""

import pytest
import numpy as np
from hypothesis import settings

# Configure Hypothesis for thorough testing
settings.register_profile("ci", max_examples=100, deadline=None)
settings.register_profile("dev", max_examples=50, deadline=None)
settings.register_profile("thorough", max_examples=200, deadline=None)

# Load the appropriate profile
settings.load_profile("dev")


@pytest.fixture
def sample_audio_16khz():
    """Generate sample audio at 16kHz."""
    duration = 3.0  # seconds
    sample_rate = 16000
    samples = int(duration * sample_rate)
    
    # Generate sine wave (440 Hz - A4 note)
    t = np.linspace(0, duration, samples)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    return audio, sample_rate


@pytest.fixture
def silent_audio_16khz():
    """Generate silent audio at 16kHz."""
    duration = 3.0  # seconds
    sample_rate = 16000
    samples = int(duration * sample_rate)
    
    audio = np.zeros(samples, dtype=np.float32)
    
    return audio, sample_rate


@pytest.fixture
def noisy_audio_16khz():
    """Generate noisy audio at 16kHz."""
    duration = 3.0  # seconds
    sample_rate = 16000
    samples = int(duration * sample_rate)
    
    # Generate sine wave with noise
    t = np.linspace(0, duration, samples)
    signal = np.sin(2 * np.pi * 440 * t)
    noise = np.random.normal(0, 0.1, samples)
    audio = (signal + noise).astype(np.float32)
    
    return audio, sample_rate


@pytest.fixture
def sample_transcription_segments():
    """Generate sample transcription segments."""
    from src.speech_analysis.refactored.models.transcription import TranscriptionSegment
    
    segments = [
        TranscriptionSegment(
            text="Xin chào",
            start_time=0.0,
            end_time=1.0,
            confidence=0.95,
            language="vi",
        ),
        TranscriptionSegment(
            text="Tôi là trợ lý AI",
            start_time=1.0,
            end_time=3.0,
            confidence=0.92,
            language="vi",
        ),
        TranscriptionSegment(
            text="Rất vui được gặp bạn",
            start_time=3.0,
            end_time=5.0,
            confidence=0.88,
            language="vi",
        ),
    ]
    
    return segments


@pytest.fixture
def sample_config():
    """Generate sample STT configuration."""
    from src.speech_analysis.refactored.models.config import (
        STTConfig,
        EngineType,
        DeviceType,
    )
    
    return STTConfig.create_profile("balanced")


@pytest.fixture
def mock_audio_device():
    """Generate mock audio device."""
    from src.speech_analysis.refactored.models.audio import AudioDevice
    
    return AudioDevice(
        index=0,
        name="Mock Microphone",
        max_input_channels=2,
        default_sample_rate=16000.0,
        is_default=True,
    )
