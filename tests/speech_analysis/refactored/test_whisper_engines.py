"""Tests for Whisper STT engine implementations."""

import pytest
import numpy as np

from src.speech_analysis.refactored.implementations import (
    OpenAIWhisperEngine,
    FasterWhisperEngine,
    VoskEngine,
)
from src.speech_analysis.refactored.models.config import EngineConfig, EngineType, DeviceType
from src.speech_analysis.refactored.interfaces.engine import EngineInfo


class TestOpenAIWhisperEngine:
    """Tests for OpenAI Whisper engine."""
    
    def test_engine_creation(self):
        """Test that OpenAI Whisper engine can be created."""
        engine = OpenAIWhisperEngine()
        assert engine is not None
        assert engine.get_name() == "openai-whisper"
    
    def test_engine_info(self):
        """Test that engine info is returned correctly."""
        engine = OpenAIWhisperEngine()
        
        # Create minimal config
        config = EngineConfig(
            engine_type=EngineType.WHISPER,
            model_size="tiny",
            device=DeviceType.CPU,
        )
        
        # Initialize (may fail if openai-whisper not installed, that's ok)
        try:
            engine.initialize(config)
        except:
            pass
        
        # Get engine info (should work even if not initialized)
        info = engine.get_engine_info()
        assert isinstance(info, EngineInfo)
        assert info.name == "openai-whisper"
        assert "vi" in info.supported_languages
        assert info.supports_timestamps is True
        assert info.supports_confidence is True
    
    def test_not_available_before_init(self):
        """Test that engine is not available before initialization."""
        engine = OpenAIWhisperEngine()
        assert engine.is_available() is False


class TestFasterWhisperEngine:
    """Tests for Faster-Whisper engine."""
    
    def test_engine_creation(self):
        """Test that Faster-Whisper engine can be created."""
        engine = FasterWhisperEngine()
        assert engine is not None
        assert engine.get_name() == "faster-whisper"
    
    def test_engine_info(self):
        """Test that engine info is returned correctly."""
        engine = FasterWhisperEngine()
        
        # Create minimal config
        config = EngineConfig(
            engine_type=EngineType.FASTER_WHISPER,
            model_size="tiny",
            device=DeviceType.CPU,
        )
        
        # Initialize (may fail if faster-whisper not installed, that's ok)
        try:
            engine.initialize(config)
        except:
            pass
        
        # Get engine info (should work even if not initialized)
        info = engine.get_engine_info()
        assert isinstance(info, EngineInfo)
        assert info.name == "faster-whisper"
        assert "vi" in info.supported_languages
        assert info.supports_timestamps is True
        assert info.supports_confidence is True
    
    def test_not_available_before_init(self):
        """Test that engine is not available before initialization."""
        engine = FasterWhisperEngine()
        assert engine.is_available() is False


class TestVoskEngine:
    """Tests for Vosk engine."""
    
    def test_engine_creation(self):
        """Test that Vosk engine can be created."""
        engine = VoskEngine()
        assert engine is not None
        assert engine.get_name() == "vosk"
    
    def test_engine_info(self):
        """Test that engine info is returned correctly."""
        engine = VoskEngine()
        
        # Create minimal config
        config = EngineConfig(
            engine_type=EngineType.VOSK,
            model_path="models/vosk-model-vn-0.4",
            device=DeviceType.CPU,
        )
        
        # Initialize (may fail if vosk not installed or model not found, that's ok)
        try:
            engine.initialize(config)
        except:
            pass
        
        # Get engine info (should work even if not initialized)
        info = engine.get_engine_info()
        assert isinstance(info, EngineInfo)
        assert info.name == "vosk"
        assert "vi" in info.supported_languages
        assert info.supports_streaming is True
        # Vosk has limited timestamp and confidence support
        assert info.supports_timestamps is False
        assert info.supports_confidence is False
    
    def test_not_available_before_init(self):
        """Test that engine is not available before initialization."""
        engine = VoskEngine()
        assert engine.is_available() is False


class TestEngineInterface:
    """Tests for engine interface compliance."""
    
    @pytest.mark.parametrize("engine_class", [
        OpenAIWhisperEngine,
        FasterWhisperEngine,
        VoskEngine,
    ])
    def test_engine_has_required_methods(self, engine_class):
        """Test that all engines implement required interface methods."""
        engine = engine_class()
        
        # Check that all required methods exist
        assert hasattr(engine, 'initialize')
        assert hasattr(engine, 'transcribe_chunk')
        assert hasattr(engine, 'transcribe_stream')
        assert hasattr(engine, 'is_available')
        assert hasattr(engine, 'get_engine_info')
        assert hasattr(engine, 'cleanup')
        assert hasattr(engine, 'get_name')
        
        # Check that methods are callable
        assert callable(engine.initialize)
        assert callable(engine.transcribe_chunk)
        assert callable(engine.transcribe_stream)
        assert callable(engine.is_available)
        assert callable(engine.get_engine_info)
        assert callable(engine.cleanup)
        assert callable(engine.get_name)
    
    @pytest.mark.parametrize("engine_class,expected_name", [
        (OpenAIWhisperEngine, "openai-whisper"),
        (FasterWhisperEngine, "faster-whisper"),
        (VoskEngine, "vosk"),
    ])
    def test_engine_name(self, engine_class, expected_name):
        """Test that engines return correct names."""
        engine = engine_class()
        assert engine.get_name() == expected_name
