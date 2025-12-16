"""Unit tests for interface definitions."""

import pytest
from abc import ABC
import inspect

from src.speech_analysis.refactored.interfaces.audio import (
    IAudioCapture,
    IAudioPreprocessor,
    IVADDetector,
)
from src.speech_analysis.refactored.interfaces.engine import (
    ISTTEngine,
    ISTTEngineManager,
)
from src.speech_analysis.refactored.interfaces.processing import (
    IHallucinationFilter,
    ITextPostProcessor,
    IQualityAnalyzer,
)
from src.speech_analysis.refactored.interfaces.monitoring import (
    IPerformanceMonitor,
)
from src.speech_analysis.refactored.interfaces.config import (
    IConfigManager,
)


class TestInterfaceDefinitions:
    """Tests to verify interface definitions are correct."""
    
    def test_audio_interfaces_are_abstract(self):
        """Test that audio interfaces are abstract base classes."""
        assert issubclass(IAudioCapture, ABC)
        assert issubclass(IAudioPreprocessor, ABC)
        assert issubclass(IVADDetector, ABC)
    
    def test_engine_interfaces_are_abstract(self):
        """Test that engine interfaces are abstract base classes."""
        assert issubclass(ISTTEngine, ABC)
        assert issubclass(ISTTEngineManager, ABC)
    
    def test_processing_interfaces_are_abstract(self):
        """Test that processing interfaces are abstract base classes."""
        assert issubclass(IHallucinationFilter, ABC)
        assert issubclass(ITextPostProcessor, ABC)
        assert issubclass(IQualityAnalyzer, ABC)
    
    def test_monitoring_interfaces_are_abstract(self):
        """Test that monitoring interfaces are abstract base classes."""
        assert issubclass(IPerformanceMonitor, ABC)
    
    def test_config_interfaces_are_abstract(self):
        """Test that config interfaces are abstract base classes."""
        assert issubclass(IConfigManager, ABC)
    
    def test_audio_capture_interface_methods(self):
        """Test IAudioCapture has required methods."""
        required_methods = [
            'start_capture',
            'stop_capture',
            'get_audio_chunk',
            'is_capturing',
            'list_devices',
            'get_current_device',
        ]
        
        for method_name in required_methods:
            assert hasattr(IAudioCapture, method_name)
            method = getattr(IAudioCapture, method_name)
            assert callable(method)
    
    def test_stt_engine_interface_methods(self):
        """Test ISTTEngine has required methods."""
        required_methods = [
            'initialize',
            'transcribe_chunk',
            'transcribe_stream',
            'is_available',
            'get_engine_info',
            'cleanup',
            'get_name',
        ]
        
        for method_name in required_methods:
            assert hasattr(ISTTEngine, method_name)
            method = getattr(ISTTEngine, method_name)
            assert callable(method)
    
    def test_engine_manager_interface_methods(self):
        """Test ISTTEngineManager has required methods."""
        required_methods = [
            'register_engine',
            'unregister_engine',
            'transcribe',
            'get_active_engine',
            'switch_engine',
            'get_available_engines',
            'get_engine_by_name',
            'get_engine_health',
        ]
        
        for method_name in required_methods:
            assert hasattr(ISTTEngineManager, method_name)
            method = getattr(ISTTEngineManager, method_name)
            assert callable(method)
    
    def test_hallucination_filter_interface_methods(self):
        """Test IHallucinationFilter has required methods."""
        required_methods = [
            'is_hallucination',
            'filter_segments',
            'add_pattern',
            'remove_pattern',
            'get_filter_stats',
            'reset_stats',
        ]
        
        for method_name in required_methods:
            assert hasattr(IHallucinationFilter, method_name)
            method = getattr(IHallucinationFilter, method_name)
            assert callable(method)
    
    def test_performance_monitor_interface_methods(self):
        """Test IPerformanceMonitor has required methods."""
        required_methods = [
            'start_monitoring',
            'stop_monitoring',
            'record_processing',
            'get_metrics',
            'get_average_metrics',
            'check_resource_limits',
            'get_rtf_history',
            'get_cpu_history',
            'get_memory_history',
            'should_adjust_performance',
            'reset_metrics',
        ]
        
        for method_name in required_methods:
            assert hasattr(IPerformanceMonitor, method_name)
            method = getattr(IPerformanceMonitor, method_name)
            assert callable(method)
    
    def test_config_manager_interface_methods(self):
        """Test IConfigManager has required methods."""
        required_methods = [
            'load_config',
            'save_config',
            'validate_config',
            'get_profile',
            'list_profiles',
            'merge_configs',
            'get_default_config',
            'check_compatibility',
        ]
        
        for method_name in required_methods:
            assert hasattr(IConfigManager, method_name)
            method = getattr(IConfigManager, method_name)
            assert callable(method)
    
    def test_cannot_instantiate_interfaces(self):
        """Test that interfaces cannot be instantiated directly."""
        with pytest.raises(TypeError):
            IAudioCapture()
        
        with pytest.raises(TypeError):
            ISTTEngine()
        
        with pytest.raises(TypeError):
            ISTTEngineManager()
        
        with pytest.raises(TypeError):
            IHallucinationFilter()
        
        with pytest.raises(TypeError):
            IPerformanceMonitor()
        
        with pytest.raises(TypeError):
            IConfigManager()
