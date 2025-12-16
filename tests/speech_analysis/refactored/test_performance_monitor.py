"""Tests for performance monitoring."""

import pytest
import time
import threading
from datetime import datetime

from src.speech_analysis.refactored.implementations import (
    PerformanceMonitor,
    PerformanceConfig,
)
from src.speech_analysis.refactored.models.performance import (
    PerformanceMetrics,
    ResourceStatus,
)


class TestPerformanceConfig:
    """Tests for PerformanceConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PerformanceConfig()
        
        assert config.max_rtf == 1.5
        assert config.max_cpu_percent == 80.0
        assert config.max_memory_mb == 2000.0
        assert config.history_size == 100
        assert config.warning_threshold == 0.8
        assert config.critical_threshold == 0.95
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PerformanceConfig(
            max_rtf=2.0,
            max_cpu_percent=90.0,
            max_memory_mb=3000.0,
            history_size=50,
        )
        
        assert config.max_rtf == 2.0
        assert config.max_cpu_percent == 90.0
        assert config.max_memory_mb == 3000.0
        assert config.history_size == 50
    
    def test_invalid_max_rtf(self):
        """Test that invalid max_rtf raises error."""
        with pytest.raises(ValueError, match="max_rtf must be positive"):
            PerformanceConfig(max_rtf=0)
        
        with pytest.raises(ValueError, match="max_rtf must be positive"):
            PerformanceConfig(max_rtf=-1.0)
    
    def test_invalid_max_cpu(self):
        """Test that invalid max_cpu_percent raises error."""
        with pytest.raises(ValueError, match="max_cpu_percent must be between"):
            PerformanceConfig(max_cpu_percent=0)
        
        with pytest.raises(ValueError, match="max_cpu_percent must be between"):
            PerformanceConfig(max_cpu_percent=101)
    
    def test_invalid_max_memory(self):
        """Test that invalid max_memory_mb raises error."""
        with pytest.raises(ValueError, match="max_memory_mb must be positive"):
            PerformanceConfig(max_memory_mb=0)
        
        with pytest.raises(ValueError, match="max_memory_mb must be positive"):
            PerformanceConfig(max_memory_mb=-100)
    
    def test_invalid_history_size(self):
        """Test that invalid history_size raises error."""
        with pytest.raises(ValueError, match="history_size must be positive"):
            PerformanceConfig(history_size=0)


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor."""
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = PerformanceMonitor()
        
        assert not monitor._monitoring
        assert monitor._start_time is None
        assert len(monitor._rtf_history) == 0
        assert len(monitor._cpu_history) == 0
        assert len(monitor._memory_history) == 0
    
    def test_initialization_with_config(self):
        """Test monitor initialization with custom config."""
        config = PerformanceConfig(max_rtf=2.0, history_size=50)
        monitor = PerformanceMonitor(config)
        
        assert monitor.config.max_rtf == 2.0
        assert monitor.config.history_size == 50
    
    def test_start_monitoring(self):
        """Test starting monitoring."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        assert monitor._monitoring
        assert monitor._start_time is not None
        assert monitor._total_audio_duration == 0.0
        assert monitor._total_processing_time == 0.0
        assert monitor._processing_count == 0
    
    def test_start_monitoring_idempotent(self):
        """Test that starting monitoring multiple times is safe."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        start_time1 = monitor._start_time
        
        time.sleep(0.01)
        monitor.start_monitoring()
        start_time2 = monitor._start_time
        
        # Should not change start time
        assert start_time1 == start_time2
    
    def test_stop_monitoring(self):
        """Test stopping monitoring."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Record some processing to have data
        monitor.record_processing(3.0, 2.0)
        
        report = monitor.stop_monitoring()
        
        assert not monitor._monitoring
        assert "monitoring_duration_seconds" in report
        assert report["monitoring_duration_seconds"] >= 0
        assert "total_audio_processed_seconds" in report
        assert "processing_count" in report
    
    def test_stop_monitoring_without_start(self):
        """Test stopping monitoring without starting returns empty report."""
        monitor = PerformanceMonitor()
        report = monitor.stop_monitoring()
        
        assert report == {}
    
    def test_record_processing(self):
        """Test recording processing metrics."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        audio_duration = 3.0
        processing_time = 2.0
        
        monitor.record_processing(audio_duration, processing_time)
        
        assert monitor._processing_count == 1
        assert monitor._total_audio_duration == 3.0
        assert monitor._total_processing_time == 2.0
        assert len(monitor._rtf_history) == 1
        assert monitor._rtf_history[0] == pytest.approx(2.0 / 3.0)
    
    def test_record_processing_multiple(self):
        """Test recording multiple processing events."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        for i in range(5):
            monitor.record_processing(3.0, 2.0)
        
        assert monitor._processing_count == 5
        assert monitor._total_audio_duration == 15.0
        assert monitor._total_processing_time == 10.0
        assert len(monitor._rtf_history) == 5
    
    def test_record_processing_invalid_audio_duration(self):
        """Test that invalid audio_duration raises error."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        with pytest.raises(ValueError, match="audio_duration must be positive"):
            monitor.record_processing(0, 1.0)
        
        with pytest.raises(ValueError, match="audio_duration must be positive"):
            monitor.record_processing(-1.0, 1.0)
    
    def test_record_processing_invalid_processing_time(self):
        """Test that invalid processing_time raises error."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        with pytest.raises(ValueError, match="processing_time must be non-negative"):
            monitor.record_processing(3.0, -1.0)
    
    def test_record_processing_without_monitoring(self):
        """Test that recording without monitoring is safe (no-op)."""
        monitor = PerformanceMonitor()
        
        # Should not raise error
        monitor.record_processing(3.0, 2.0)
        
        assert monitor._processing_count == 0
    
    def test_get_metrics(self):
        """Test getting current metrics."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        monitor.record_processing(3.0, 2.0)
        
        metrics = monitor.get_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.rtf == pytest.approx(2.0 / 3.0)
        assert metrics.cpu_usage >= 0
        assert metrics.memory_usage_mb > 0
        assert metrics.latency_ms == pytest.approx(2000.0)
    
    def test_get_metrics_before_recording(self):
        """Test getting metrics before any recording."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        metrics = monitor.get_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.rtf == 0.0
        assert metrics.latency_ms == 0.0
    
    def test_get_average_metrics(self):
        """Test getting average metrics."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Record multiple events with different RTF
        monitor.record_processing(3.0, 1.5)  # RTF = 0.5
        monitor.record_processing(3.0, 3.0)  # RTF = 1.0
        monitor.record_processing(3.0, 4.5)  # RTF = 1.5
        
        avg_metrics = monitor.get_average_metrics()
        
        assert isinstance(avg_metrics, PerformanceMetrics)
        assert avg_metrics.rtf == pytest.approx(1.0)  # Average of 0.5, 1.0, 1.5
    
    def test_check_resource_limits_normal(self):
        """Test resource limit checking with normal values."""
        config = PerformanceConfig(max_rtf=2.0, max_cpu_percent=80.0)
        monitor = PerformanceMonitor(config)
        monitor.start_monitoring()
        
        # Record low RTF values
        for _ in range(10):
            monitor.record_processing(3.0, 1.5)  # RTF = 0.5
        
        status = monitor.check_resource_limits()
        assert status == ResourceStatus.NORMAL
    
    def test_check_resource_limits_warning(self):
        """Test resource limit checking with warning values."""
        config = PerformanceConfig(max_rtf=2.0, warning_threshold=0.8)
        monitor = PerformanceMonitor(config)
        monitor.start_monitoring()
        
        # Record RTF at 85% of limit (1.7 / 2.0 = 0.85)
        for _ in range(10):
            monitor.record_processing(3.0, 5.1)  # RTF = 1.7
        
        status = monitor.check_resource_limits()
        assert status == ResourceStatus.WARNING
    
    def test_check_resource_limits_critical(self):
        """Test resource limit checking with critical values."""
        config = PerformanceConfig(max_rtf=2.0, critical_threshold=0.95)
        monitor = PerformanceMonitor(config)
        monitor.start_monitoring()
        
        # Record RTF at 100% of limit
        for _ in range(10):
            monitor.record_processing(3.0, 6.0)  # RTF = 2.0
        
        status = monitor.check_resource_limits()
        assert status == ResourceStatus.CRITICAL
    
    def test_check_resource_limits_unknown(self):
        """Test resource limit checking with no data."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        status = monitor.check_resource_limits()
        assert status == ResourceStatus.UNKNOWN
    
    def test_get_rtf_history(self):
        """Test getting RTF history."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        monitor.record_processing(3.0, 1.5)
        monitor.record_processing(3.0, 3.0)
        monitor.record_processing(3.0, 4.5)
        
        history = monitor.get_rtf_history()
        
        assert len(history) == 3
        assert history[0] == pytest.approx(0.5)
        assert history[1] == pytest.approx(1.0)
        assert history[2] == pytest.approx(1.5)
    
    def test_get_cpu_history(self):
        """Test getting CPU history."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        monitor.record_processing(3.0, 2.0)
        monitor.record_processing(3.0, 2.0)
        
        history = monitor.get_cpu_history()
        
        assert len(history) == 2
        assert all(cpu >= 0 for cpu in history)
    
    def test_get_memory_history(self):
        """Test getting memory history."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        monitor.record_processing(3.0, 2.0)
        monitor.record_processing(3.0, 2.0)
        
        history = monitor.get_memory_history()
        
        assert len(history) == 2
        assert all(mem > 0 for mem in history)
    
    def test_should_adjust_performance_normal(self):
        """Test performance adjustment check with normal status."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Record low RTF
        for _ in range(10):
            monitor.record_processing(3.0, 1.5)
        
        assert not monitor.should_adjust_performance()
    
    def test_should_adjust_performance_warning(self):
        """Test performance adjustment check with warning status."""
        config = PerformanceConfig(max_rtf=2.0, warning_threshold=0.8)
        monitor = PerformanceMonitor(config)
        monitor.start_monitoring()
        
        # Record high RTF
        for _ in range(10):
            monitor.record_processing(3.0, 5.1)
        
        assert monitor.should_adjust_performance()
    
    def test_should_adjust_performance_critical(self):
        """Test performance adjustment check with critical status."""
        config = PerformanceConfig(max_rtf=2.0)
        monitor = PerformanceMonitor(config)
        monitor.start_monitoring()
        
        # Record very high RTF
        for _ in range(10):
            monitor.record_processing(3.0, 6.0)
        
        assert monitor.should_adjust_performance()
    
    def test_reset_metrics(self):
        """Test resetting metrics."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Record some data
        monitor.record_processing(3.0, 2.0)
        monitor.record_processing(3.0, 2.0)
        
        assert monitor._processing_count == 2
        assert len(monitor._rtf_history) == 2
        
        # Reset
        monitor.reset_metrics()
        
        assert monitor._processing_count == 0
        assert len(monitor._rtf_history) == 0
        assert len(monitor._cpu_history) == 0
        assert len(monitor._memory_history) == 0
        assert monitor._total_audio_duration == 0.0
        assert monitor._total_processing_time == 0.0
    
    def test_reset_metrics_preserves_monitoring_state(self):
        """Test that reset preserves monitoring state."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        assert monitor._monitoring
        start_time_before = monitor._start_time
        
        monitor.reset_metrics()
        
        assert monitor._monitoring
        assert monitor._start_time >= start_time_before
    
    def test_history_size_limit(self):
        """Test that history respects size limit."""
        config = PerformanceConfig(history_size=5)
        monitor = PerformanceMonitor(config)
        monitor.start_monitoring()
        
        # Record more than history size
        for i in range(10):
            monitor.record_processing(3.0, 2.0)
        
        # Should only keep last 5
        assert len(monitor._rtf_history) == 5
        assert len(monitor._cpu_history) == 5
        assert len(monitor._memory_history) == 5
    
    def test_thread_safety(self):
        """Test that monitor is thread-safe."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        def worker():
            for _ in range(10):
                monitor.record_processing(3.0, 2.0)
        
        # Create multiple threads
        threads = [threading.Thread(target=worker) for _ in range(5)]
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Should have recorded all events
        assert monitor._processing_count == 50
        assert len(monitor._rtf_history) == 50
    
    def test_complete_workflow(self):
        """Test complete monitoring workflow."""
        monitor = PerformanceMonitor()
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor._monitoring
        
        # Record multiple processing events
        for i in range(5):
            monitor.record_processing(3.0, 2.0 + i * 0.5)
        
        # Get current metrics
        metrics = monitor.get_metrics()
        assert metrics.rtf > 0
        
        # Get average metrics
        avg_metrics = monitor.get_average_metrics()
        assert avg_metrics.rtf > 0
        
        # Check resource status
        status = monitor.check_resource_limits()
        assert status in (ResourceStatus.NORMAL, ResourceStatus.WARNING, ResourceStatus.CRITICAL)
        
        # Get histories
        rtf_history = monitor.get_rtf_history()
        assert len(rtf_history) == 5
        
        # Stop monitoring
        report = monitor.stop_monitoring()
        assert report["processing_count"] == 5
        assert report["total_audio_processed_seconds"] == 15.0
        assert "average_rtf" in report
        assert "max_rtf" in report
        
        assert not monitor._monitoring
