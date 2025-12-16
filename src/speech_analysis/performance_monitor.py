"""
Performance Monitor Module

Monitors CPU usage, memory usage, and real-time factor for speech analysis.
Provides performance statistics and logging for optimization.
"""

import time
import psutil
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for a time period.
    
    Attributes:
        timestamp: When the metrics were collected
        cpu_percent: CPU usage percentage (0-100)
        memory_mb: Memory usage in MB
        real_time_factor: Processing speed relative to real-time (>1.0 is faster)
        audio_duration: Duration of audio processed (seconds)
        processing_time: Time taken to process (seconds)
    """
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    real_time_factor: float
    audio_duration: float
    processing_time: float


@dataclass
class PerformanceStatistics:
    """
    Aggregated performance statistics.
    
    Attributes:
        avg_cpu_percent: Average CPU usage
        max_cpu_percent: Maximum CPU usage
        avg_memory_mb: Average memory usage
        max_memory_mb: Maximum memory usage
        avg_real_time_factor: Average real-time factor
        min_real_time_factor: Minimum real-time factor
        total_audio_duration: Total audio processed (seconds)
        total_processing_time: Total processing time (seconds)
        sample_count: Number of samples collected
    """
    avg_cpu_percent: float = 0.0
    max_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0
    max_memory_mb: float = 0.0
    avg_real_time_factor: float = 0.0
    min_real_time_factor: float = float('inf')
    total_audio_duration: float = 0.0
    total_processing_time: float = 0.0
    sample_count: int = 0


class PerformanceMonitor:
    """
    Monitors performance metrics for speech analysis.
    
    Tracks CPU usage, memory usage, and real-time factor.
    Provides statistics and logging for performance optimization.
    """
    
    def __init__(
        self,
        cpu_limit_percent: float = 50.0,
        memory_limit_mb: float = 500.0,
        min_real_time_factor: float = 1.0,
        monitoring_interval: float = 1.0,
        history_size: int = 100
    ):
        """
        Initialize performance monitor.
        
        Args:
            cpu_limit_percent: CPU usage limit (0-100)
            memory_limit_mb: Memory usage limit in MB
            min_real_time_factor: Minimum acceptable real-time factor
            monitoring_interval: How often to collect metrics (seconds)
            history_size: Number of metrics to keep in history
        """
        self.cpu_limit_percent = cpu_limit_percent
        self.memory_limit_mb = memory_limit_mb
        self.min_real_time_factor = min_real_time_factor
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        
        # Metrics history
        self._metrics_history: deque = deque(maxlen=history_size)
        
        # Current process
        self._process = psutil.Process()
        
        # Monitoring thread
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Running state
        self._is_monitoring = False
        
        # Performance tracking
        self._audio_duration_total = 0.0
        self._processing_time_total = 0.0
        
        logger.info(
            f"Performance monitor initialized: "
            f"cpu_limit={cpu_limit_percent}%, "
            f"memory_limit={memory_limit_mb}MB, "
            f"min_rtf={min_real_time_factor}"
        )
    
    def start_monitoring(self) -> None:
        """Start performance monitoring in background thread."""
        if self._is_monitoring:
            return
        
        self._stop_event.clear()
        self._is_monitoring = True
        
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="PerformanceMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if not self._is_monitoring:
            return
        
        self._stop_event.set()
        self._is_monitoring = False
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Store in history
                with self._lock:
                    self._metrics_history.append(metrics)
                
                # Check limits and log warnings
                self._check_limits(metrics)
                
                # Wait for next interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # Get CPU usage (percentage for this process)
        cpu_percent = self._process.cpu_percent(interval=0.1)
        
        # Get memory usage (in MB)
        memory_info = self._process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # Convert bytes to MB
        
        # Calculate real-time factor
        with self._lock:
            if self._audio_duration_total > 0:
                real_time_factor = self._audio_duration_total / self._processing_time_total
            else:
                real_time_factor = 0.0
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            real_time_factor=real_time_factor,
            audio_duration=self._audio_duration_total,
            processing_time=self._processing_time_total
        )
    
    def _check_limits(self, metrics: PerformanceMetrics) -> None:
        """Check if metrics exceed limits and log warnings."""
        # Check CPU limit
        if metrics.cpu_percent > self.cpu_limit_percent:
            logger.warning(
                f"CPU usage ({metrics.cpu_percent:.1f}%) exceeds limit "
                f"({self.cpu_limit_percent}%)"
            )
        
        # Check memory limit
        if metrics.memory_mb > self.memory_limit_mb:
            logger.warning(
                f"Memory usage ({metrics.memory_mb:.1f}MB) exceeds limit "
                f"({self.memory_limit_mb}MB)"
            )
        
        # Check real-time factor
        if metrics.real_time_factor > 0 and metrics.real_time_factor < self.min_real_time_factor:
            logger.warning(
                f"Real-time factor ({metrics.real_time_factor:.2f}x) below minimum "
                f"({self.min_real_time_factor}x)"
            )
    
    def record_processing(self, audio_duration: float, processing_time: float) -> None:
        """
        Record a processing operation.
        
        Args:
            audio_duration: Duration of audio processed (seconds)
            processing_time: Time taken to process (seconds)
        """
        with self._lock:
            self._audio_duration_total += audio_duration
            self._processing_time_total += processing_time
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """
        Get the most recent performance metrics.
        
        Returns:
            Latest PerformanceMetrics or None if no data
        """
        with self._lock:
            if self._metrics_history:
                return self._metrics_history[-1]
            return None
    
    def get_statistics(self) -> PerformanceStatistics:
        """
        Get aggregated performance statistics.
        
        Returns:
            PerformanceStatistics with aggregated data
        """
        with self._lock:
            if not self._metrics_history:
                return PerformanceStatistics()
            
            # Calculate statistics
            cpu_values = [m.cpu_percent for m in self._metrics_history]
            memory_values = [m.memory_mb for m in self._metrics_history]
            rtf_values = [m.real_time_factor for m in self._metrics_history if m.real_time_factor > 0]
            
            stats = PerformanceStatistics(
                avg_cpu_percent=sum(cpu_values) / len(cpu_values) if cpu_values else 0.0,
                max_cpu_percent=max(cpu_values) if cpu_values else 0.0,
                avg_memory_mb=sum(memory_values) / len(memory_values) if memory_values else 0.0,
                max_memory_mb=max(memory_values) if memory_values else 0.0,
                avg_real_time_factor=sum(rtf_values) / len(rtf_values) if rtf_values else 0.0,
                min_real_time_factor=min(rtf_values) if rtf_values else 0.0,
                total_audio_duration=self._audio_duration_total,
                total_processing_time=self._processing_time_total,
                sample_count=len(self._metrics_history)
            )
            
            return stats
    
    def get_metrics_history(self) -> List[PerformanceMetrics]:
        """
        Get all metrics from history.
        
        Returns:
            List of PerformanceMetrics
        """
        with self._lock:
            return list(self._metrics_history)
    
    def reset_statistics(self) -> None:
        """Reset all statistics and history."""
        with self._lock:
            self._metrics_history.clear()
            self._audio_duration_total = 0.0
            self._processing_time_total = 0.0
        
        logger.info("Performance statistics reset")
    
    def log_statistics(self) -> None:
        """Log current performance statistics."""
        stats = self.get_statistics()
        
        logger.info(
            f"Performance Statistics:\n"
            f"  CPU: avg={stats.avg_cpu_percent:.1f}%, max={stats.max_cpu_percent:.1f}%\n"
            f"  Memory: avg={stats.avg_memory_mb:.1f}MB, max={stats.max_memory_mb:.1f}MB\n"
            f"  Real-time Factor: avg={stats.avg_real_time_factor:.2f}x, "
            f"min={stats.min_real_time_factor:.2f}x\n"
            f"  Total Audio: {stats.total_audio_duration:.1f}s\n"
            f"  Total Processing: {stats.total_processing_time:.1f}s\n"
            f"  Samples: {stats.sample_count}"
        )
    
    def get_cpu_usage(self) -> float:
        """
        Get current CPU usage percentage.
        
        Returns:
            CPU usage (0-100)
        """
        return self._process.cpu_percent(interval=0.1)
    
    def get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        memory_info = self._process.memory_info()
        return memory_info.rss / (1024 * 1024)
    
    def get_real_time_factor(self) -> float:
        """
        Get current real-time factor.
        
        Returns:
            Real-time factor (processing_speed / real_time)
        """
        with self._lock:
            if self._processing_time_total > 0:
                return self._audio_duration_total / self._processing_time_total
            return 0.0
    
    def is_within_limits(self) -> bool:
        """
        Check if current performance is within limits.
        
        Returns:
            True if all metrics are within limits
        """
        metrics = self.get_current_metrics()
        if not metrics:
            return True
        
        cpu_ok = metrics.cpu_percent <= self.cpu_limit_percent
        memory_ok = metrics.memory_mb <= self.memory_limit_mb
        rtf_ok = metrics.real_time_factor == 0.0 or metrics.real_time_factor >= self.min_real_time_factor
        
        return cpu_ok and memory_ok and rtf_ok
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive performance report.
        
        Returns:
            Dictionary with performance data
        """
        stats = self.get_statistics()
        current = self.get_current_metrics()
        
        report = {
            "current": {
                "cpu_percent": current.cpu_percent if current else 0.0,
                "memory_mb": current.memory_mb if current else 0.0,
                "real_time_factor": current.real_time_factor if current else 0.0,
            },
            "statistics": {
                "avg_cpu_percent": stats.avg_cpu_percent,
                "max_cpu_percent": stats.max_cpu_percent,
                "avg_memory_mb": stats.avg_memory_mb,
                "max_memory_mb": stats.max_memory_mb,
                "avg_real_time_factor": stats.avg_real_time_factor,
                "min_real_time_factor": stats.min_real_time_factor,
                "total_audio_duration": stats.total_audio_duration,
                "total_processing_time": stats.total_processing_time,
                "sample_count": stats.sample_count,
            },
            "limits": {
                "cpu_limit_percent": self.cpu_limit_percent,
                "memory_limit_mb": self.memory_limit_mb,
                "min_real_time_factor": self.min_real_time_factor,
            },
            "within_limits": self.is_within_limits(),
        }
        
        return report
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop_monitoring()
