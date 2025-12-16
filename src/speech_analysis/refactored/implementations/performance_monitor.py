"""Performance monitoring implementation."""

import time
import psutil
import threading
from collections import deque
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..interfaces.monitoring import IPerformanceMonitor
from ..models.performance import PerformanceMetrics, ResourceStatus


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring."""
    
    max_rtf: float = 1.5  # Maximum acceptable RTF
    max_cpu_percent: float = 80.0  # Maximum CPU usage
    max_memory_mb: float = 2000.0  # Maximum memory usage
    history_size: int = 100  # Number of samples to keep in history
    warning_threshold: float = 0.8  # Threshold for warning (80% of limit)
    critical_threshold: float = 0.95  # Threshold for critical (95% of limit)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_rtf <= 0:
            raise ValueError("max_rtf must be positive")
        if not 0 < self.max_cpu_percent <= 100:
            raise ValueError("max_cpu_percent must be between 0 and 100")
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if self.history_size <= 0:
            raise ValueError("history_size must be positive")


class PerformanceMonitor(IPerformanceMonitor):
    """
    Performance monitor implementation.
    
    Tracks RTF, CPU usage, memory usage, latency, and throughput.
    Provides resource limit checking and adaptive performance adjustment.
    """
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        """
        Initialize performance monitor.
        
        Args:
            config: Performance monitoring configuration.
        """
        self.config = config or PerformanceConfig()
        
        # Monitoring state
        self._monitoring = False
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()
        
        # Process handle for resource monitoring
        self._process = psutil.Process()
        
        # Initialize CPU percent (first call needs to be made to start tracking)
        try:
            self._process.cpu_percent(interval=0)
        except Exception:
            pass
        
        # Metrics history (using deque for efficient FIFO)
        self._rtf_history: deque = deque(maxlen=self.config.history_size)
        self._cpu_history: deque = deque(maxlen=self.config.history_size)
        self._memory_history: deque = deque(maxlen=self.config.history_size)
        self._latency_history: deque = deque(maxlen=self.config.history_size)
        
        # Cumulative metrics
        self._total_audio_duration = 0.0
        self._total_processing_time = 0.0
        self._processing_count = 0
        
        # Current metrics
        self._current_metrics: Optional[PerformanceMetrics] = None
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        with self._lock:
            if self._monitoring:
                return
            
            self._monitoring = True
            self._start_time = time.time()
            
            # Reset metrics
            self._rtf_history.clear()
            self._cpu_history.clear()
            self._memory_history.clear()
            self._latency_history.clear()
            
            self._total_audio_duration = 0.0
            self._total_processing_time = 0.0
            self._processing_count = 0
            
            self._current_metrics = None
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """
        Stop monitoring and generate report.
        
        Returns:
            Dictionary containing monitoring summary.
        """
        with self._lock:
            if not self._monitoring:
                return {}
            
            self._monitoring = False
            elapsed_time = time.time() - self._start_time if self._start_time else 0
            
            # Generate summary report
            report = {
                "monitoring_duration_seconds": elapsed_time,
                "total_audio_processed_seconds": self._total_audio_duration,
                "total_processing_time_seconds": self._total_processing_time,
                "processing_count": self._processing_count,
                "average_rtf": self._calculate_average(self._rtf_history),
                "average_cpu_percent": self._calculate_average(self._cpu_history),
                "average_memory_mb": self._calculate_average(self._memory_history),
                "average_latency_ms": self._calculate_average(self._latency_history),
                "max_rtf": max(self._rtf_history) if self._rtf_history else 0.0,
                "max_cpu_percent": max(self._cpu_history) if self._cpu_history else 0.0,
                "max_memory_mb": max(self._memory_history) if self._memory_history else 0.0,
                "max_latency_ms": max(self._latency_history) if self._latency_history else 0.0,
                "resource_status": self.check_resource_limits().value,
            }
            
            return report
    
    def record_processing(
        self,
        audio_duration: float,
        processing_time: float,
    ) -> None:
        """
        Record processing metrics.
        
        Args:
            audio_duration: Duration of audio processed in seconds.
            processing_time: Time taken to process in seconds.
        """
        if audio_duration <= 0:
            raise ValueError("audio_duration must be positive")
        if processing_time < 0:
            raise ValueError("processing_time must be non-negative")
        
        with self._lock:
            if not self._monitoring:
                return
            
            # Calculate RTF
            rtf = processing_time / audio_duration
            
            # Get current resource usage
            cpu_percent = self._get_cpu_usage()
            memory_mb = self._get_memory_usage()
            
            # Calculate latency (processing time in ms)
            latency_ms = processing_time * 1000
            
            # Calculate throughput ratio
            elapsed_time = time.time() - self._start_time if self._start_time else 1.0
            throughput_ratio = self._total_audio_duration / elapsed_time if elapsed_time > 0 else 0.0
            
            # Update history
            self._rtf_history.append(rtf)
            self._cpu_history.append(cpu_percent)
            self._memory_history.append(memory_mb)
            self._latency_history.append(latency_ms)
            
            # Update cumulative metrics
            self._total_audio_duration += audio_duration
            self._total_processing_time += processing_time
            self._processing_count += 1
            
            # Update current metrics
            self._current_metrics = PerformanceMetrics(
                rtf=rtf,
                cpu_usage=cpu_percent,
                memory_usage_mb=memory_mb,
                latency_ms=latency_ms,
                throughput_ratio=throughput_ratio,
                timestamp=datetime.now(),
            )
    
    def get_metrics(self) -> PerformanceMetrics:
        """
        Get current performance metrics.
        
        Returns:
            Current performance metrics.
        """
        with self._lock:
            if self._current_metrics is None:
                # Return default metrics if no processing recorded yet
                return PerformanceMetrics(
                    rtf=0.0,
                    cpu_usage=self._get_cpu_usage(),
                    memory_usage_mb=self._get_memory_usage(),
                    latency_ms=0.0,
                    throughput_ratio=0.0,
                    timestamp=datetime.now(),
                )
            
            return self._current_metrics
    
    def get_average_metrics(self) -> PerformanceMetrics:
        """
        Get average performance metrics over monitoring period.
        
        Returns:
            Average performance metrics.
        """
        with self._lock:
            avg_rtf = self._calculate_average(self._rtf_history)
            avg_cpu = self._calculate_average(self._cpu_history)
            avg_memory = self._calculate_average(self._memory_history)
            avg_latency = self._calculate_average(self._latency_history)
            
            # Calculate average throughput
            elapsed_time = time.time() - self._start_time if self._start_time else 1.0
            avg_throughput = self._total_audio_duration / elapsed_time if elapsed_time > 0 else 0.0
            
            return PerformanceMetrics(
                rtf=avg_rtf,
                cpu_usage=avg_cpu,
                memory_usage_mb=avg_memory,
                latency_ms=avg_latency,
                throughput_ratio=avg_throughput,
                timestamp=datetime.now(),
            )
    
    def check_resource_limits(self) -> ResourceStatus:
        """
        Check if resource limits are exceeded.
        
        Returns:
            Resource status (NORMAL, WARNING, CRITICAL).
        """
        with self._lock:
            if not self._rtf_history or not self._cpu_history or not self._memory_history:
                return ResourceStatus.UNKNOWN
            
            # Get recent averages (last 10 samples)
            recent_size = min(10, len(self._rtf_history))
            recent_rtf = sum(list(self._rtf_history)[-recent_size:]) / recent_size
            recent_cpu = sum(list(self._cpu_history)[-recent_size:]) / recent_size
            recent_memory = sum(list(self._memory_history)[-recent_size:]) / recent_size
            
            # Check against limits
            rtf_ratio = recent_rtf / self.config.max_rtf
            cpu_ratio = recent_cpu / self.config.max_cpu_percent
            memory_ratio = recent_memory / self.config.max_memory_mb
            
            # Find worst ratio
            max_ratio = max(rtf_ratio, cpu_ratio, memory_ratio)
            
            if max_ratio >= self.config.critical_threshold:
                return ResourceStatus.CRITICAL
            elif max_ratio >= self.config.warning_threshold:
                return ResourceStatus.WARNING
            else:
                return ResourceStatus.NORMAL
    
    def get_rtf_history(self) -> List[float]:
        """
        Get history of RTF values.
        
        Returns:
            List of RTF values over time.
        """
        with self._lock:
            return list(self._rtf_history)
    
    def get_cpu_history(self) -> List[float]:
        """
        Get history of CPU usage values.
        
        Returns:
            List of CPU usage percentages over time.
        """
        with self._lock:
            return list(self._cpu_history)
    
    def get_memory_history(self) -> List[float]:
        """
        Get history of memory usage values.
        
        Returns:
            List of memory usage in MB over time.
        """
        with self._lock:
            return list(self._memory_history)
    
    def should_adjust_performance(self) -> bool:
        """
        Check if performance adjustment is needed.
        
        Returns:
            True if adjustment needed, False otherwise.
        """
        status = self.check_resource_limits()
        return status in (ResourceStatus.WARNING, ResourceStatus.CRITICAL)
    
    def reset_metrics(self) -> None:
        """Reset all metrics and history."""
        with self._lock:
            self._rtf_history.clear()
            self._cpu_history.clear()
            self._memory_history.clear()
            self._latency_history.clear()
            
            self._total_audio_duration = 0.0
            self._total_processing_time = 0.0
            self._processing_count = 0
            
            self._current_metrics = None
            
            if self._monitoring:
                self._start_time = time.time()
    
    def _get_cpu_usage(self) -> float:
        """
        Get current CPU usage percentage.
        
        Returns:
            CPU usage percentage (0-100).
        """
        try:
            # Get CPU percent for this process
            # interval=None uses cached value (non-blocking)
            # We initialized cpu_percent in __init__ so this is safe
            cpu = self._process.cpu_percent(interval=None)
            # If cpu is 0.0, it might be the first call, try with minimal interval
            if cpu == 0.0:
                cpu = self._process.cpu_percent(interval=0)
            return cpu
        except Exception:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in MB.
        """
        try:
            # Get memory info for this process
            memory_info = self._process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert bytes to MB
        except Exception:
            return 0.0
    
    @staticmethod
    def _calculate_average(values: deque) -> float:
        """
        Calculate average of values in deque.
        
        Args:
            values: Deque of numeric values.
        
        Returns:
            Average value, or 0.0 if empty.
        """
        if not values:
            return 0.0
        return sum(values) / len(values)
