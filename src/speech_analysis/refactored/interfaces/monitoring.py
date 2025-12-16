"""Performance monitoring interfaces."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from datetime import datetime

from ..models.performance import PerformanceMetrics, ResourceStatus


class IPerformanceMonitor(ABC):
    """Interface for performance monitoring."""
    
    @abstractmethod
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        pass
    
    @abstractmethod
    def stop_monitoring(self) -> Dict[str, Any]:
        """
        Stop monitoring and generate report.
        
        Returns:
            Dictionary containing monitoring summary.
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_metrics(self) -> PerformanceMetrics:
        """
        Get current performance metrics.
        
        Returns:
            Current performance metrics.
        """
        pass
    
    @abstractmethod
    def get_average_metrics(self) -> PerformanceMetrics:
        """
        Get average performance metrics over monitoring period.
        
        Returns:
            Average performance metrics.
        """
        pass
    
    @abstractmethod
    def check_resource_limits(self) -> ResourceStatus:
        """
        Check if resource limits are exceeded.
        
        Returns:
            Resource status (NORMAL, WARNING, CRITICAL).
        """
        pass
    
    @abstractmethod
    def get_rtf_history(self) -> List[float]:
        """
        Get history of RTF values.
        
        Returns:
            List of RTF values over time.
        """
        pass
    
    @abstractmethod
    def get_cpu_history(self) -> List[float]:
        """
        Get history of CPU usage values.
        
        Returns:
            List of CPU usage percentages over time.
        """
        pass
    
    @abstractmethod
    def get_memory_history(self) -> List[float]:
        """
        Get history of memory usage values.
        
        Returns:
            List of memory usage in MB over time.
        """
        pass
    
    @abstractmethod
    def should_adjust_performance(self) -> bool:
        """
        Check if performance adjustment is needed.
        
        Returns:
            True if adjustment needed, False otherwise.
        """
        pass
    
    @abstractmethod
    def reset_metrics(self) -> None:
        """Reset all metrics and history."""
        pass
