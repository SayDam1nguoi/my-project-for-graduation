"""Performance monitoring data models."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ResourceStatus(Enum):
    """Resource usage status."""
    
    NORMAL = "normal"  # Within limits
    WARNING = "warning"  # Approaching limits
    CRITICAL = "critical"  # Exceeding limits
    UNKNOWN = "unknown"  # Status unknown


@dataclass
class PerformanceMetrics:
    """Performance metrics for STT processing."""
    
    rtf: float  # Real-time factor (processing_time / audio_duration)
    cpu_usage: float  # CPU usage percentage (0-100)
    memory_usage_mb: float  # Memory usage in MB
    latency_ms: float  # End-to-end latency in milliseconds
    throughput_ratio: float  # Audio processed / time elapsed
    timestamp: datetime
    
    @property
    def is_realtime(self) -> bool:
        """Check if processing is real-time (RTF < 1.0)."""
        return self.rtf < 1.0
    
    @property
    def cpu_status(self) -> ResourceStatus:
        """Get CPU usage status."""
        if self.cpu_usage < 60:
            return ResourceStatus.NORMAL
        elif self.cpu_usage < 80:
            return ResourceStatus.WARNING
        else:
            return ResourceStatus.CRITICAL
    
    @property
    def memory_status(self) -> ResourceStatus:
        """Get memory usage status (assuming 2GB limit)."""
        if self.memory_usage_mb < 1000:
            return ResourceStatus.NORMAL
        elif self.memory_usage_mb < 1500:
            return ResourceStatus.WARNING
        else:
            return ResourceStatus.CRITICAL
    
    def __post_init__(self):
        """Validate performance metrics data."""
        if self.rtf < 0:
            raise ValueError("RTF must be non-negative")
        if not 0.0 <= self.cpu_usage <= 100.0:
            raise ValueError("CPU usage must be between 0 and 100")
        if self.memory_usage_mb < 0:
            raise ValueError("Memory usage must be non-negative")
        if self.latency_ms < 0:
            raise ValueError("Latency must be non-negative")
