"""Configuration data models."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class EngineType(Enum):
    """STT Engine types."""
    
    WHISPER = "whisper"
    FASTER_WHISPER = "faster-whisper"
    VOSK = "vosk"
    GOOGLE = "google"


class DeviceType(Enum):
    """Processing device types."""
    
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


@dataclass
class AudioConfig:
    """Audio capture and processing configuration."""
    
    sample_rate: int = 16000
    chunk_duration: float = 3.0
    overlap_duration: float = 0.5
    channels: int = 1
    device_index: Optional[int] = None
    buffer_size_seconds: float = 30.0
    
    def __post_init__(self):
        """Validate audio configuration."""
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.chunk_duration <= 0:
            raise ValueError("Chunk duration must be positive")
        if self.overlap_duration < 0:
            raise ValueError("Overlap duration must be non-negative")
        if self.channels not in [1, 2]:
            raise ValueError("Channels must be 1 (mono) or 2 (stereo)")


@dataclass
class EngineConfig:
    """STT Engine configuration."""
    
    engine_type: EngineType
    model_size: str = "base"
    device: DeviceType = DeviceType.CPU
    language: str = "vi"
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    compute_type: str = "int8"
    num_workers: int = 1
    download_root: Optional[str] = None
    model_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate engine configuration."""
        if self.beam_size <= 0:
            raise ValueError("Beam size must be positive")
        if self.best_of <= 0:
            raise ValueError("Best of must be positive")
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("Temperature must be between 0 and 1")


@dataclass
class ProcessingConfig:
    """Audio processing pipeline configuration."""
    
    enable_vad: bool = True
    enable_preprocessing: bool = True
    enable_hallucination_filter: bool = True
    enable_post_processing: bool = True
    vad_method: str = "silero"  # silero, webrtc, energy
    noise_reduction: bool = True
    normalization: bool = True
    pre_emphasis: bool = True
    
    # Hallucination filter settings
    min_confidence: float = 0.5
    no_speech_threshold: float = 0.6
    compression_ratio_threshold: float = 2.4
    early_segment_threshold_seconds: float = 2.0
    
    def __post_init__(self):
        """Validate processing configuration."""
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("Min confidence must be between 0 and 1")
        if not 0.0 <= self.no_speech_threshold <= 1.0:
            raise ValueError("No speech threshold must be between 0 and 1")


@dataclass
class PerformanceConfig:
    """Performance and resource management configuration."""
    
    max_memory_mb: int = 2000
    cpu_limit_percent: float = 80.0
    target_rtf: float = 1.5
    enable_adaptive_adjustment: bool = True
    monitoring_interval_seconds: float = 5.0
    
    def __post_init__(self):
        """Validate performance configuration."""
        if self.max_memory_mb <= 0:
            raise ValueError("Max memory must be positive")
        if not 0.0 <= self.cpu_limit_percent <= 100.0:
            raise ValueError("CPU limit must be between 0 and 100")
        if self.target_rtf <= 0:
            raise ValueError("Target RTF must be positive")


@dataclass
class QualityConfig:
    """Quality analysis configuration."""
    
    min_clarity_score: float = 0.5
    min_fluency_score: float = 0.5
    min_snr_db: float = 10.0
    max_hallucination_risk: float = 0.3
    
    def __post_init__(self):
        """Validate quality configuration."""
        if not 0.0 <= self.min_clarity_score <= 1.0:
            raise ValueError("Min clarity score must be between 0 and 1")
        if not 0.0 <= self.min_fluency_score <= 1.0:
            raise ValueError("Min fluency score must be between 0 and 1")


@dataclass
class STTConfig:
    """Complete STT system configuration."""
    
    audio: AudioConfig = field(default_factory=AudioConfig)
    engine: EngineConfig = field(default_factory=lambda: EngineConfig(
        engine_type=EngineType.FASTER_WHISPER
    ))
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    
    # Additional settings
    log_level: str = "INFO"
    enable_metrics: bool = True
    output_format: str = "text"  # text, json, srt
    custom_vocabulary: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create_profile(cls, profile_name: str) -> "STTConfig":
        """Create configuration from predefined profile."""
        profiles = {
            "accuracy": cls(
                engine=EngineConfig(
                    engine_type=EngineType.WHISPER,
                    model_size="large-v3",
                    beam_size=10,
                    best_of=10,
                ),
                performance=PerformanceConfig(target_rtf=3.0),
            ),
            "balanced": cls(
                engine=EngineConfig(
                    engine_type=EngineType.FASTER_WHISPER,
                    model_size="base",
                ),
            ),
            "speed": cls(
                engine=EngineConfig(
                    engine_type=EngineType.FASTER_WHISPER,
                    model_size="tiny",
                    beam_size=3,
                ),
                processing=ProcessingConfig(
                    enable_preprocessing=False,
                    enable_post_processing=False,
                ),
                performance=PerformanceConfig(target_rtf=1.0),
            ),
            "cpu-optimized": cls(
                engine=EngineConfig(
                    engine_type=EngineType.FASTER_WHISPER,
                    model_size="base",
                    device=DeviceType.CPU,
                    compute_type="int8",
                ),
                performance=PerformanceConfig(
                    max_memory_mb=1000,
                    cpu_limit_percent=70.0,
                ),
            ),
        }
        
        if profile_name not in profiles:
            raise ValueError(
                f"Unknown profile: {profile_name}. "
                f"Available profiles: {list(profiles.keys())}"
            )
        
        return profiles[profile_name]
