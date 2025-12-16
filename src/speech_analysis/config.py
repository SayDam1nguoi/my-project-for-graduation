"""
Configuration loader for speech analysis.

This module loads and validates configuration from speech_config.yaml.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class AudioConfig:
    """Audio capture configuration."""
    sample_rate: int = 16000
    channels: int = 1
    format: str = "paInt16"
    chunk_size: int = 1024
    buffer_duration: int = 30
    device_index: Optional[int] = None
    
    def validate(self) -> None:
        """Validate audio configuration."""
        if self.sample_rate <= 0:
            raise ValueError(f"Invalid sample_rate: {self.sample_rate}")
        if self.channels not in [1, 2]:
            raise ValueError(f"Invalid channels: {self.channels}")
        if self.chunk_size <= 0:
            raise ValueError(f"Invalid chunk_size: {self.chunk_size}")
        if self.buffer_duration <= 0:
            raise ValueError(f"Invalid buffer_duration: {self.buffer_duration}")


@dataclass
class STTConfig:
    """Speech-to-Text configuration."""
    vosk_model_path: str = "models/vosk-model-vn-0.4"
    language: str = "vi"
    sample_rate: int = 16000
    chunk_duration: float = 2.0
    overlap_duration: float = 0.5
    max_latency: float = 2.0
    
    def validate(self) -> None:
        """Validate STT configuration."""
        if not self.vosk_model_path:
            raise ValueError("vosk_model_path cannot be empty")
        
        if self.sample_rate <= 0:
            raise ValueError(f"Invalid sample_rate: {self.sample_rate}")
        
        if self.chunk_duration <= 0:
            raise ValueError(f"Invalid chunk_duration: {self.chunk_duration}")
        
        if self.overlap_duration < 0:
            raise ValueError(f"Invalid overlap_duration: {self.overlap_duration}")
        
        if self.max_latency <= 0:
            raise ValueError(f"Invalid max_latency: {self.max_latency}")


@dataclass
class GoogleSTTConfig(STTConfig):
    """Google Cloud Speech-to-Text configuration."""
    
    # Google STT specific
    credentials_path: Optional[str] = None  # Path to Google Cloud credentials JSON
    use_enhanced: bool = True  # Use enhanced model (better quality, higher cost)
    enable_word_time_offsets: bool = True  # Get word-level timestamps
    enable_automatic_punctuation: bool = True  # Auto add punctuation
    model: str = "default"  # "default", "phone_call", "video", "command_and_search"
    use_streaming: bool = False  # Use streaming API for real-time
    
    def validate(self) -> None:
        """Validate Google STT configuration."""
        super().validate()
        
        # Validate model
        valid_models = ["default", "phone_call", "video", "command_and_search"]
        if self.model not in valid_models:
            raise ValueError(f"Invalid model: {self.model}. Must be one of {valid_models}")


@dataclass
class EnhancedSTTConfig(STTConfig):
    """Enhanced STT configuration extending base STT config.
    
    This configuration supports both Whisper and VOSK quantized models
    with audio cleaning, VAD, custom vocabulary, and overlap handling.
    """
    
    # Model selection (Whisper is default for better accuracy)
    model_type: str = "whisper"  # "whisper" (default) or "vosk" (fallback)
    model_size: str = "base"  # tiny, base (recommended), small, medium, large
    model_path: Optional[str] = None
    
    # Quantization (for CPU optimization)
    compute_type: str = "int8"  # int8, int16, float16, float32
    device: str = "cpu"
    
    # Audio processing
    enable_audio_cleaning: bool = True
    enable_vad: bool = True
    vad_method: str = "silero"  # silero, webrtc, energy
    
    # Overlap handling
    overlap_duration: float = 0.5
    max_buffer_size: int = 10
    
    # Custom vocabulary
    vocabulary_file: Optional[str] = None
    enable_vocabulary: bool = False
    
    # Performance
    num_threads: int = 4
    max_memory_mb: int = 500
    cpu_limit_percent: float = 50.0
    min_real_time_factor: float = 1.0
    
    # Backward compatibility
    fallback_to_vosk: bool = True
    
    def validate(self) -> None:
        """Validate Enhanced STT configuration."""
        # Validate base config
        super().validate()
        
        # Validate model type
        valid_model_types = ["whisper", "vosk"]
        if self.model_type not in valid_model_types:
            raise ValueError(f"Invalid model_type: {self.model_type}. Must be one of {valid_model_types}")
        
        # Validate model size (for Whisper)
        if self.model_type == "whisper":
            valid_sizes = ["tiny", "base", "small", "medium", "large"]
            if self.model_size not in valid_sizes:
                raise ValueError(f"Invalid model_size: {self.model_size}. Must be one of {valid_sizes}")
        
        # Validate compute type
        valid_compute_types = ["int8", "int16", "float16", "float32"]
        if self.compute_type not in valid_compute_types:
            raise ValueError(f"Invalid compute_type: {self.compute_type}. Must be one of {valid_compute_types}")
        
        # Validate device
        if self.device not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid device: {self.device}. Must be 'cpu' or 'cuda'")
        
        # Validate VAD method
        valid_vad_methods = ["silero", "webrtc", "energy"]
        if self.vad_method not in valid_vad_methods:
            raise ValueError(f"Invalid vad_method: {self.vad_method}. Must be one of {valid_vad_methods}")
        
        # Validate numeric values
        if self.num_threads <= 0:
            raise ValueError(f"Invalid num_threads: {self.num_threads}")
        
        if self.overlap_duration < 0:
            raise ValueError(f"Invalid overlap_duration: {self.overlap_duration}")
        
        if self.max_buffer_size <= 0:
            raise ValueError(f"Invalid max_buffer_size: {self.max_buffer_size}")
        
        if self.max_memory_mb <= 0:
            raise ValueError(f"Invalid max_memory_mb: {self.max_memory_mb}")
        
        if self.cpu_limit_percent <= 0 or self.cpu_limit_percent > 100:
            raise ValueError(f"Invalid cpu_limit_percent: {self.cpu_limit_percent}")
        
        if self.min_real_time_factor <= 0:
            raise ValueError(f"Invalid min_real_time_factor: {self.min_real_time_factor}")
        
        # Validate vocabulary file if enabled
        if self.enable_vocabulary and not self.vocabulary_file:
            raise ValueError("vocabulary_file must be specified when enable_vocabulary is True")
    
    def to_legacy_config(self) -> STTConfig:
        """Convert to legacy STTConfig format for backward compatibility."""
        return STTConfig(
            vosk_model_path=self.vosk_model_path,
            language=self.language,
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
            overlap_duration=self.overlap_duration,
            max_latency=self.max_latency
        )


@dataclass
class WhisperSTTConfig(STTConfig):
    """Whisper STT configuration extending base STT config.
    
    DEPRECATED: Use EnhancedSTTConfig instead. This class is kept for backward compatibility.
    """
    
    # Whisper mode
    use_whisper: bool = False
    
    # Model selection
    model_size: str = "base"  # tiny, base, small, medium, large
    model_path: Optional[str] = None
    
    # Quantization (for CPU optimization)
    compute_type: str = "int8"  # int8, int16, float16, float32
    device: str = "cpu"
    num_threads: int = 4
    
    # Language settings
    language: str = "vi"  # Vietnamese
    task: str = "transcribe"  # transcribe or translate
    
    # Audio processing
    enable_audio_cleaning: bool = True
    enable_vad: bool = True
    vad_method: str = "silero"  # silero, webrtc, energy
    
    # Overlap handling
    overlap_buffer_duration: float = 0.5
    max_buffer_size: int = 10
    
    # Custom vocabulary
    vocabulary_file: Optional[str] = None
    enable_vocabulary: bool = False
    
    # Performance
    max_memory_mb: int = 500
    beam_size: int = 5  # Beam search size for decoding
    best_of: int = 5  # Number of candidates to consider
    
    # Output options
    word_timestamps: bool = True  # Extract word-level timestamps
    condition_on_previous_text: bool = True  # Use context from previous segments
    
    def validate(self) -> None:
        """Validate Whisper STT configuration."""
        # Validate base config
        super().validate()
        
        # Validate model size
        valid_sizes = ["tiny", "base", "small", "medium", "large"]
        if self.model_size not in valid_sizes:
            raise ValueError(f"Invalid model_size: {self.model_size}. Must be one of {valid_sizes}")
        
        # Validate compute type
        valid_compute_types = ["int8", "int16", "float16", "float32"]
        if self.compute_type not in valid_compute_types:
            raise ValueError(f"Invalid compute_type: {self.compute_type}. Must be one of {valid_compute_types}")
        
        # Validate device
        if self.device not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid device: {self.device}. Must be 'cpu' or 'cuda'")
        
        # Validate task
        if self.task not in ["transcribe", "translate"]:
            raise ValueError(f"Invalid task: {self.task}. Must be 'transcribe' or 'translate'")
        
        # Validate VAD method
        valid_vad_methods = ["silero", "webrtc", "energy"]
        if self.vad_method not in valid_vad_methods:
            raise ValueError(f"Invalid vad_method: {self.vad_method}. Must be one of {valid_vad_methods}")
        
        # Validate numeric values
        if self.num_threads <= 0:
            raise ValueError(f"Invalid num_threads: {self.num_threads}")
        
        if self.overlap_buffer_duration < 0:
            raise ValueError(f"Invalid overlap_buffer_duration: {self.overlap_buffer_duration}")
        
        if self.max_buffer_size <= 0:
            raise ValueError(f"Invalid max_buffer_size: {self.max_buffer_size}")
        
        if self.max_memory_mb <= 0:
            raise ValueError(f"Invalid max_memory_mb: {self.max_memory_mb}")
        
        if self.beam_size <= 0:
            raise ValueError(f"Invalid beam_size: {self.beam_size}")
        
        if self.best_of <= 0:
            raise ValueError(f"Invalid best_of: {self.best_of}")
        
        # Validate vocabulary file if enabled
        if self.enable_vocabulary and not self.vocabulary_file:
            raise ValueError("vocabulary_file must be specified when enable_vocabulary is True")
    
    def to_legacy_config(self) -> STTConfig:
        """Convert to legacy STTConfig format for backward compatibility."""
        return STTConfig(
            vosk_model_path=self.vosk_model_path,
            language=self.language,
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
            overlap_duration=self.overlap_duration,
            max_latency=self.max_latency
        )


@dataclass
class VideoTranscriptionConfig:
    """Configuration for video transcription."""
    
    # Whisper settings
    whisper_model_size: str = "large-v3"  # tiny, base, small, medium, large, large-v3
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    
    # Processing settings
    chunk_duration: float = 30.0  # seconds
    overlap_duration: float = 1.0  # seconds
    max_parallel_chunks: int = 2
    
    # Language detection
    auto_detect_language: bool = True
    default_language: str = "vi"  # Vietnamese
    language_detection_duration: float = 30.0
    
    # Speaker diarization
    enable_diarization: bool = False
    min_speakers: int = 1
    max_speakers: int = 10
    
    # Subtitle generation
    subtitle_max_chars_per_line: int = 42
    subtitle_max_lines: int = 2
    subtitle_min_duration: float = 1.0
    subtitle_max_duration: float = 7.0
    
    # Cache settings
    enable_cache: bool = True
    cache_dir: str = ".cache/transcriptions"
    cache_max_size_mb: int = 1000
    
    # Vietnamese optimization
    enable_vietnamese_optimization: bool = True
    vietnamese_vocabulary_file: Optional[str] = None
    vietnamese_diacritic_validation: bool = True
    
    # Performance
    use_audio_enhancement: bool = True
    use_vad: bool = True
    
    def validate(self) -> None:
        """Validate video transcription configuration."""
        # Validate model size
        valid_sizes = ["tiny", "base", "small", "medium", "large", "large-v3"]
        if self.whisper_model_size not in valid_sizes:
            raise ValueError(f"Invalid whisper_model_size: {self.whisper_model_size}. Must be one of {valid_sizes}")
        
        # Validate device
        if self.whisper_device not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid whisper_device: {self.whisper_device}. Must be 'cpu' or 'cuda'")
        
        # Validate compute type
        valid_compute_types = ["int8", "int16", "float16", "float32"]
        if self.whisper_compute_type not in valid_compute_types:
            raise ValueError(f"Invalid whisper_compute_type: {self.whisper_compute_type}. Must be one of {valid_compute_types}")
        
        # Validate numeric values
        if self.chunk_duration <= 0:
            raise ValueError(f"Invalid chunk_duration: {self.chunk_duration}")
        
        if self.overlap_duration < 0:
            raise ValueError(f"Invalid overlap_duration: {self.overlap_duration}")
        
        if self.overlap_duration >= self.chunk_duration:
            raise ValueError("overlap_duration must be less than chunk_duration")
        
        if self.max_parallel_chunks <= 0:
            raise ValueError(f"Invalid max_parallel_chunks: {self.max_parallel_chunks}")
        
        if self.language_detection_duration <= 0:
            raise ValueError(f"Invalid language_detection_duration: {self.language_detection_duration}")
        
        if self.min_speakers <= 0:
            raise ValueError(f"Invalid min_speakers: {self.min_speakers}")
        
        if self.max_speakers < self.min_speakers:
            raise ValueError("max_speakers must be >= min_speakers")
        
        if self.subtitle_max_chars_per_line <= 0:
            raise ValueError(f"Invalid subtitle_max_chars_per_line: {self.subtitle_max_chars_per_line}")
        
        if self.subtitle_max_lines <= 0:
            raise ValueError(f"Invalid subtitle_max_lines: {self.subtitle_max_lines}")
        
        if self.subtitle_min_duration <= 0:
            raise ValueError(f"Invalid subtitle_min_duration: {self.subtitle_min_duration}")
        
        if self.subtitle_max_duration <= self.subtitle_min_duration:
            raise ValueError("subtitle_max_duration must be > subtitle_min_duration")
        
        if self.cache_max_size_mb <= 0:
            raise ValueError(f"Invalid cache_max_size_mb: {self.cache_max_size_mb}")


@dataclass
class QualityConfig:
    """Quality analysis configuration."""
    clarity_threshold: float = 60.0
    snr_weight: float = 0.4
    spectral_weight: float = 0.4
    zcr_weight: float = 0.2
    fluency_threshold: float = 60.0
    speech_rate_weight: float = 0.3
    pause_weight: float = 0.4
    rhythm_weight: float = 0.3
    optimal_speech_rate_min: float = 3.0
    optimal_speech_rate_max: float = 5.0
    pause_threshold: float = 0.3
    long_pause_threshold: float = 1.0
    update_interval: float = 2.0
    
    def validate(self) -> None:
        """Validate quality configuration."""
        if not 0.0 <= self.clarity_threshold <= 100.0:
            raise ValueError(f"Invalid clarity_threshold: {self.clarity_threshold}")
        
        if not 0.0 <= self.fluency_threshold <= 100.0:
            raise ValueError(f"Invalid fluency_threshold: {self.fluency_threshold}")
        
        # Validate weights sum to 1.0
        clarity_weights_sum = self.snr_weight + self.spectral_weight + self.zcr_weight
        if not 0.99 <= clarity_weights_sum <= 1.01:  # Allow small floating point error
            raise ValueError(f"Clarity weights must sum to 1.0, got {clarity_weights_sum}")
        
        fluency_weights_sum = self.speech_rate_weight + self.pause_weight + self.rhythm_weight
        if not 0.99 <= fluency_weights_sum <= 1.01:
            raise ValueError(f"Fluency weights must sum to 1.0, got {fluency_weights_sum}")
        
        if self.optimal_speech_rate_min >= self.optimal_speech_rate_max:
            raise ValueError("optimal_speech_rate_min must be less than optimal_speech_rate_max")
        
        if self.pause_threshold <= 0:
            raise ValueError(f"Invalid pause_threshold: {self.pause_threshold}")
        
        if self.long_pause_threshold <= self.pause_threshold:
            raise ValueError("long_pause_threshold must be greater than pause_threshold")
        
        if self.update_interval <= 0:
            raise ValueError(f"Invalid update_interval: {self.update_interval}")


@dataclass
class StorageConfig:
    """Text storage configuration."""
    output_dir: str = "transcripts"
    filename_format: str = "transcript_%Y%m%d_%H%M%S.txt"
    encoding: str = "utf-8"
    auto_save_interval: int = 30
    create_backup: bool = True
    include_metadata: bool = True
    include_statistics: bool = True
    include_recommendations: bool = True
    
    def validate(self) -> None:
        """Validate storage configuration."""
        if not self.output_dir:
            raise ValueError("output_dir cannot be empty")
        
        if not self.filename_format:
            raise ValueError("filename_format cannot be empty")
        
        if self.encoding not in ["utf-8", "utf-16", "ascii"]:
            raise ValueError(f"Invalid encoding: {self.encoding}")
        
        if self.auto_save_interval <= 0:
            raise ValueError(f"Invalid auto_save_interval: {self.auto_save_interval}")


@dataclass
class PerformanceConfig:
    """Performance configuration."""
    max_memory_mb: int = 100
    audio_thread_priority: str = "high"
    processing_thread_priority: str = "normal"
    gui_update_rate: int = 10
    
    def validate(self) -> None:
        """Validate performance configuration."""
        if self.max_memory_mb <= 0:
            raise ValueError(f"Invalid max_memory_mb: {self.max_memory_mb}")
        
        valid_priorities = ["low", "normal", "high"]
        if self.audio_thread_priority not in valid_priorities:
            raise ValueError(f"Invalid audio_thread_priority: {self.audio_thread_priority}")
        
        if self.processing_thread_priority not in valid_priorities:
            raise ValueError(f"Invalid processing_thread_priority: {self.processing_thread_priority}")
        
        if self.gui_update_rate <= 0:
            raise ValueError(f"Invalid gui_update_rate: {self.gui_update_rate}")


@dataclass
class LoggingConfig:
    """Logging configuration."""
    enabled: bool = True
    log_file: str = "logs/speech_analysis.log"
    log_level: str = "INFO"
    max_log_size_mb: int = 10
    backup_count: int = 3
    
    def validate(self) -> None:
        """Validate logging configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            raise ValueError(f"Invalid log_level: {self.log_level}. Must be one of {valid_levels}")
        
        if self.max_log_size_mb <= 0:
            raise ValueError(f"Invalid max_log_size_mb: {self.max_log_size_mb}")
        
        if self.backup_count < 0:
            raise ValueError(f"Invalid backup_count: {self.backup_count}")


@dataclass
class SpeechAnalysisConfig:
    """Complete speech analysis configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    speech_to_text: STTConfig = field(default_factory=STTConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def validate(self) -> None:
        """Validate all configuration sections."""
        self.audio.validate()
        self.speech_to_text.validate()
        self.quality.validate()
        self.storage.validate()
        self.performance.validate()
        self.logging.validate()


def load_config(config_path: Optional[str] = None) -> SpeechAnalysisConfig:
    """
    Load speech analysis configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default path.
        
    Returns:
        SpeechAnalysisConfig object with loaded configuration.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config validation fails.
        yaml.YAMLError: If YAML parsing fails.
    """
    if config_path is None:
        # Default path relative to project root
        config_path = "config/speech_config.yaml"
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load YAML
    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        config_dict = {}
    
    # Create config objects from dictionary
    config = SpeechAnalysisConfig(
        audio=_create_audio_config(config_dict.get('audio', {})),
        speech_to_text=_create_stt_config(config_dict.get('speech_to_text', {})),
        quality=_create_quality_config(config_dict.get('quality', {})),
        storage=_create_storage_config(config_dict.get('storage', {})),
        performance=_create_performance_config(config_dict.get('performance', {})),
        logging=_create_logging_config(config_dict.get('logging', {}))
    )
    
    # Validate configuration
    config.validate()
    
    return config


def _create_audio_config(config_dict: Dict[str, Any]) -> AudioConfig:
    """Create AudioConfig from dictionary."""
    return AudioConfig(
        sample_rate=config_dict.get('sample_rate', 16000),
        channels=config_dict.get('channels', 1),
        format=config_dict.get('format', 'paInt16'),
        chunk_size=config_dict.get('chunk_size', 1024),
        buffer_duration=config_dict.get('buffer_duration', 30),
        device_index=config_dict.get('device_index')
    )


def _create_stt_config(config_dict: Dict[str, Any]) -> STTConfig:
    """Create STTConfig, EnhancedSTTConfig, or WhisperSTTConfig from dictionary.
    
    Supports backward compatibility with legacy config formats:
    - Old format: Basic STTConfig with vosk_model_path, language, etc.
    - Whisper format: WhisperSTTConfig with use_whisper flag
    - Enhanced format: EnhancedSTTConfig with model_type field
    """
    # Check for enhanced STT config (new format)
    use_enhanced = config_dict.get('use_enhanced', False)
    model_type = config_dict.get('model_type')
    
    # Check for Whisper STT (legacy enhanced format)
    use_whisper = config_dict.get('use_whisper', False)
    
    # Priority: use_enhanced > model_type > use_whisper > basic
    if use_enhanced or model_type:
        return EnhancedSTTConfig(
            # Base STT config
            vosk_model_path=config_dict.get('vosk_model_path', 'models/vosk-model-vn-0.4'),
            language=config_dict.get('language', 'vi'),
            sample_rate=config_dict.get('sample_rate', 16000),
            chunk_duration=config_dict.get('chunk_duration', 2.0),
            overlap_duration=config_dict.get('overlap_duration', 0.5),
            max_latency=config_dict.get('max_latency', 2.0),
            # Enhanced STT config
            model_type=config_dict.get('model_type', 'whisper'),
            model_size=config_dict.get('model_size', 'base'),
            model_path=config_dict.get('model_path'),
            compute_type=config_dict.get('compute_type', 'int8'),
            device=config_dict.get('device', 'cpu'),
            enable_audio_cleaning=config_dict.get('enable_audio_cleaning', True),
            enable_vad=config_dict.get('enable_vad', True),
            vad_method=config_dict.get('vad_method', 'silero'),
            max_buffer_size=config_dict.get('max_buffer_size', 10),
            vocabulary_file=config_dict.get('vocabulary_file'),
            enable_vocabulary=config_dict.get('enable_vocabulary', False),
            num_threads=config_dict.get('num_threads', 4),
            max_memory_mb=config_dict.get('max_memory_mb', 500),
            fallback_to_vosk=config_dict.get('fallback_to_vosk', True)
        )
    elif use_whisper:
        # Legacy Whisper format for backward compatibility
        return WhisperSTTConfig(
            # Base STT config
            vosk_model_path=config_dict.get('vosk_model_path', 'models/vosk-model-vn-0.4'),
            language=config_dict.get('language', 'vi'),
            sample_rate=config_dict.get('sample_rate', 16000),
            chunk_duration=config_dict.get('chunk_duration', 2.0),
            overlap_duration=config_dict.get('overlap_duration', 0.5),
            max_latency=config_dict.get('max_latency', 2.0),
            # Whisper STT config
            use_whisper=True,
            model_size=config_dict.get('model_size', 'base'),
            model_path=config_dict.get('model_path'),
            compute_type=config_dict.get('compute_type', 'int8'),
            device=config_dict.get('device', 'cpu'),
            num_threads=config_dict.get('num_threads', 4),
            task=config_dict.get('task', 'transcribe'),
            enable_audio_cleaning=config_dict.get('enable_audio_cleaning', True),
            enable_vad=config_dict.get('enable_vad', True),
            vad_method=config_dict.get('vad_method', 'silero'),
            overlap_buffer_duration=config_dict.get('overlap_buffer_duration', 0.5),
            max_buffer_size=config_dict.get('max_buffer_size', 10),
            vocabulary_file=config_dict.get('vocabulary_file'),
            enable_vocabulary=config_dict.get('enable_vocabulary', False),
            max_memory_mb=config_dict.get('max_memory_mb', 500),
            beam_size=config_dict.get('beam_size', 5),
            best_of=config_dict.get('best_of', 5),
            word_timestamps=config_dict.get('word_timestamps', True),
            condition_on_previous_text=config_dict.get('condition_on_previous_text', True)
        )
    else:
        # Basic STT config (legacy format)
        return STTConfig(
            vosk_model_path=config_dict.get('vosk_model_path', 'models/vosk-model-vn-0.4'),
            language=config_dict.get('language', 'vi'),
            sample_rate=config_dict.get('sample_rate', 16000),
            chunk_duration=config_dict.get('chunk_duration', 2.0),
            overlap_duration=config_dict.get('overlap_duration', 0.5),
            max_latency=config_dict.get('max_latency', 2.0)
        )


def _create_quality_config(config_dict: Dict[str, Any]) -> QualityConfig:
    """Create QualityConfig from dictionary."""
    return QualityConfig(
        clarity_threshold=config_dict.get('clarity_threshold', 60.0),
        snr_weight=config_dict.get('snr_weight', 0.4),
        spectral_weight=config_dict.get('spectral_weight', 0.4),
        zcr_weight=config_dict.get('zcr_weight', 0.2),
        fluency_threshold=config_dict.get('fluency_threshold', 60.0),
        speech_rate_weight=config_dict.get('speech_rate_weight', 0.3),
        pause_weight=config_dict.get('pause_weight', 0.4),
        rhythm_weight=config_dict.get('rhythm_weight', 0.3),
        optimal_speech_rate_min=config_dict.get('optimal_speech_rate_min', 3.0),
        optimal_speech_rate_max=config_dict.get('optimal_speech_rate_max', 5.0),
        pause_threshold=config_dict.get('pause_threshold', 0.3),
        long_pause_threshold=config_dict.get('long_pause_threshold', 1.0),
        update_interval=config_dict.get('update_interval', 2.0)
    )


def _create_storage_config(config_dict: Dict[str, Any]) -> StorageConfig:
    """Create StorageConfig from dictionary."""
    return StorageConfig(
        output_dir=config_dict.get('output_dir', 'transcripts'),
        filename_format=config_dict.get('filename_format', 'transcript_%Y%m%d_%H%M%S.txt'),
        encoding=config_dict.get('encoding', 'utf-8'),
        auto_save_interval=config_dict.get('auto_save_interval', 30),
        create_backup=config_dict.get('create_backup', True),
        include_metadata=config_dict.get('include_metadata', True),
        include_statistics=config_dict.get('include_statistics', True),
        include_recommendations=config_dict.get('include_recommendations', True)
    )


def _create_performance_config(config_dict: Dict[str, Any]) -> PerformanceConfig:
    """Create PerformanceConfig from dictionary."""
    return PerformanceConfig(
        max_memory_mb=config_dict.get('max_memory_mb', 100),
        audio_thread_priority=config_dict.get('audio_thread_priority', 'high'),
        processing_thread_priority=config_dict.get('processing_thread_priority', 'normal'),
        gui_update_rate=config_dict.get('gui_update_rate', 10)
    )


def _create_logging_config(config_dict: Dict[str, Any]) -> LoggingConfig:
    """Create LoggingConfig from dictionary."""
    return LoggingConfig(
        enabled=config_dict.get('enabled', True),
        log_file=config_dict.get('log_file', 'logs/speech_analysis.log'),
        log_level=config_dict.get('log_level', 'INFO'),
        max_log_size_mb=config_dict.get('max_log_size_mb', 10),
        backup_count=config_dict.get('backup_count', 3)
    )


def load_video_transcription_config(config_path: Optional[str] = None) -> VideoTranscriptionConfig:
    """
    Load video transcription configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default path.
        
    Returns:
        VideoTranscriptionConfig object with loaded configuration.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config validation fails.
        yaml.YAMLError: If YAML parsing fails.
    """
    if config_path is None:
        # Default path relative to project root
        config_path = "config/video_transcription_config.yaml"
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load YAML
    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        config_dict = {}
    
    # Create config object from dictionary
    config = _create_video_transcription_config(config_dict)
    
    # Validate configuration
    config.validate()
    
    return config


def _create_video_transcription_config(config_dict: Dict[str, Any]) -> VideoTranscriptionConfig:
    """Create VideoTranscriptionConfig from dictionary."""
    return VideoTranscriptionConfig(
        # Whisper settings
        whisper_model_size=config_dict.get('whisper_model_size', 'large-v3'),
        whisper_device=config_dict.get('whisper_device', 'cpu'),
        whisper_compute_type=config_dict.get('whisper_compute_type', 'int8'),
        # Processing settings
        chunk_duration=config_dict.get('chunk_duration', 30.0),
        overlap_duration=config_dict.get('overlap_duration', 1.0),
        max_parallel_chunks=config_dict.get('max_parallel_chunks', 2),
        # Language detection
        auto_detect_language=config_dict.get('auto_detect_language', True),
        default_language=config_dict.get('default_language', 'vi'),
        language_detection_duration=config_dict.get('language_detection_duration', 30.0),
        # Speaker diarization
        enable_diarization=config_dict.get('enable_diarization', False),
        min_speakers=config_dict.get('min_speakers', 1),
        max_speakers=config_dict.get('max_speakers', 10),
        # Subtitle generation
        subtitle_max_chars_per_line=config_dict.get('subtitle_max_chars_per_line', 42),
        subtitle_max_lines=config_dict.get('subtitle_max_lines', 2),
        subtitle_min_duration=config_dict.get('subtitle_min_duration', 1.0),
        subtitle_max_duration=config_dict.get('subtitle_max_duration', 7.0),
        # Cache settings
        enable_cache=config_dict.get('enable_cache', True),
        cache_dir=config_dict.get('cache_dir', '.cache/transcriptions'),
        cache_max_size_mb=config_dict.get('cache_max_size_mb', 1000),
        # Vietnamese optimization
        enable_vietnamese_optimization=config_dict.get('enable_vietnamese_optimization', True),
        vietnamese_vocabulary_file=config_dict.get('vietnamese_vocabulary_file'),
        vietnamese_diacritic_validation=config_dict.get('vietnamese_diacritic_validation', True),
        # Performance
        use_audio_enhancement=config_dict.get('use_audio_enhancement', True),
        use_vad=config_dict.get('use_vad', True)
    )


def save_video_transcription_config(config: VideoTranscriptionConfig, config_path: str) -> None:
    """
    Save video transcription configuration to YAML file.
    
    Args:
        config: VideoTranscriptionConfig object to save.
        config_path: Path where to save the config file.
    """
    config_dict = {
        # Whisper settings
        'whisper_model_size': config.whisper_model_size,
        'whisper_device': config.whisper_device,
        'whisper_compute_type': config.whisper_compute_type,
        # Processing settings
        'chunk_duration': config.chunk_duration,
        'overlap_duration': config.overlap_duration,
        'max_parallel_chunks': config.max_parallel_chunks,
        # Language detection
        'auto_detect_language': config.auto_detect_language,
        'default_language': config.default_language,
        'language_detection_duration': config.language_detection_duration,
        # Speaker diarization
        'enable_diarization': config.enable_diarization,
        'min_speakers': config.min_speakers,
        'max_speakers': config.max_speakers,
        # Subtitle generation
        'subtitle_max_chars_per_line': config.subtitle_max_chars_per_line,
        'subtitle_max_lines': config.subtitle_max_lines,
        'subtitle_min_duration': config.subtitle_min_duration,
        'subtitle_max_duration': config.subtitle_max_duration,
        # Cache settings
        'enable_cache': config.enable_cache,
        'cache_dir': config.cache_dir,
        'cache_max_size_mb': config.cache_max_size_mb,
        # Vietnamese optimization
        'enable_vietnamese_optimization': config.enable_vietnamese_optimization,
        'vietnamese_vocabulary_file': config.vietnamese_vocabulary_file,
        'vietnamese_diacritic_validation': config.vietnamese_diacritic_validation,
        # Performance
        'use_audio_enhancement': config.use_audio_enhancement,
        'use_vad': config.use_vad
    }
    
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def save_config(config: SpeechAnalysisConfig, config_path: str) -> None:
    """
    Save speech analysis configuration to YAML file.
    
    Args:
        config: Configuration object to save.
        config_path: Path where to save the config file.
    """
    # Build speech_to_text config dict
    stt_dict = {
        'vosk_model_path': config.speech_to_text.vosk_model_path,
        'language': config.speech_to_text.language,
        'chunk_duration': config.speech_to_text.chunk_duration,
        'overlap_duration': config.speech_to_text.overlap_duration,
        'max_latency': config.speech_to_text.max_latency
    }
    
    # Add Enhanced STT fields if using EnhancedSTTConfig
    if isinstance(config.speech_to_text, EnhancedSTTConfig):
        stt_dict.update({
            'use_enhanced': True,
            'model_type': config.speech_to_text.model_type,
            'model_size': config.speech_to_text.model_size,
            'model_path': config.speech_to_text.model_path,
            'compute_type': config.speech_to_text.compute_type,
            'device': config.speech_to_text.device,
            'enable_audio_cleaning': config.speech_to_text.enable_audio_cleaning,
            'enable_vad': config.speech_to_text.enable_vad,
            'vad_method': config.speech_to_text.vad_method,
            'max_buffer_size': config.speech_to_text.max_buffer_size,
            'vocabulary_file': config.speech_to_text.vocabulary_file,
            'enable_vocabulary': config.speech_to_text.enable_vocabulary,
            'num_threads': config.speech_to_text.num_threads,
            'max_memory_mb': config.speech_to_text.max_memory_mb,
            'fallback_to_vosk': config.speech_to_text.fallback_to_vosk
        })
    # Add Whisper STT fields if using WhisperSTTConfig (legacy)
    elif isinstance(config.speech_to_text, WhisperSTTConfig):
        stt_dict.update({
            'use_whisper': config.speech_to_text.use_whisper,
            'model_size': config.speech_to_text.model_size,
            'model_path': config.speech_to_text.model_path,
            'compute_type': config.speech_to_text.compute_type,
            'device': config.speech_to_text.device,
            'num_threads': config.speech_to_text.num_threads,
            'task': config.speech_to_text.task,
            'enable_audio_cleaning': config.speech_to_text.enable_audio_cleaning,
            'enable_vad': config.speech_to_text.enable_vad,
            'vad_method': config.speech_to_text.vad_method,
            'overlap_buffer_duration': config.speech_to_text.overlap_buffer_duration,
            'max_buffer_size': config.speech_to_text.max_buffer_size,
            'vocabulary_file': config.speech_to_text.vocabulary_file,
            'enable_vocabulary': config.speech_to_text.enable_vocabulary,
            'max_memory_mb': config.speech_to_text.max_memory_mb,
            'beam_size': config.speech_to_text.beam_size,
            'best_of': config.speech_to_text.best_of,
            'word_timestamps': config.speech_to_text.word_timestamps,
            'condition_on_previous_text': config.speech_to_text.condition_on_previous_text
        })
    
    config_dict = {
        'audio': {
            'sample_rate': config.audio.sample_rate,
            'channels': config.audio.channels,
            'format': config.audio.format,
            'chunk_size': config.audio.chunk_size,
            'buffer_duration': config.audio.buffer_duration,
            'device_index': config.audio.device_index
        },
        'speech_to_text': stt_dict,
        'quality': {
            'clarity_threshold': config.quality.clarity_threshold,
            'snr_weight': config.quality.snr_weight,
            'spectral_weight': config.quality.spectral_weight,
            'zcr_weight': config.quality.zcr_weight,
            'fluency_threshold': config.quality.fluency_threshold,
            'speech_rate_weight': config.quality.speech_rate_weight,
            'pause_weight': config.quality.pause_weight,
            'rhythm_weight': config.quality.rhythm_weight,
            'optimal_speech_rate_min': config.quality.optimal_speech_rate_min,
            'optimal_speech_rate_max': config.quality.optimal_speech_rate_max,
            'pause_threshold': config.quality.pause_threshold,
            'long_pause_threshold': config.quality.long_pause_threshold,
            'update_interval': config.quality.update_interval
        },
        'storage': {
            'output_dir': config.storage.output_dir,
            'filename_format': config.storage.filename_format,
            'encoding': config.storage.encoding,
            'auto_save_interval': config.storage.auto_save_interval,
            'create_backup': config.storage.create_backup,
            'include_metadata': config.storage.include_metadata,
            'include_statistics': config.storage.include_statistics,
            'include_recommendations': config.storage.include_recommendations
        },
        'performance': {
            'max_memory_mb': config.performance.max_memory_mb,
            'audio_thread_priority': config.performance.audio_thread_priority,
            'processing_thread_priority': config.performance.processing_thread_priority,
            'gui_update_rate': config.performance.gui_update_rate
        },
        'logging': {
            'enabled': config.logging.enabled,
            'log_file': config.logging.log_file,
            'log_level': config.logging.log_level,
            'max_log_size_mb': config.logging.max_log_size_mb,
            'backup_count': config.logging.backup_count
        }
    }
    
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
