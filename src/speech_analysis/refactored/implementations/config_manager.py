"""Configuration manager implementation."""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from copy import deepcopy
import logging

from ..interfaces.config import IConfigManager, ValidationResult
from ..models.config import (
    STTConfig,
    AudioConfig,
    EngineConfig,
    ProcessingConfig,
    PerformanceConfig,
    QualityConfig,
    EngineType,
    DeviceType,
)
from ..exceptions import ConfigurationException


logger = logging.getLogger(__name__)


class ConfigManager(IConfigManager):
    """
    Configuration manager for STT system.
    
    Handles loading, saving, validation, and management of configurations.
    Supports predefined profiles and hot-reloading.
    """
    
    def __init__(self):
        """Initialize configuration manager."""
        self._profiles: Dict[str, STTConfig] = self._initialize_profiles()
        self._config_cache: Dict[Path, STTConfig] = {}
        self._file_timestamps: Dict[Path, float] = {}
    
    def load_config(self, path: Path) -> STTConfig:
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to configuration file.
            
        Returns:
            Loaded configuration.
            
        Raises:
            FileNotFoundError: If config file not found.
            ConfigurationError: If config is invalid.
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            if config_dict is None:
                raise ConfigurationException("Configuration file is empty")
            
            # Convert dictionary to STTConfig
            config = self._dict_to_config(config_dict)
            
            # Validate configuration
            validation = self.validate_config(config)
            if not validation.is_valid:
                error_msg = "Configuration validation failed:\n" + "\n".join(
                    f"  - {error}" for error in validation.errors
                )
                raise ConfigurationException(error_msg)
            
            # Log warnings if any
            for warning in validation.warnings:
                logger.warning(f"Configuration warning: {warning}")
            
            # Cache configuration and timestamp
            self._config_cache[path] = config
            self._file_timestamps[path] = path.stat().st_mtime
            
            logger.info(f"Configuration loaded from {path}")
            return config
            
        except yaml.YAMLError as e:
            raise ConfigurationException(f"Failed to parse YAML: {e}")
        except Exception as e:
            raise ConfigurationException(f"Failed to load configuration: {e}")
    
    def save_config(self, config: STTConfig, path: Path) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration to save.
            path: Path to save configuration file.
        """
        try:
            # Convert config to dictionary
            config_dict = self._config_to_dict(config)
            
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to YAML
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    config_dict,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
            
            logger.info(f"Configuration saved to {path}")
            
        except Exception as e:
            raise ConfigurationException(f"Failed to save configuration: {e}")
    
    def validate_config(self, config: STTConfig) -> ValidationResult:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate.
            
        Returns:
            Validation result with errors and warnings.
        """
        result = ValidationResult()
        
        # Validate audio config
        try:
            if config.audio.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
                result.add_warning(
                    f"Unusual sample rate: {config.audio.sample_rate}. "
                    "Recommended: 16000 for speech."
                )
            
            if config.audio.chunk_duration > 30.0:
                result.add_warning(
                    f"Large chunk duration: {config.audio.chunk_duration}s. "
                    "May increase latency."
                )
            
            if config.audio.overlap_duration >= config.audio.chunk_duration:
                result.add_error(
                    "Overlap duration must be less than chunk duration"
                )
        except Exception as e:
            result.add_error(f"Audio config validation failed: {e}")
        
        # Validate engine config
        try:
            valid_models = {
                EngineType.WHISPER: ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                EngineType.FASTER_WHISPER: ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                EngineType.VOSK: ["vosk-model-small", "vosk-model-en", "vosk-model-vn"],
            }
            
            if config.engine.engine_type in valid_models:
                if config.engine.model_size not in valid_models[config.engine.engine_type]:
                    result.add_warning(
                        f"Unusual model size '{config.engine.model_size}' "
                        f"for {config.engine.engine_type.value}"
                    )
        except Exception as e:
            result.add_error(f"Engine config validation failed: {e}")
        
        # Validate processing config
        try:
            if config.processing.min_confidence < 0.3:
                result.add_warning(
                    f"Very low confidence threshold: {config.processing.min_confidence}. "
                    "May include low-quality transcriptions."
                )
            
            if config.processing.compression_ratio_threshold < 2.0:
                result.add_warning(
                    "Low compression ratio threshold may miss hallucinations"
                )
        except Exception as e:
            result.add_error(f"Processing config validation failed: {e}")
        
        # Validate performance config
        try:
            if config.performance.max_memory_mb < 500:
                result.add_error(
                    f"Insufficient memory: {config.performance.max_memory_mb}MB. "
                    "Minimum 500MB required."
                )
            
            if config.performance.target_rtf < 0.5:
                result.add_warning(
                    f"Very aggressive RTF target: {config.performance.target_rtf}. "
                    "May not be achievable."
                )
        except Exception as e:
            result.add_error(f"Performance config validation failed: {e}")
        
        # Check compatibility
        compat_result = self.check_compatibility(config)
        result.errors.extend(compat_result.errors)
        result.warnings.extend(compat_result.warnings)
        if not compat_result.is_valid:
            result.is_valid = False
        
        return result
    
    def get_profile(self, profile_name: str) -> STTConfig:
        """
        Get predefined configuration profile.
        
        Args:
            profile_name: Name of profile.
            
        Returns:
            Configuration for the profile.
            
        Raises:
            ValueError: If profile not found.
        """
        if profile_name not in self._profiles:
            available = ", ".join(self._profiles.keys())
            raise ValueError(
                f"Unknown profile: {profile_name}. "
                f"Available profiles: {available}"
            )
        
        # Return a deep copy to prevent modification
        return deepcopy(self._profiles[profile_name])
    
    def list_profiles(self) -> List[str]:
        """
        List available configuration profiles.
        
        Returns:
            List of profile names.
        """
        return list(self._profiles.keys())
    
    def merge_configs(
        self,
        base_config: STTConfig,
        override_config: Dict[str, Any],
    ) -> STTConfig:
        """
        Merge configuration with overrides.
        
        Args:
            base_config: Base configuration.
            override_config: Dictionary of overrides.
            
        Returns:
            Merged configuration.
        """
        # Convert base config to dict
        config_dict = self._config_to_dict(base_config)
        
        # Deep merge override config
        merged_dict = self._deep_merge(config_dict, override_config)
        
        # Convert back to STTConfig
        return self._dict_to_config(merged_dict)
    
    def get_default_config(self) -> STTConfig:
        """
        Get default configuration.
        
        Returns:
            Default configuration (balanced profile).
        """
        return self.get_profile("balanced")
    
    def check_compatibility(self, config: STTConfig) -> ValidationResult:
        """
        Check compatibility between configuration settings.
        
        Args:
            config: Configuration to check.
            
        Returns:
            Validation result with compatibility issues.
        """
        result = ValidationResult()
        
        # Check engine and device compatibility
        if config.engine.device == DeviceType.CUDA:
            try:
                import torch
                if not torch.cuda.is_available():
                    result.add_error(
                        "CUDA device specified but CUDA is not available. "
                        "Install CUDA or use CPU."
                    )
            except ImportError:
                result.add_error(
                    "CUDA device specified but PyTorch is not installed"
                )
        
        # Check model size and memory compatibility
        model_memory_requirements = {
            "tiny": 100,
            "base": 200,
            "small": 500,
            "medium": 1500,
            "large": 3000,
            "large-v2": 3000,
            "large-v3": 3000,
        }
        
        required_memory = model_memory_requirements.get(
            config.engine.model_size,
            1000
        )
        
        if config.performance.max_memory_mb < required_memory:
            result.add_warning(
                f"Model '{config.engine.model_size}' typically requires "
                f"{required_memory}MB but max_memory is set to "
                f"{config.performance.max_memory_mb}MB. "
                "May cause out-of-memory errors."
            )
        
        # Check RTF and model size compatibility
        if config.engine.model_size in ["large", "large-v2", "large-v3"]:
            if config.performance.target_rtf < 2.0:
                result.add_warning(
                    f"Large models typically have RTF > 2.0 on CPU. "
                    f"Target RTF of {config.performance.target_rtf} may not be achievable."
                )
        
        # Check VAD and preprocessing compatibility
        if not config.processing.enable_vad and config.processing.enable_preprocessing:
            result.add_warning(
                "VAD is disabled but preprocessing is enabled. "
                "Preprocessing may process non-speech audio."
            )
        
        # Check hallucination filter and confidence threshold
        if config.processing.enable_hallucination_filter:
            if config.processing.min_confidence > 0.8:
                result.add_warning(
                    "High confidence threshold with hallucination filter "
                    "may filter too much content."
                )
        
        return result
    
    def check_hot_reload(self, path: Path) -> Optional[STTConfig]:
        """
        Check if configuration file has been modified and reload if needed.
        
        Args:
            path: Path to configuration file.
            
        Returns:
            Reloaded configuration if file was modified, None otherwise.
        """
        if path not in self._file_timestamps:
            return None
        
        try:
            current_mtime = path.stat().st_mtime
            if current_mtime > self._file_timestamps[path]:
                logger.info(f"Configuration file modified, reloading: {path}")
                return self.load_config(path)
        except Exception as e:
            logger.error(f"Failed to check hot reload: {e}")
        
        return None
    
    def _initialize_profiles(self) -> Dict[str, STTConfig]:
        """Initialize predefined configuration profiles."""
        return {
            "accuracy": STTConfig(
                audio=AudioConfig(
                    chunk_duration=5.0,
                    overlap_duration=1.0,
                ),
                engine=EngineConfig(
                    engine_type=EngineType.WHISPER,
                    model_size="large-v3",
                    beam_size=10,
                    best_of=10,
                    temperature=0.0,
                ),
                processing=ProcessingConfig(
                    enable_vad=True,
                    enable_preprocessing=True,
                    enable_hallucination_filter=True,
                    enable_post_processing=True,
                    min_confidence=0.6,
                ),
                performance=PerformanceConfig(
                    max_memory_mb=3000,
                    target_rtf=3.0,
                    enable_adaptive_adjustment=False,
                ),
            ),
            "balanced": STTConfig(
                audio=AudioConfig(
                    chunk_duration=3.0,
                    overlap_duration=0.5,
                ),
                engine=EngineConfig(
                    engine_type=EngineType.FASTER_WHISPER,
                    model_size="base",
                    beam_size=5,
                    best_of=5,
                ),
                processing=ProcessingConfig(
                    enable_vad=True,
                    enable_preprocessing=True,
                    enable_hallucination_filter=True,
                    enable_post_processing=True,
                ),
                performance=PerformanceConfig(
                    max_memory_mb=1500,
                    target_rtf=1.5,
                ),
            ),
            "speed": STTConfig(
                audio=AudioConfig(
                    chunk_duration=2.0,
                    overlap_duration=0.3,
                ),
                engine=EngineConfig(
                    engine_type=EngineType.FASTER_WHISPER,
                    model_size="tiny",
                    beam_size=3,
                    best_of=3,
                ),
                processing=ProcessingConfig(
                    enable_vad=True,
                    enable_preprocessing=False,
                    enable_hallucination_filter=True,
                    enable_post_processing=False,
                ),
                performance=PerformanceConfig(
                    max_memory_mb=800,
                    target_rtf=1.0,
                ),
            ),
            "cpu-optimized": STTConfig(
                audio=AudioConfig(
                    chunk_duration=3.0,
                    overlap_duration=0.5,
                ),
                engine=EngineConfig(
                    engine_type=EngineType.FASTER_WHISPER,
                    model_size="base",
                    device=DeviceType.CPU,
                    compute_type="int8",
                    beam_size=5,
                ),
                processing=ProcessingConfig(
                    enable_vad=True,
                    enable_preprocessing=True,
                    enable_hallucination_filter=True,
                    enable_post_processing=True,
                ),
                performance=PerformanceConfig(
                    max_memory_mb=1000,
                    cpu_limit_percent=70.0,
                    target_rtf=1.5,
                ),
            ),
            "low-memory": STTConfig(
                audio=AudioConfig(
                    chunk_duration=2.0,
                    overlap_duration=0.3,
                    buffer_size_seconds=15.0,
                ),
                engine=EngineConfig(
                    engine_type=EngineType.FASTER_WHISPER,
                    model_size="tiny",
                    compute_type="int8",
                ),
                processing=ProcessingConfig(
                    enable_vad=True,
                    enable_preprocessing=False,
                    enable_hallucination_filter=True,
                    enable_post_processing=False,
                ),
                performance=PerformanceConfig(
                    max_memory_mb=500,
                    target_rtf=1.2,
                ),
            ),
        }
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> STTConfig:
        """Convert dictionary to STTConfig."""
        # Extract sub-configs
        audio_dict = config_dict.get("audio", {})
        engine_dict = config_dict.get("engine", {})
        processing_dict = config_dict.get("processing", {})
        performance_dict = config_dict.get("performance", {})
        quality_dict = config_dict.get("quality", {})
        
        # Convert engine_type string to enum
        if "engine_type" in engine_dict:
            engine_type_str = engine_dict["engine_type"]
            engine_dict["engine_type"] = EngineType(engine_type_str)
        
        # Convert device string to enum
        if "device" in engine_dict:
            device_str = engine_dict["device"]
            engine_dict["device"] = DeviceType(device_str)
        
        # Create config objects
        audio_config = AudioConfig(**audio_dict) if audio_dict else AudioConfig()
        engine_config = EngineConfig(**engine_dict) if engine_dict else EngineConfig(
            engine_type=EngineType.FASTER_WHISPER
        )
        processing_config = ProcessingConfig(**processing_dict) if processing_dict else ProcessingConfig()
        performance_config = PerformanceConfig(**performance_dict) if performance_dict else PerformanceConfig()
        quality_config = QualityConfig(**quality_dict) if quality_dict else QualityConfig()
        
        # Create main config
        return STTConfig(
            audio=audio_config,
            engine=engine_config,
            processing=processing_config,
            performance=performance_config,
            quality=quality_config,
            log_level=config_dict.get("log_level", "INFO"),
            enable_metrics=config_dict.get("enable_metrics", True),
            output_format=config_dict.get("output_format", "text"),
            custom_vocabulary=config_dict.get("custom_vocabulary", []),
            metadata=config_dict.get("metadata", {}),
        )
    
    def _config_to_dict(self, config: STTConfig) -> Dict[str, Any]:
        """Convert STTConfig to dictionary."""
        return {
            "audio": {
                "sample_rate": config.audio.sample_rate,
                "chunk_duration": config.audio.chunk_duration,
                "overlap_duration": config.audio.overlap_duration,
                "channels": config.audio.channels,
                "device_index": config.audio.device_index,
                "buffer_size_seconds": config.audio.buffer_size_seconds,
            },
            "engine": {
                "engine_type": config.engine.engine_type.value,
                "model_size": config.engine.model_size,
                "device": config.engine.device.value,
                "language": config.engine.language,
                "beam_size": config.engine.beam_size,
                "best_of": config.engine.best_of,
                "temperature": config.engine.temperature,
                "compute_type": config.engine.compute_type,
                "num_workers": config.engine.num_workers,
                "download_root": config.engine.download_root,
                "model_path": config.engine.model_path,
            },
            "processing": {
                "enable_vad": config.processing.enable_vad,
                "enable_preprocessing": config.processing.enable_preprocessing,
                "enable_hallucination_filter": config.processing.enable_hallucination_filter,
                "enable_post_processing": config.processing.enable_post_processing,
                "vad_method": config.processing.vad_method,
                "noise_reduction": config.processing.noise_reduction,
                "normalization": config.processing.normalization,
                "pre_emphasis": config.processing.pre_emphasis,
                "min_confidence": config.processing.min_confidence,
                "no_speech_threshold": config.processing.no_speech_threshold,
                "compression_ratio_threshold": config.processing.compression_ratio_threshold,
                "early_segment_threshold_seconds": config.processing.early_segment_threshold_seconds,
            },
            "performance": {
                "max_memory_mb": config.performance.max_memory_mb,
                "cpu_limit_percent": config.performance.cpu_limit_percent,
                "target_rtf": config.performance.target_rtf,
                "enable_adaptive_adjustment": config.performance.enable_adaptive_adjustment,
                "monitoring_interval_seconds": config.performance.monitoring_interval_seconds,
            },
            "quality": {
                "min_clarity_score": config.quality.min_clarity_score,
                "min_fluency_score": config.quality.min_fluency_score,
                "min_snr_db": config.quality.min_snr_db,
                "max_hallucination_risk": config.quality.max_hallucination_risk,
            },
            "log_level": config.log_level,
            "enable_metrics": config.enable_metrics,
            "output_format": config.output_format,
            "custom_vocabulary": config.custom_vocabulary,
            "metadata": config.metadata,
        }
    
    def _deep_merge(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
