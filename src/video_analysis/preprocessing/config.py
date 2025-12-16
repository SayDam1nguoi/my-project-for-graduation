"""
Configuration classes and loaders for video preprocessing pipeline.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, Any
import yaml
from pathlib import Path


@dataclass
class NormalizationConfig:
    """Configuration for video normalization step."""
    enabled: bool = True
    target_fps: int = 30
    target_resolution: Tuple[int, int] = (640, 480)
    sharpening: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'strength': 0.5
    })
    denoising: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'strength': 10,
        'method': 'fast_nlmeans'
    })


@dataclass
class StabilizationConfig:
    """Configuration for face stabilization step."""
    enabled: bool = True
    window_size: int = 5
    method: str = "affine"  # affine, similarity
    smoothing_factor: float = 0.7


@dataclass
class LightingConfig:
    """Configuration for lighting correction step."""
    enabled: bool = True
    clahe: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'clip_limit': 2.0,
        'tile_size': [8, 8]
    })
    white_balance: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'method': 'gray_world'
    })
    contrast: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'factor': 1.2
    })


@dataclass
class QualityFilterConfig:
    """Configuration for frame quality filtering step."""
    enabled: bool = True
    blur_threshold: float = 100.0
    confidence_threshold: float = 0.9
    max_pose_angle: float = 30.0
    check_occlusion: bool = True
    min_face_size: Tuple[int, int] = (80, 80)
    min_fps_after_filter: float = 10.0


@dataclass
class AlignmentConfig:
    """Configuration for face alignment step."""
    enabled: bool = True
    method: str = "landmarks"  # landmarks, 3d_model
    crop_margin: float = 0.2
    target_size: Tuple[int, int] = (224, 224)
    correct_rotation: bool = True
    correct_perspective: bool = True


@dataclass
class TemporalSmoothingConfig:
    """Configuration for temporal smoothing step."""
    enabled: bool = True
    method: str = "ema"  # ema, sliding_window, hybrid
    window_size: int = 7
    ema_alpha: float = 0.3
    min_confidence_for_smoothing: float = 0.6


@dataclass
class PerformanceConfig:
    """Configuration for performance settings."""
    device: str = "auto"  # auto, cuda, cpu
    num_threads: int = 4
    batch_size: int = 1
    enable_caching: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    log_statistics: bool = True
    log_filtered_frames: bool = True
    save_debug_frames: bool = False
    debug_output_dir: str = "debug/preprocessing"


@dataclass
class PreprocessingConfig:
    """
    Main configuration class for video preprocessing pipeline.
    
    This class contains all configuration parameters for the 6-step preprocessing pipeline:
    1. Video Normalization
    2. Face Stabilization
    3. Lighting Correction
    4. Frame Quality Filtering
    5. Face Alignment
    6. Temporal Smoothing
    
    Attributes:
        enabled: Enable/disable entire preprocessing pipeline
        normalization: Configuration for video normalization step
        stabilization: Configuration for face stabilization step
        lighting: Configuration for lighting correction step
        quality_filter: Configuration for frame quality filtering step
        alignment: Configuration for face alignment step
        temporal_smoothing: Configuration for temporal smoothing step
        performance: Performance-related settings
        logging: Logging-related settings
    """
    
    enabled: bool = True
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    stabilization: StabilizationConfig = field(default_factory=StabilizationConfig)
    lighting: LightingConfig = field(default_factory=LightingConfig)
    quality_filter: QualityFilterConfig = field(default_factory=QualityFilterConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    temporal_smoothing: TemporalSmoothingConfig = field(default_factory=TemporalSmoothingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        # Validate normalization
        if self.normalization.enabled:
            if self.normalization.target_fps <= 0:
                raise ValueError("target_fps must be positive")
            if any(dim <= 0 for dim in self.normalization.target_resolution):
                raise ValueError("target_resolution dimensions must be positive")
            if not 0 <= self.normalization.sharpening['strength'] <= 1:
                raise ValueError("sharpening strength must be between 0 and 1")
        
        # Validate stabilization
        if self.stabilization.enabled:
            if self.stabilization.window_size < 1:
                raise ValueError("stabilization window_size must be at least 1")
            if self.stabilization.method not in ['affine', 'similarity']:
                raise ValueError("stabilization method must be 'affine' or 'similarity'")
            if not 0 <= self.stabilization.smoothing_factor <= 1:
                raise ValueError("smoothing_factor must be between 0 and 1")
        
        # Validate lighting
        if self.lighting.enabled:
            if self.lighting.clahe['enabled']:
                if self.lighting.clahe['clip_limit'] <= 0:
                    raise ValueError("CLAHE clip_limit must be positive")
            if self.lighting.contrast['enabled']:
                if self.lighting.contrast['factor'] <= 0:
                    raise ValueError("contrast factor must be positive")
        
        # Validate quality filter
        if self.quality_filter.enabled:
            if not 0 <= self.quality_filter.confidence_threshold <= 1:
                raise ValueError("confidence_threshold must be between 0 and 1")
            if self.quality_filter.max_pose_angle < 0:
                raise ValueError("max_pose_angle must be non-negative")
            if self.quality_filter.min_fps_after_filter <= 0:
                raise ValueError("min_fps_after_filter must be positive")
        
        # Validate alignment
        if self.alignment.enabled:
            if self.alignment.method not in ['landmarks', '3d_model']:
                raise ValueError("alignment method must be 'landmarks' or '3d_model'")
            if not 0 <= self.alignment.crop_margin <= 1:
                raise ValueError("crop_margin must be between 0 and 1")
            if any(dim <= 0 for dim in self.alignment.target_size):
                raise ValueError("alignment target_size dimensions must be positive")
        
        # Validate temporal smoothing
        if self.temporal_smoothing.enabled:
            if self.temporal_smoothing.method not in ['ema', 'sliding_window', 'hybrid']:
                raise ValueError("temporal_smoothing method must be 'ema', 'sliding_window', or 'hybrid'")
            if self.temporal_smoothing.window_size < 1:
                raise ValueError("temporal_smoothing window_size must be at least 1")
            if not 0 <= self.temporal_smoothing.ema_alpha <= 1:
                raise ValueError("ema_alpha must be between 0 and 1")
            if not 0 <= self.temporal_smoothing.min_confidence_for_smoothing <= 1:
                raise ValueError("min_confidence_for_smoothing must be between 0 and 1")
        
        # Validate performance
        if self.performance.device not in ['auto', 'cuda', 'cpu']:
            raise ValueError("device must be 'auto', 'cuda', or 'cpu'")
        if self.performance.num_threads < 1:
            raise ValueError("num_threads must be at least 1")
        if self.performance.batch_size < 1:
            raise ValueError("batch_size must be at least 1")


def load_preprocessing_config(config_path: Optional[str] = None) -> PreprocessingConfig:
    """
    Load preprocessing configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default config/config.yaml
        
    Returns:
        PreprocessingConfig object with loaded configuration
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If configuration is invalid
    """
    if config_path is None:
        config_path = "config/config.yaml"
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Extract video_preprocessing section
    preprocessing_dict = yaml_config.get('video_preprocessing', {})
    
    # If video_preprocessing section doesn't exist, return default config
    if not preprocessing_dict:
        return PreprocessingConfig()
    
    # Parse nested configurations
    config = PreprocessingConfig(
        enabled=preprocessing_dict.get('enabled', True),
        normalization=_parse_normalization_config(preprocessing_dict.get('normalization', {})),
        stabilization=_parse_stabilization_config(preprocessing_dict.get('stabilization', {})),
        lighting=_parse_lighting_config(preprocessing_dict.get('lighting', {})),
        quality_filter=_parse_quality_filter_config(preprocessing_dict.get('quality_filter', {})),
        alignment=_parse_alignment_config(preprocessing_dict.get('alignment', {})),
        temporal_smoothing=_parse_temporal_smoothing_config(preprocessing_dict.get('temporal_smoothing', {})),
        performance=_parse_performance_config(preprocessing_dict.get('performance', {})),
        logging=_parse_logging_config(preprocessing_dict.get('logging', {}))
    )
    
    # Validate configuration
    config.validate()
    
    return config


def _parse_normalization_config(config_dict: Dict[str, Any]) -> NormalizationConfig:
    """Parse normalization configuration from dictionary."""
    return NormalizationConfig(
        enabled=config_dict.get('enabled', True),
        target_fps=config_dict.get('target_fps', 30),
        target_resolution=tuple(config_dict.get('target_resolution', [640, 480])),
        sharpening=config_dict.get('sharpening', {
            'enabled': True,
            'strength': 0.5
        }),
        denoising=config_dict.get('denoising', {
            'enabled': True,
            'strength': 10,
            'method': 'fast_nlmeans'
        })
    )


def _parse_stabilization_config(config_dict: Dict[str, Any]) -> StabilizationConfig:
    """Parse stabilization configuration from dictionary."""
    return StabilizationConfig(
        enabled=config_dict.get('enabled', True),
        window_size=config_dict.get('window_size', 5),
        method=config_dict.get('method', 'affine'),
        smoothing_factor=config_dict.get('smoothing_factor', 0.7)
    )


def _parse_lighting_config(config_dict: Dict[str, Any]) -> LightingConfig:
    """Parse lighting configuration from dictionary."""
    clahe_dict = config_dict.get('clahe', {})
    white_balance_dict = config_dict.get('white_balance', {})
    contrast_dict = config_dict.get('contrast', {})
    
    return LightingConfig(
        enabled=config_dict.get('enabled', True),
        clahe={
            'enabled': clahe_dict.get('enabled', True),
            'clip_limit': clahe_dict.get('clip_limit', 2.0),
            'tile_size': clahe_dict.get('tile_size', [8, 8])
        },
        white_balance={
            'enabled': white_balance_dict.get('enabled', True),
            'method': white_balance_dict.get('method', 'gray_world')
        },
        contrast={
            'enabled': contrast_dict.get('enabled', True),
            'factor': contrast_dict.get('factor', 1.2)
        }
    )


def _parse_quality_filter_config(config_dict: Dict[str, Any]) -> QualityFilterConfig:
    """Parse quality filter configuration from dictionary."""
    return QualityFilterConfig(
        enabled=config_dict.get('enabled', True),
        blur_threshold=config_dict.get('blur_threshold', 100.0),
        confidence_threshold=config_dict.get('confidence_threshold', 0.9),
        max_pose_angle=config_dict.get('max_pose_angle', 30.0),
        check_occlusion=config_dict.get('check_occlusion', True),
        min_face_size=tuple(config_dict.get('min_face_size', [80, 80])),
        min_fps_after_filter=config_dict.get('min_fps_after_filter', 10.0)
    )


def _parse_alignment_config(config_dict: Dict[str, Any]) -> AlignmentConfig:
    """Parse alignment configuration from dictionary."""
    return AlignmentConfig(
        enabled=config_dict.get('enabled', True),
        method=config_dict.get('method', 'landmarks'),
        crop_margin=config_dict.get('crop_margin', 0.2),
        target_size=tuple(config_dict.get('target_size', [224, 224])),
        correct_rotation=config_dict.get('correct_rotation', True),
        correct_perspective=config_dict.get('correct_perspective', True)
    )


def _parse_temporal_smoothing_config(config_dict: Dict[str, Any]) -> TemporalSmoothingConfig:
    """Parse temporal smoothing configuration from dictionary."""
    return TemporalSmoothingConfig(
        enabled=config_dict.get('enabled', True),
        method=config_dict.get('method', 'ema'),
        window_size=config_dict.get('window_size', 7),
        ema_alpha=config_dict.get('ema_alpha', 0.3),
        min_confidence_for_smoothing=config_dict.get('min_confidence_for_smoothing', 0.6)
    )


def _parse_performance_config(config_dict: Dict[str, Any]) -> PerformanceConfig:
    """Parse performance configuration from dictionary."""
    return PerformanceConfig(
        device=config_dict.get('device', 'auto'),
        num_threads=config_dict.get('num_threads', 4),
        batch_size=config_dict.get('batch_size', 1),
        enable_caching=config_dict.get('enable_caching', True)
    )


def _parse_logging_config(config_dict: Dict[str, Any]) -> LoggingConfig:
    """Parse logging configuration from dictionary."""
    return LoggingConfig(
        log_statistics=config_dict.get('log_statistics', True),
        log_filtered_frames=config_dict.get('log_filtered_frames', True),
        save_debug_frames=config_dict.get('save_debug_frames', False),
        debug_output_dir=config_dict.get('debug_output_dir', 'debug/preprocessing')
    )
