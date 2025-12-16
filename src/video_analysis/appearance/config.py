# -*- coding: utf-8 -*-
"""
Configuration for Appearance Assessment

Defines AppearanceConfig dataclass with YAML serialization support.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any, Optional
import yaml
from pathlib import Path


@dataclass
class AppearanceConfig:
    """Configuration for appearance assessment system."""
    
    # Enable/disable assessments
    lighting_enabled: bool = False
    clothing_enabled: bool = False
    
    # Real-time warnings
    real_time_warnings_enabled: bool = True
    warning_update_interval: float = 5.0  # seconds
    warning_threshold: float = 60.0  # score threshold
    
    # Lighting thresholds
    min_brightness: float = 80.0
    max_brightness: float = 200.0
    min_contrast: float = 30.0
    min_uniformity: float = 60.0
    
    # Clothing colors (HSV ranges)
    # Format: {color_name: ([h_min, s_min, v_min], [h_max, s_max, v_max])}
    professional_colors: Dict[str, Tuple[List[int], List[int]]] = field(default_factory=lambda: {
        'white': ([0, 0, 200], [180, 30, 255]),
        'black': ([0, 0, 0], [180, 255, 50]),
        'blue': ([100, 50, 50], [130, 255, 255]),
        'gray': ([0, 0, 50], [180, 50, 200]),
        'navy': ([100, 50, 30], [130, 255, 100])
    })
    
    unprofessional_colors: Dict[str, Tuple[List[int], List[int]]] = field(default_factory=lambda: {
        'bright_red': ([0, 150, 150], [10, 255, 255]),
        'bright_yellow': ([20, 150, 150], [30, 255, 255]),
        'bright_green': ([40, 150, 150], [80, 255, 255]),
        'pink': ([140, 50, 150], [170, 255, 255])
    })
    
    # Performance
    max_performance_impact: float = 0.10  # 10% max FPS degradation
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate threshold ranges and configuration values."""
        # Validate brightness thresholds
        if not (0 <= self.min_brightness <= 255):
            raise ValueError(f"min_brightness must be in range [0, 255], got {self.min_brightness}")
        if not (0 <= self.max_brightness <= 255):
            raise ValueError(f"max_brightness must be in range [0, 255], got {self.max_brightness}")
        if self.min_brightness >= self.max_brightness:
            raise ValueError(f"min_brightness ({self.min_brightness}) must be less than max_brightness ({self.max_brightness})")
        
        # Validate contrast threshold
        if not (0 <= self.min_contrast <= 100):
            raise ValueError(f"min_contrast must be in range [0, 100], got {self.min_contrast}")
        
        # Validate uniformity threshold
        if not (0 <= self.min_uniformity <= 100):
            raise ValueError(f"min_uniformity must be in range [0, 100], got {self.min_uniformity}")
        
        # Validate warning threshold
        if not (0 <= self.warning_threshold <= 100):
            raise ValueError(f"warning_threshold must be in range [0, 100], got {self.warning_threshold}")
        
        # Validate warning update interval
        if self.warning_update_interval <= 0:
            raise ValueError(f"warning_update_interval must be positive, got {self.warning_update_interval}")
        
        # Validate performance impact
        if not (0 <= self.max_performance_impact <= 1.0):
            raise ValueError(f"max_performance_impact must be in range [0, 1.0], got {self.max_performance_impact}")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'AppearanceConfig':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            AppearanceConfig instance
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML file is invalid
            ValueError: If configuration values are invalid
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Extract appearance_assessment section if it exists
        if 'appearance_assessment' in data:
            data = data['appearance_assessment']
        
        # Convert color dictionaries from YAML format to tuple format
        if 'professional_colors' in data:
            data['professional_colors'] = {
                name: (ranges['min'], ranges['max'])
                for name, ranges in data['professional_colors'].items()
            }
        
        if 'unprofessional_colors' in data:
            data['unprofessional_colors'] = {
                name: (ranges['min'], ranges['max'])
                for name, ranges in data['unprofessional_colors'].items()
            }
        
        return cls(**data)
    
    def to_yaml(self, yaml_path: str):
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration file
        """
        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        data = asdict(self)
        
        # Convert color tuples to YAML-friendly format
        data['professional_colors'] = {
            name: {'min': ranges[0], 'max': ranges[1]}
            for name, ranges in data['professional_colors'].items()
        }
        
        data['unprofessional_colors'] = {
            name: {'min': ranges[0], 'max': ranges[1]}
            for name, ranges in data['unprofessional_colors'].items()
        }
        
        # Wrap in appearance_assessment section
        yaml_data = {'appearance_assessment': data}
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppearanceConfig':
        """Create configuration from dictionary."""
        return cls(**data)
