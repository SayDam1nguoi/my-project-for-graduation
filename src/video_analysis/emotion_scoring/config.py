"""
Configuration loader for Emotion Scoring System.

Provides utilities to load and validate scoring configurations from YAML files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from .models import ScoringConfig


def load_config_from_yaml(config_path: str) -> ScoringConfig:
    """
    Load scoring configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        ScoringConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # Extract scoring configuration
    criterion_weights = config_data.get('criterion_weights', {})
    score_thresholds = config_data.get('score_thresholds', {})
    
    # Convert threshold lists to tuples
    score_thresholds = {
        name: tuple(threshold) 
        for name, threshold in score_thresholds.items()
    }
    
    # Create ScoringConfig
    scoring_config = ScoringConfig(
        criterion_weights=criterion_weights,
        score_thresholds=score_thresholds,
        enable_micro_expression_analysis=config_data.get('enable_micro_expression_analysis', True),
        enable_gaze_tracking=config_data.get('enable_gaze_tracking', True),
        frame_sampling_rate=config_data.get('frame_sampling_rate', 1),
    )
    
    # Add final_score_multiplier if present (for new system: 1.0 to keep 0-10 scale)
    if 'final_score_multiplier' in config_data:
        scoring_config.final_score_multiplier = config_data['final_score_multiplier']
    
    return scoring_config


def save_config_to_yaml(config: ScoringConfig, output_path: str) -> None:
    """
    Save scoring configuration to YAML file.
    
    Args:
        config: ScoringConfig instance
        output_path: Path to output YAML file
    """
    config_data = {
        'criterion_weights': config.criterion_weights,
        'score_thresholds': {
            name: list(threshold)
            for name, threshold in config.score_thresholds.items()
        },
        'enable_micro_expression_analysis': config.enable_micro_expression_analysis,
        'enable_gaze_tracking': config.enable_gaze_tracking,
        'frame_sampling_rate': config.frame_sampling_rate,
    }
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)


def get_default_config_path() -> str:
    """
    Get path to default configuration file.
    
    Returns:
        Path to default config file
    """
    # Assume config is in project root/config directory
    return 'config/emotion_scoring_config.yaml'


def load_default_config() -> ScoringConfig:
    """
    Load default configuration (4-criterion format).
    
    Returns:
        Default ScoringConfig instance with 4 core criteria
        
    Note:
        Falls back to programmatic default if config file doesn't exist
    """
    try:
        config_path = get_default_config_path()
        return load_config_from_yaml(config_path)
    except FileNotFoundError:
        # Fall back to programmatic default (new 4-criteria system)
        config = ScoringConfig(
            criterion_weights={
                'emotion_stability': 0.40,
                'emotion_content_alignment': 0.35,
                'positive_ratio': 0.15,
                'negative_overload': 0.10,
            },
            score_thresholds={
                'emotion_stability': (4.0, 7.0),
                'emotion_content_alignment': (4.0, 7.0),
                'positive_ratio': (4.0, 7.0),
                'negative_overload': (4.0, 7.0),
            },
            enable_micro_expression_analysis=True,
            enable_gaze_tracking=True,
            frame_sampling_rate=1,
        )
        # Add final_score_multiplier for new system (keep 0-10 scale)
        config.final_score_multiplier = 1.0
        return config


def validate_config(config_data: Dict[str, Any]) -> bool:
    """
    Validate configuration data before creating ScoringConfig.
    
    Args:
        config_data: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required fields
        if 'criterion_weights' not in config_data:
            return False
        if 'score_thresholds' not in config_data:
            return False
        
        # Validate weights sum to 1.0
        weights = config_data['criterion_weights']
        weight_sum = sum(weights.values())
        if not (0.99 <= weight_sum <= 1.01):
            return False
        
        # Validate thresholds
        thresholds = config_data['score_thresholds']
        for criterion, threshold in thresholds.items():
            if len(threshold) != 2:
                return False
            low, high = threshold
            if not (0 <= low <= high <= 10):
                return False
        
        return True
    except Exception:
        return False
