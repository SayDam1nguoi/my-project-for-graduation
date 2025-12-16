"""Configuration management interfaces."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path

from ..models.config import STTConfig


class ValidationResult:
    """Result of configuration validation."""
    
    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def add_error(self, error: str) -> None:
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add validation warning."""
        self.warnings.append(warning)


class IConfigManager(ABC):
    """Interface for configuration management."""
    
    @abstractmethod
    def load_config(self, path: Path) -> STTConfig:
        """
        Load configuration from file.
        
        Args:
            path: Path to configuration file (YAML).
            
        Returns:
            Loaded configuration.
            
        Raises:
            FileNotFoundError: If config file not found.
            ValueError: If config is invalid.
        """
        pass
    
    @abstractmethod
    def save_config(self, config: STTConfig, path: Path) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save.
            path: Path to save configuration file.
        """
        pass
    
    @abstractmethod
    def validate_config(self, config: STTConfig) -> ValidationResult:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate.
            
        Returns:
            Validation result with errors and warnings.
        """
        pass
    
    @abstractmethod
    def get_profile(self, profile_name: str) -> STTConfig:
        """
        Get predefined configuration profile.
        
        Args:
            profile_name: Name of profile (accuracy, balanced, speed, etc.).
            
        Returns:
            Configuration for the profile.
            
        Raises:
            ValueError: If profile not found.
        """
        pass
    
    @abstractmethod
    def list_profiles(self) -> List[str]:
        """
        List available configuration profiles.
        
        Returns:
            List of profile names.
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_default_config(self) -> STTConfig:
        """
        Get default configuration.
        
        Returns:
            Default configuration.
        """
        pass
    
    @abstractmethod
    def check_compatibility(
        self,
        config: STTConfig,
    ) -> ValidationResult:
        """
        Check compatibility between configuration settings.
        
        Args:
            config: Configuration to check.
            
        Returns:
            Validation result with compatibility issues.
        """
        pass
