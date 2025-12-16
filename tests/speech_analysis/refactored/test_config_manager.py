"""Tests for configuration manager."""

import pytest
import yaml
from pathlib import Path
from tempfile import TemporaryDirectory

from src.speech_analysis.refactored.implementations import ConfigManager
from src.speech_analysis.refactored.models.config import (
    STTConfig,
    AudioConfig,
    EngineConfig,
    ProcessingConfig,
    PerformanceConfig,
    QualityConfig,
    EngineType,
    DeviceType,
)
from src.speech_analysis.refactored.exceptions import ConfigurationException


class TestConfigManager:
    """Test configuration manager."""
    
    def test_initialization(self):
        """Test configuration manager initialization."""
        manager = ConfigManager()
        
        # Should have predefined profiles
        profiles = manager.list_profiles()
        assert len(profiles) > 0
        assert "balanced" in profiles
        assert "accuracy" in profiles
        assert "speed" in profiles
    
    def test_list_profiles(self):
        """Test listing available profiles."""
        manager = ConfigManager()
        profiles = manager.list_profiles()
        
        expected_profiles = [
            "accuracy",
            "balanced",
            "speed",
            "cpu-optimized",
            "low-memory",
        ]
        
        for profile in expected_profiles:
            assert profile in profiles
    
    def test_get_profile_balanced(self):
        """Test getting balanced profile."""
        manager = ConfigManager()
        config = manager.get_profile("balanced")
        
        assert isinstance(config, STTConfig)
        assert config.engine.engine_type == EngineType.FASTER_WHISPER
        assert config.engine.model_size == "base"
        assert config.processing.enable_vad is True
        assert config.performance.target_rtf == 1.5
    
    def test_get_profile_accuracy(self):
        """Test getting accuracy profile."""
        manager = ConfigManager()
        config = manager.get_profile("accuracy")
        
        assert config.engine.engine_type == EngineType.WHISPER
        assert config.engine.model_size == "large-v3"
        assert config.engine.beam_size == 10
        assert config.performance.target_rtf == 3.0
    
    def test_get_profile_speed(self):
        """Test getting speed profile."""
        manager = ConfigManager()
        config = manager.get_profile("speed")
        
        assert config.engine.model_size == "tiny"
        assert config.processing.enable_preprocessing is False
        assert config.performance.target_rtf == 1.0
    
    def test_get_profile_invalid(self):
        """Test getting invalid profile raises error."""
        manager = ConfigManager()
        
        with pytest.raises(ValueError, match="Unknown profile"):
            manager.get_profile("nonexistent")
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        manager = ConfigManager()
        config = manager.get_default_config()
        
        # Default should be balanced profile
        balanced = manager.get_profile("balanced")
        assert config.engine.engine_type == balanced.engine.engine_type
        assert config.engine.model_size == balanced.engine.model_size
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        manager = ConfigManager()
        config = manager.get_profile("balanced")
        
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            
            # Save configuration
            manager.save_config(config, config_path)
            assert config_path.exists()
            
            # Load configuration
            loaded_config = manager.load_config(config_path)
            
            # Verify loaded config matches original
            assert loaded_config.engine.engine_type == config.engine.engine_type
            assert loaded_config.engine.model_size == config.engine.model_size
            assert loaded_config.audio.sample_rate == config.audio.sample_rate
            assert loaded_config.performance.max_memory_mb == config.performance.max_memory_mb
    
    def test_load_config_file_not_found(self):
        """Test loading non-existent config file."""
        manager = ConfigManager()
        
        with pytest.raises(FileNotFoundError):
            manager.load_config(Path("nonexistent.yaml"))
    
    def test_load_config_invalid_yaml(self):
        """Test loading invalid YAML file."""
        manager = ConfigManager()
        
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid.yaml"
            config_path.write_text("invalid: yaml: content: [")
            
            with pytest.raises(ConfigurationException, match="Failed to parse YAML"):
                manager.load_config(config_path)
    
    def test_load_config_empty_file(self):
        """Test loading empty config file."""
        manager = ConfigManager()
        
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "empty.yaml"
            config_path.write_text("")
            
            with pytest.raises(ConfigurationException, match="empty"):
                manager.load_config(config_path)
    
    def test_validate_config_valid(self):
        """Test validating valid configuration."""
        manager = ConfigManager()
        config = manager.get_profile("balanced")
        
        result = manager.validate_config(config)
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_validate_config_invalid_overlap(self):
        """Test validating config with invalid overlap."""
        manager = ConfigManager()
        config = manager.get_profile("balanced")
        
        # Set overlap >= chunk duration (invalid)
        config.audio.overlap_duration = config.audio.chunk_duration
        
        result = manager.validate_config(config)
        assert result.is_valid is False
        assert any("overlap" in error.lower() for error in result.errors)
    
    def test_validate_config_low_memory(self):
        """Test validating config with insufficient memory."""
        manager = ConfigManager()
        config = manager.get_profile("balanced")
        
        # Set very low memory
        config.performance.max_memory_mb = 100
        
        result = manager.validate_config(config)
        assert result.is_valid is False
        assert any("memory" in error.lower() for error in result.errors)
    
    def test_validate_config_warnings(self):
        """Test validation warnings."""
        manager = ConfigManager()
        config = manager.get_profile("balanced")
        
        # Set unusual sample rate
        config.audio.sample_rate = 11025
        
        result = manager.validate_config(config)
        # Should still be valid but with warnings
        assert len(result.warnings) > 0
        assert any("sample rate" in warning.lower() for warning in result.warnings)
    
    def test_check_compatibility_valid(self):
        """Test compatibility check for valid config."""
        manager = ConfigManager()
        config = manager.get_profile("balanced")
        
        result = manager.check_compatibility(config)
        assert result.is_valid is True
    
    def test_check_compatibility_large_model_low_memory(self):
        """Test compatibility check for large model with low memory."""
        manager = ConfigManager()
        config = manager.get_profile("accuracy")
        
        # Set memory too low for large model
        config.performance.max_memory_mb = 500
        
        result = manager.check_compatibility(config)
        # Should have warnings about memory
        assert len(result.warnings) > 0
        assert any("memory" in warning.lower() for warning in result.warnings)
    
    def test_check_compatibility_large_model_low_rtf(self):
        """Test compatibility check for large model with low RTF target."""
        manager = ConfigManager()
        config = manager.get_profile("accuracy")
        
        # Set unrealistic RTF for large model on CPU
        config.performance.target_rtf = 1.0
        
        result = manager.check_compatibility(config)
        assert len(result.warnings) > 0
        assert any("rtf" in warning.lower() for warning in result.warnings)
    
    def test_merge_configs_simple(self):
        """Test merging configurations."""
        manager = ConfigManager()
        base_config = manager.get_profile("balanced")
        
        overrides = {
            "engine": {
                "model_size": "medium",
            },
        }
        
        merged = manager.merge_configs(base_config, overrides)
        
        # Override should be applied
        assert merged.engine.model_size == "medium"
        # Other settings should remain
        assert merged.engine.engine_type == base_config.engine.engine_type
        assert merged.audio.sample_rate == base_config.audio.sample_rate
    
    def test_merge_configs_nested(self):
        """Test merging nested configurations."""
        manager = ConfigManager()
        base_config = manager.get_profile("balanced")
        
        overrides = {
            "engine": {
                "model_size": "small",
                "beam_size": 8,
            },
            "performance": {
                "max_memory_mb": 2000,
            },
        }
        
        merged = manager.merge_configs(base_config, overrides)
        
        assert merged.engine.model_size == "small"
        assert merged.engine.beam_size == 8
        assert merged.performance.max_memory_mb == 2000
        # Other engine settings should remain
        assert merged.engine.engine_type == base_config.engine.engine_type
    
    def test_merge_configs_preserves_base(self):
        """Test that merging doesn't modify base config."""
        manager = ConfigManager()
        base_config = manager.get_profile("balanced")
        original_model_size = base_config.engine.model_size
        
        overrides = {
            "engine": {
                "model_size": "large",
            },
        }
        
        merged = manager.merge_configs(base_config, overrides)
        
        # Base config should not be modified
        assert base_config.engine.model_size == original_model_size
        # Merged config should have override
        assert merged.engine.model_size == "large"
    
    def test_hot_reload_no_change(self):
        """Test hot reload when file hasn't changed."""
        manager = ConfigManager()
        config = manager.get_profile("balanced")
        
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            manager.save_config(config, config_path)
            
            # Load config
            manager.load_config(config_path)
            
            # Check hot reload immediately (no change)
            reloaded = manager.check_hot_reload(config_path)
            assert reloaded is None
    
    def test_hot_reload_with_change(self):
        """Test hot reload when file has changed."""
        manager = ConfigManager()
        config = manager.get_profile("balanced")
        
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            manager.save_config(config, config_path)
            
            # Load config
            original = manager.load_config(config_path)
            
            # Modify file
            import time
            time.sleep(0.1)  # Ensure timestamp changes
            modified_config = manager.get_profile("speed")
            manager.save_config(modified_config, config_path)
            
            # Check hot reload
            reloaded = manager.check_hot_reload(config_path)
            assert reloaded is not None
            assert reloaded.engine.model_size != original.engine.model_size
    
    def test_config_to_dict_and_back(self):
        """Test converting config to dict and back."""
        manager = ConfigManager()
        original_config = manager.get_profile("balanced")
        
        # Convert to dict
        config_dict = manager._config_to_dict(original_config)
        assert isinstance(config_dict, dict)
        assert "audio" in config_dict
        assert "engine" in config_dict
        
        # Convert back to config
        restored_config = manager._dict_to_config(config_dict)
        
        # Verify restoration
        assert restored_config.engine.engine_type == original_config.engine.engine_type
        assert restored_config.engine.model_size == original_config.engine.model_size
        assert restored_config.audio.sample_rate == original_config.audio.sample_rate
    
    def test_yaml_format_readable(self):
        """Test that saved YAML is human-readable."""
        manager = ConfigManager()
        config = manager.get_profile("balanced")
        
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            manager.save_config(config, config_path)
            
            # Read YAML file
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Should contain readable keys
            assert "audio:" in content
            assert "engine:" in content
            assert "sample_rate:" in content
            assert "model_size:" in content
    
    def test_custom_vocabulary_in_config(self):
        """Test custom vocabulary in configuration."""
        manager = ConfigManager()
        config = manager.get_profile("balanced")
        
        # Add custom vocabulary
        config.custom_vocabulary = ["term1", "term2", "term3"]
        
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            manager.save_config(config, config_path)
            
            # Load and verify
            loaded = manager.load_config(config_path)
            assert loaded.custom_vocabulary == ["term1", "term2", "term3"]
    
    def test_metadata_in_config(self):
        """Test metadata in configuration."""
        manager = ConfigManager()
        config = manager.get_profile("balanced")
        
        # Add metadata
        config.metadata = {
            "description": "Test configuration",
            "version": "1.0",
            "author": "Test",
        }
        
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            manager.save_config(config, config_path)
            
            # Load and verify
            loaded = manager.load_config(config_path)
            assert loaded.metadata["description"] == "Test configuration"
            assert loaded.metadata["version"] == "1.0"
    
    def test_profile_independence(self):
        """Test that getting a profile returns independent copy."""
        manager = ConfigManager()
        
        # Get profile twice
        config1 = manager.get_profile("balanced")
        config2 = manager.get_profile("balanced")
        
        # Modify one
        config1.engine.model_size = "large"
        
        # Other should not be affected
        assert config2.engine.model_size == "base"
    
    def test_all_profiles_valid(self):
        """Test that all predefined profiles are valid."""
        manager = ConfigManager()
        
        for profile_name in manager.list_profiles():
            config = manager.get_profile(profile_name)
            result = manager.validate_config(config)
            
            # All profiles should be valid
            assert result.is_valid, f"Profile '{profile_name}' is invalid: {result.errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
