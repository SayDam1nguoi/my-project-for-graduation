"""Settings management for audio recording system.

Handles loading, saving, and managing audio recording settings.
Implements Requirements 3.1, 3.2, 5.1, 5.2, 5.3
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from .config import AudioRecordingConfig


logger = logging.getLogger(__name__)


class AudioRecordingSettings:
    """Manages audio recording settings persistence.
    
    Handles saving and loading settings to/from a JSON configuration file.
    Settings include:
    - Device selection
    - Sample rate
    - Bit depth
    - Output directory
    
    Implements Requirements:
    - 3.2: Save device selection
    - 5.2: Save sample rate selection
    - 5.3: Save bit depth selection
    """
    
    DEFAULT_SETTINGS_FILE = Path("config/audio_recording_settings.json")
    
    def __init__(self, settings_file: Optional[Path] = None):
        """Initialize settings manager.
        
        Args:
            settings_file: Path to settings file (uses default if None)
        """
        self.settings_file = settings_file or self.DEFAULT_SETTINGS_FILE
        logger.info(f"Settings file: {self.settings_file}")
    
    def load_settings(self) -> AudioRecordingConfig:
        """Load settings from file.
        
        Returns:
            AudioRecordingConfig with loaded settings, or default config if file doesn't exist
            
        Implements Requirements 3.2, 5.2, 5.3
        """
        if not self.settings_file.exists():
            logger.info("Settings file not found, using defaults")
            return AudioRecordingConfig()
        
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                settings_data = json.load(f)
            
            logger.info(f"Loaded settings: {settings_data}")
            
            # Create config from settings
            config = AudioRecordingConfig(
                sample_rate=settings_data.get('sample_rate', 16000),
                channels=settings_data.get('channels', 1),
                bit_depth=settings_data.get('bit_depth', 16),
                chunk_size=settings_data.get('chunk_size', 1024),
                output_dir=Path(settings_data.get('output_dir', 'recordings')),
                format=settings_data.get('format', 'wav'),
                device_id=settings_data.get('device_id')
            )
            
            return config
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in settings file: {e}")
            return AudioRecordingConfig()
        except Exception as e:
            logger.error(f"Error loading settings: {e}", exc_info=True)
            return AudioRecordingConfig()
    
    def save_settings(self, config: AudioRecordingConfig) -> bool:
        """Save settings to file.
        
        Args:
            config: AudioRecordingConfig to save
            
        Returns:
            True if saved successfully, False otherwise
            
        Implements Requirements 3.2, 5.2, 5.3
        """
        try:
            # Ensure config directory exists
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert config to dictionary
            settings_data = {
                'sample_rate': config.sample_rate,
                'channels': config.channels,
                'bit_depth': config.bit_depth,
                'chunk_size': config.chunk_size,
                'output_dir': str(config.output_dir),
                'format': config.format,
                'device_id': config.device_id
            }
            
            # Write to file
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings_data, f, indent=2)
            
            logger.info(f"Settings saved: {settings_data}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving settings: {e}", exc_info=True)
            return False
    
    def get_default_config(self) -> AudioRecordingConfig:
        """Get default configuration.
        
        Returns:
            Default AudioRecordingConfig
        """
        return AudioRecordingConfig()
    
    def reset_to_defaults(self) -> bool:
        """Reset settings to defaults.
        
        Returns:
            True if reset successfully
        """
        try:
            default_config = self.get_default_config()
            return self.save_settings(default_config)
        except Exception as e:
            logger.error(f"Error resetting settings: {e}")
            return False
