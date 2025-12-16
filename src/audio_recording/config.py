"""Configuration for audio recording system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AudioRecordingConfig:
    """Configuration for audio recording.
    
    Attributes:
        sample_rate: Sample rate in Hz (16000, 44100, 48000)
        channels: Number of audio channels (1 for mono, 2 for stereo)
        bit_depth: Bit depth in bits (16 or 24)
        chunk_size: Number of samples per chunk
        output_dir: Directory to save recordings
        format: Audio file format (currently only 'wav' supported)
        device_id: Optional specific device ID to use (None for default)
    """
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16
    chunk_size: int = 1024
    output_dir: Path = field(default_factory=lambda: Path("recordings"))
    format: str = "wav"
    device_id: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate sample rate
        valid_sample_rates = [16000, 44100, 48000]
        if self.sample_rate not in valid_sample_rates:
            raise ValueError(
                f"Invalid sample_rate: {self.sample_rate}. "
                f"Must be one of {valid_sample_rates}"
            )
        
        # Validate channels
        if self.channels not in [1, 2]:
            raise ValueError(
                f"Invalid channels: {self.channels}. Must be 1 (mono) or 2 (stereo)"
            )
        
        # Validate bit depth
        valid_bit_depths = [16, 24]
        if self.bit_depth not in valid_bit_depths:
            raise ValueError(
                f"Invalid bit_depth: {self.bit_depth}. "
                f"Must be one of {valid_bit_depths}"
            )
        
        # Validate chunk size
        if self.chunk_size <= 0:
            raise ValueError(
                f"Invalid chunk_size: {self.chunk_size}. Must be positive"
            )
        
        # Validate format
        if self.format != "wav":
            raise ValueError(
                f"Invalid format: {self.format}. Currently only 'wav' is supported"
            )
        
        # Convert output_dir to Path if it's a string
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
