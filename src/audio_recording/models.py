"""Data models for audio recording system."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class RecordingState(Enum):
    """Recording state enumeration."""
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPING = "stopping"


@dataclass
class AudioDevice:
    """Represents an audio input device.
    
    Attributes:
        id: Device identifier
        name: Human-readable device name
        channels: Number of input channels
        default_sample_rate: Default sample rate for the device
        is_default: Whether this is the system default device
    """
    id: int
    name: str
    channels: int
    default_sample_rate: int
    is_default: bool


@dataclass
class RecordingInfo:
    """Metadata for a recording.
    
    Attributes:
        file_path: Full path to the recording file
        file_name: Name of the recording file
        created_at: Timestamp when recording was created
        duration: Duration of recording in seconds
        file_size: Size of file in bytes
        sample_rate: Sample rate used for recording
        channels: Number of audio channels
    """
    file_path: Path
    file_name: str
    created_at: datetime
    duration: float
    file_size: int
    sample_rate: int
    channels: int
