"""Audio recording system for capturing and managing audio recordings."""

from .config import AudioRecordingConfig
from .models import AudioDevice, RecordingInfo, RecordingState
from .exceptions import (
    AudioRecordingError,
    DeviceError,
    FileSystemError,
    AudioFormatError,
    StateError,
)
from .capture import AudioCapture
from .recorder import AudioRecorder
from .monitor import AudioMonitor
from .coordinator import AudioRecordingCoordinator
from .manager import RecordingManager
from .settings import AudioRecordingSettings
from .utils import (
    check_disk_space,
    get_permission_instructions,
    format_file_size,
    estimate_recording_size,
    get_error_message_with_solution,
)
from .logging_config import (
    configure_logging,
    get_logger,
    set_level,
    disable_logging,
    enable_logging,
)
from .transcriber import AudioTranscriber, get_transcriber

__all__ = [
    "AudioRecordingConfig",
    "AudioDevice",
    "RecordingInfo",
    "RecordingState",
    "AudioRecordingError",
    "DeviceError",
    "FileSystemError",
    "AudioFormatError",
    "StateError",
    "AudioCapture",
    "AudioRecorder",
    "AudioMonitor",
    "AudioRecordingCoordinator",
    "RecordingManager",
    "AudioRecordingSettings",
    "check_disk_space",
    "get_permission_instructions",
    "format_file_size",
    "estimate_recording_size",
    "get_error_message_with_solution",
    "configure_logging",
    "get_logger",
    "set_level",
    "disable_logging",
    "enable_logging",
    "AudioTranscriber",
    "get_transcriber",
]
