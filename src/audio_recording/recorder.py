"""Audio recorder component for writing WAV files."""

import logging
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import AudioRecordingConfig
from .exceptions import FileSystemError


logger = logging.getLogger(__name__)


class AudioRecorder:
    """Records audio to WAV file.
    
    Handles WAV file creation with timestamp-based naming,
    streaming audio data writing, and file finalization.
    """
    
    def __init__(self, output_dir: Path, config: AudioRecordingConfig):
        """Initialize AudioRecorder.
        
        Args:
            output_dir: Directory to save recordings
            config: Audio configuration (sample_rate, channels, bit_depth)
        """
        self.output_dir = Path(output_dir)
        self.config = config
        self._wav_file: Optional[wave.Wave_write] = None
        self._current_file_path: Optional[Path] = None
        self._frames_written = 0
        
        logger.info(
            "AudioRecorder initialized: output_dir=%s, sample_rate=%d, channels=%d, bit_depth=%d",
            output_dir, config.sample_rate, config.channels, config.bit_depth
        )
    
    def create_file(self) -> Path:
        """Create new WAV file with timestamp name.
        
        Creates the output directory if it doesn't exist.
        Generates filename in format: recording_YYYYMMDD_HHMMSS.wav
        
        Returns:
            Path to the created WAV file
            
        Raises:
            FileSystemError: If directory cannot be created or file cannot be opened
        """
        # Create output directory if it doesn't exist
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Recordings directory ready: %s", self.output_dir)
        except (OSError, PermissionError) as e:
            logger.error("Failed to create recordings directory '%s': %s", self.output_dir, e)
            raise FileSystemError(
                f"Cannot create recordings directory '{self.output_dir}': {e}"
            ) from e
        
        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        self._current_file_path = self.output_dir / filename
        
        # Open WAV file for writing
        try:
            self._wav_file = wave.open(str(self._current_file_path), 'wb')
            
            # Set WAV file parameters
            self._wav_file.setnchannels(self.config.channels)
            self._wav_file.setsampwidth(self.config.bit_depth // 8)  # Convert bits to bytes
            self._wav_file.setframerate(self.config.sample_rate)
            
            self._frames_written = 0
            
            logger.info("Created WAV file: %s", self._current_file_path)
            
        except (OSError, PermissionError, wave.Error) as e:
            logger.error("Failed to create WAV file '%s': %s", self._current_file_path, e)
            raise FileSystemError(
                f"Cannot create WAV file '{self._current_file_path}': {e}"
            ) from e
        
        return self._current_file_path
    
    def write_chunk(self, audio_data: bytes) -> None:
        """Write audio chunk to file.
        
        Args:
            audio_data: Raw audio data bytes to write
            
        Raises:
            FileSystemError: If file write fails (disk full, permission denied)
            StateError: If no file is currently open
        """
        if self._wav_file is None:
            from .exceptions import StateError
            raise StateError("No WAV file is open. Call create_file() first.")
        
        try:
            self._wav_file.writeframes(audio_data)
            self._frames_written += len(audio_data) // (self.config.bit_depth // 8)
            logger.debug("Wrote %d bytes to WAV file", len(audio_data))
        except (OSError, wave.Error) as e:
            logger.error("Failed to write to WAV file '%s': %s", self._current_file_path, e)
            raise FileSystemError(
                f"Cannot write to WAV file '{self._current_file_path}': {e}"
            ) from e
    
    def finalize_file(self) -> Path:
        """Finalize and close WAV file.
        
        Closes the WAV file properly, ensuring all data is written
        and file headers are updated.
        
        Returns:
            Path to the finalized WAV file
            
        Raises:
            StateError: If no file is currently open
            FileSystemError: If file cannot be closed properly
        """
        if self._wav_file is None:
            from .exceptions import StateError
            raise StateError("No WAV file is open. Nothing to finalize.")
        
        try:
            self._wav_file.close()
            file_path = self._current_file_path
            
            logger.info("Finalized WAV file: %s (%d frames written)", file_path, self._frames_written)
            
            # Reset state
            self._wav_file = None
            self._current_file_path = None
            self._frames_written = 0
            
            return file_path
            
        except (OSError, wave.Error) as e:
            logger.error("Failed to close WAV file '%s': %s", self._current_file_path, e)
            raise FileSystemError(
                f"Cannot close WAV file '{self._current_file_path}': {e}"
            ) from e
    
    def get_current_file_size(self) -> int:
        """Get current file size in bytes.
        
        Returns:
            File size in bytes, or 0 if no file is open or file doesn't exist
        """
        if self._current_file_path is None or not self._current_file_path.exists():
            return 0
        
        try:
            return self._current_file_path.stat().st_size
        except OSError:
            return 0
    
    def is_recording(self) -> bool:
        """Check if currently recording to a file.
        
        Returns:
            True if a WAV file is open for writing, False otherwise
        """
        return self._wav_file is not None
    
    def get_current_file_path(self) -> Optional[Path]:
        """Get path to current recording file.
        
        Returns:
            Path to current file, or None if no file is open
        """
        return self._current_file_path
