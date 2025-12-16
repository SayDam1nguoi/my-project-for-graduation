"""Recording manager for managing saved audio recordings."""

import logging
import wave
from datetime import datetime
from pathlib import Path
from typing import List

from .exceptions import FileSystemError
from .models import RecordingInfo


logger = logging.getLogger(__name__)


class RecordingManager:
    """Manages saved audio recordings.
    
    Provides functionality to:
    - List all recordings in a directory
    - Get metadata for recordings
    - Delete recordings
    """
    
    def __init__(self, recordings_dir: Path):
        """Initialize RecordingManager.
        
        Args:
            recordings_dir: Directory containing recordings
        """
        self.recordings_dir = Path(recordings_dir)
        logger.info("RecordingManager initialized: recordings_dir=%s", recordings_dir)
    
    def list_recordings(self) -> List[RecordingInfo]:
        """List all recordings with metadata.
        
        Scans the recordings directory for WAV files and returns
        metadata for each recording.
        
        Returns:
            List of RecordingInfo objects for all WAV files
            
        Raises:
            FileSystemError: If recordings directory cannot be accessed
        """
        if not self.recordings_dir.exists():
            logger.debug("Recordings directory does not exist: %s", self.recordings_dir)
            return []
        
        try:
            recordings = []
            for wav_file in self.recordings_dir.glob("*.wav"):
                try:
                    info = self.get_recording_info(wav_file)
                    recordings.append(info)
                except Exception as e:
                    # Skip files that can't be read
                    logger.warning("Failed to read recording info for %s: %s", wav_file, e)
                    continue
            
            # Sort by creation time, newest first
            recordings.sort(key=lambda x: x.created_at, reverse=True)
            logger.info("Listed %d recording(s)", len(recordings))
            return recordings
            
        except Exception as e:
            logger.error("Failed to list recordings: %s", e)
            raise FileSystemError(f"Failed to list recordings: {e}")
    
    def get_recording_info(self, file_path: Path) -> RecordingInfo:
        """Get metadata for a recording.
        
        Extracts metadata from a WAV file including duration,
        sample rate, channels, file size, and creation time.
        
        Args:
            file_path: Path to the recording file
            
        Returns:
            RecordingInfo object with recording metadata
            
        Raises:
            FileSystemError: If file cannot be read or is not a valid WAV file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error("Recording file not found: %s", file_path)
            raise FileSystemError(f"Recording file not found: {file_path}")
        
        try:
            # Get file size
            file_size = file_path.stat().st_size
            
            # Get creation time from file stats
            created_timestamp = file_path.stat().st_ctime
            created_at = datetime.fromtimestamp(created_timestamp)
            
            # Extract WAV metadata
            with wave.open(str(file_path), 'rb') as wav_file:
                channels = wav_file.getnchannels()
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                # Calculate duration in seconds
                duration = n_frames / float(sample_rate) if sample_rate > 0 else 0.0
            
            logger.debug(
                "Retrieved recording info: %s (duration=%.2fs, size=%d bytes)",
                file_path.name, duration, file_size
            )
            
            return RecordingInfo(
                file_path=file_path,
                file_name=file_path.name,
                created_at=created_at,
                duration=duration,
                file_size=file_size,
                sample_rate=sample_rate,
                channels=channels
            )
            
        except wave.Error as e:
            logger.error("Invalid WAV file '%s': %s", file_path, e)
            raise FileSystemError(f"Invalid WAV file: {e}")
        except Exception as e:
            logger.error("Failed to read recording info for '%s': %s", file_path, e)
            raise FileSystemError(f"Failed to read recording info: {e}")
    
    def delete_recording(self, file_path: Path) -> bool:
        """Delete a recording file.
        
        Args:
            file_path: Path to the recording file to delete
            
        Returns:
            True if file was successfully deleted
            
        Raises:
            FileSystemError: If file cannot be deleted
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error("Cannot delete - recording file not found: %s", file_path)
            raise FileSystemError(f"Recording file not found: {file_path}")
        
        try:
            file_path.unlink()
            logger.info("Deleted recording: %s", file_path)
            return True
        except Exception as e:
            logger.error("Failed to delete recording '%s': %s", file_path, e)
            raise FileSystemError(f"Failed to delete recording: {e}")
