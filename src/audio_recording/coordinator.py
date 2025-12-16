"""Audio recording coordinator for managing the recording workflow."""

import logging
import threading
from pathlib import Path
from typing import Callable, List, Optional

from .capture import AudioCapture
from .config import AudioRecordingConfig
from .exceptions import StateError, DeviceError, FileSystemError, AudioRecordingError
from .models import AudioDevice, RecordingState
from .monitor import AudioMonitor
from .recorder import AudioRecorder
from .utils import check_disk_space, estimate_recording_size


logger = logging.getLogger(__name__)


class AudioRecordingCoordinator:
    """Coordinates audio recording components.
    
    This class manages the complete recording workflow by coordinating:
    - AudioCapture: Microphone input
    - AudioRecorder: WAV file writing
    - AudioMonitor: Real-time monitoring and callbacks
    
    The coordinator implements a state machine with states:
    - IDLE: Not recording
    - RECORDING: Actively recording
    - PAUSED: Recording paused
    - STOPPING: In process of stopping
    
    Threading:
    - Audio capture runs in a separate thread to avoid blocking the GUI
    - The monitor provides periodic callbacks for UI updates
    """
    
    def __init__(self, config: AudioRecordingConfig, gui_callback: Optional[Callable[[float, float], None]] = None):
        """Initialize the audio recording coordinator.
        
        Args:
            config: Recording configuration
            gui_callback: Optional callback for GUI updates (level, elapsed_time)
        """
        self.config = config
        self._gui_callback = gui_callback
        
        # Initialize components
        self._capture = AudioCapture(config)
        self._recorder = AudioRecorder(config.output_dir, config)
        self._monitor = AudioMonitor(self._monitor_callback)
        
        # State management
        self._state = RecordingState.IDLE
        self._state_lock = threading.Lock()
        
        # Recording thread
        self._recording_thread: Optional[threading.Thread] = None
        self._stop_recording_event = threading.Event()
        self._pause_event = threading.Event()
        
        # Error tracking
        self._last_error: Optional[Exception] = None
        
        logger.info("AudioRecordingCoordinator initialized")
    
    def start_recording(self) -> bool:
        """Start recording audio.
        
        Transitions from IDLE to RECORDING state and begins audio capture.
        
        Returns:
            True if recording started successfully
            
        Raises:
            StateError: If not in IDLE state
            DeviceError: If microphone cannot be accessed
            FileSystemError: If output file cannot be created or insufficient disk space
        """
        with self._state_lock:
            if self._state != RecordingState.IDLE:
                logger.warning("Cannot start recording from state: %s", self._state.value)
                raise StateError(f"Cannot start recording from state: {self._state.value}")
            
            logger.debug("State transition: IDLE -> RECORDING")
            self._state = RecordingState.RECORDING
        
        try:
            # Check disk space before starting (estimate 10 minutes of recording)
            estimated_size_mb = estimate_recording_size(
                duration_seconds=600,  # 10 minutes
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
                bit_depth=self.config.bit_depth
            ) // (1024 * 1024)
            
            # Require at least estimated size + 50 MB buffer
            required_mb = max(estimated_size_mb + 50, 100)
            
            has_space, available_mb = check_disk_space(self.config.output_dir, required_mb)
            
            if not has_space:
                raise FileSystemError(
                    f"Insufficient disk space. Available: {available_mb} MB, "
                    f"Required: {required_mb} MB. Please free up space or choose a different location."
                )
            
            logger.info(f"Disk space check passed: {available_mb} MB available")
            
            # Create output file
            file_path = self._recorder.create_file()
            logger.info(f"Created recording file: {file_path}")
            
            # Start audio capture
            self._capture.start_capture()
            logger.info("Audio capture started")
            
            # Start monitoring
            self._monitor.start()
            logger.info("Audio monitoring started")
            
            # Start recording thread
            self._stop_recording_event.clear()
            self._pause_event.clear()
            self._recording_thread = threading.Thread(
                target=self._recording_loop,
                daemon=True,
                name="AudioRecordingThread"
            )
            self._recording_thread.start()
            logger.info("Recording thread started")
            
            return True
            
        except Exception as e:
            # Rollback state on error
            logger.error(f"Error starting recording: {e}", exc_info=True)
            self._cleanup_on_error()
            with self._state_lock:
                self._state = RecordingState.IDLE
            raise
    
    def stop_recording(self) -> str:
        """Stop recording and return file path.
        
        Transitions from RECORDING or PAUSED to STOPPING, then to IDLE.
        Finalizes the recording file and returns its path.
        
        Returns:
            Full path to the saved recording file
            
        Raises:
            StateError: If not in RECORDING or PAUSED state
            FileSystemError: If file cannot be finalized
        """
        with self._state_lock:
            if self._state not in (RecordingState.RECORDING, RecordingState.PAUSED):
                logger.warning("Cannot stop recording from state: %s", self._state.value)
                raise StateError(f"Cannot stop recording from state: {self._state.value}")
            
            logger.debug("State transition: %s -> STOPPING", self._state.value)
            self._state = RecordingState.STOPPING
        
        try:
            # Signal recording thread to stop
            self._stop_recording_event.set()
            
            # If paused, unpause to allow thread to exit
            if self._pause_event.is_set():
                self._pause_event.clear()
            
            # Wait for recording thread to finish
            if self._recording_thread and self._recording_thread.is_alive():
                self._recording_thread.join(timeout=5.0)
                if self._recording_thread.is_alive():
                    logger.warning("Recording thread did not stop within timeout")
            
            # Stop monitoring
            self._monitor.stop()
            logger.info("Audio monitoring stopped")
            
            # Stop capture
            self._capture.stop_capture()
            logger.info("Audio capture stopped")
            
            # Finalize recording file
            file_path = self._recorder.finalize_file()
            logger.info(f"Recording finalized: {file_path}")
            
            # Transition to IDLE
            with self._state_lock:
                logger.debug("State transition: STOPPING -> IDLE")
                self._state = RecordingState.IDLE
            
            return str(file_path.absolute())
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}", exc_info=True)
            self._cleanup_on_error()
            with self._state_lock:
                self._state = RecordingState.IDLE
            raise
    
    def pause_recording(self) -> bool:
        """Pause recording.
        
        Transitions from RECORDING to PAUSED state.
        Audio capture continues but data is not written to file.
        
        Returns:
            True if paused successfully
            
        Raises:
            StateError: If not in RECORDING state
        """
        with self._state_lock:
            if self._state != RecordingState.RECORDING:
                logger.warning("Cannot pause from state: %s", self._state.value)
                raise StateError(f"Cannot pause from state: {self._state.value}")
            
            logger.debug("State transition: RECORDING -> PAUSED")
            self._state = RecordingState.PAUSED
        
        # Signal the recording thread to pause
        self._pause_event.set()
        logger.info("Recording paused")
        
        return True
    
    def resume_recording(self) -> bool:
        """Resume recording.
        
        Transitions from PAUSED to RECORDING state.
        Continues writing to the same file.
        
        Returns:
            True if resumed successfully
            
        Raises:
            StateError: If not in PAUSED state
        """
        with self._state_lock:
            if self._state != RecordingState.PAUSED:
                logger.warning("Cannot resume from state: %s", self._state.value)
                raise StateError(f"Cannot resume from state: {self._state.value}")
            
            logger.debug("State transition: PAUSED -> RECORDING")
            self._state = RecordingState.RECORDING
        
        # Signal the recording thread to resume
        self._pause_event.clear()
        logger.info("Recording resumed")
        
        return True
    
    def get_available_devices(self) -> List[AudioDevice]:
        """Get list of available microphone devices.
        
        Returns:
            List of AudioDevice objects
            
        Raises:
            DeviceError: If no devices are found
        """
        return self._capture.get_devices()
    
    def set_device(self, device_id: int) -> bool:
        """Set active microphone device.
        
        Can only be called when not recording.
        
        Args:
            device_id: Device ID to use for recording
            
        Returns:
            True if device was set successfully
            
        Raises:
            StateError: If currently recording
        """
        with self._state_lock:
            if self._state != RecordingState.IDLE:
                logger.warning("Cannot change device while in state: %s", self._state.value)
                raise StateError("Cannot change device while recording")
        
        # Update config with new device
        self.config.device_id = device_id
        logger.info(f"Device set to: {device_id}")
        
        return True
    
    def get_state(self) -> RecordingState:
        """Get current recording state.
        
        Returns:
            Current RecordingState
        """
        with self._state_lock:
            return self._state
    
    def cleanup(self) -> None:
        """Clean up resources.
        
        Stops recording if active and releases all resources.
        Should be called when done with the coordinator.
        """
        # Stop recording if active
        if self._state in (RecordingState.RECORDING, RecordingState.PAUSED):
            try:
                self.stop_recording()
            except Exception as e:
                logger.error(f"Error stopping recording during cleanup: {e}")
        
        # Clean up capture resources
        try:
            self._capture.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up capture: {e}")
        
        logger.info("AudioRecordingCoordinator cleaned up")
    
    def _recording_loop(self) -> None:
        """Internal recording loop that runs in a separate thread.
        
        Continuously reads audio chunks from capture and writes to recorder.
        Handles pause/resume and stop signals.
        """
        logger.info("Recording loop started")
        
        try:
            while not self._stop_recording_event.is_set():
                # Check if paused
                if self._pause_event.is_set():
                    # Wait while paused (check every 100ms)
                    self._stop_recording_event.wait(timeout=0.1)
                    continue
                
                # Read audio chunk
                try:
                    audio_data = self._capture.read_chunk()
                    
                    if audio_data:
                        # Write to file (only if not paused)
                        if not self._pause_event.is_set():
                            self._recorder.write_chunk(audio_data)
                        
                        # Update audio level for monitoring
                        self._monitor.update_audio_level(audio_data)
                    
                except DeviceError as e:
                    logger.error(f"Device error in recording loop: {e}")
                    self._last_error = e
                    break
                    
                except FileSystemError as e:
                    logger.error(f"File system error in recording loop: {e}")
                    self._last_error = e
                    break
                    
                except Exception as e:
                    logger.error(f"Unexpected error in recording loop: {e}", exc_info=True)
                    self._last_error = e
                    break
        
        finally:
            logger.info("Recording loop ended")
    
    def _monitor_callback(self, level: float, elapsed_time: float) -> None:
        """Internal callback for monitor updates.
        
        Forwards updates to GUI callback if provided.
        
        Args:
            level: Audio level (0.0 to 1.0)
            elapsed_time: Elapsed time in seconds
        """
        if self._gui_callback:
            try:
                self._gui_callback(level, elapsed_time)
            except Exception as e:
                logger.error(f"Error in GUI callback: {e}")
    
    def _cleanup_on_error(self) -> None:
        """Clean up resources after an error.
        
        Ensures all resources are properly released even if an error occurs.
        """
        logger.info("Cleaning up after error")
        
        # Stop recording thread
        self._stop_recording_event.set()
        if self._pause_event.is_set():
            self._pause_event.clear()
        
        if self._recording_thread and self._recording_thread.is_alive():
            self._recording_thread.join(timeout=2.0)
        
        # Stop monitoring
        try:
            self._monitor.stop()
        except Exception as e:
            logger.error(f"Error stopping monitor during cleanup: {e}")
        
        # Stop capture
        try:
            self._capture.stop_capture()
        except Exception as e:
            logger.error(f"Error stopping capture during cleanup: {e}")
        
        # Try to finalize file if recording
        if self._recorder.is_recording():
            try:
                self._recorder.finalize_file()
            except Exception as e:
                logger.error(f"Error finalizing file during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
        return False
