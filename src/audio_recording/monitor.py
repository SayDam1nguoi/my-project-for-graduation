"""Audio monitoring for recording system."""

import logging
import struct
import time
from typing import Callable, Optional
import threading


logger = logging.getLogger(__name__)


class AudioMonitor:
    """Monitors audio recording status and calculates audio levels.
    
    This class provides real-time monitoring of audio recording, including:
    - Audio level calculation using RMS (Root Mean Square)
    - Elapsed time tracking
    - Callback mechanism for GUI updates
    
    The monitor runs in a separate thread to avoid blocking the main application.
    """
    
    def __init__(self, callback: Callable[[float, float], None]):
        """Initialize the audio monitor.
        
        Args:
            callback: Function to call with updates. Should accept two arguments:
                     - audio_level (float): Audio level from 0.0 to 1.0
                     - elapsed_time (float): Elapsed time in seconds
        """
        self._callback = callback
        self._start_time: Optional[float] = None
        self._is_monitoring = False
        self._current_level = 0.0
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        logger.info("AudioMonitor initialized")
        
    def start(self) -> None:
        """Start monitoring.
        
        Initializes the start time and begins the monitoring loop.
        """
        with self._lock:
            if self._is_monitoring:
                return
            
            self._start_time = time.time()
            self._is_monitoring = True
            self._current_level = 0.0
            self._stop_event.clear()
            
            # Start monitoring thread for periodic callbacks
            self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitor_thread.start()
            
            logger.info("Audio monitoring started")
    
    def stop(self) -> None:
        """Stop monitoring.
        
        Stops the monitoring loop and resets internal state.
        """
        # First, signal the thread to stop
        with self._lock:
            if not self._is_monitoring:
                return
            self._is_monitoring = False
        
        # Signal the event to wake up the thread
        self._stop_event.set()
        
        # Wait for monitoring thread to finish (outside the lock)
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        
        # Now reset state
        with self._lock:
            self._start_time = None
            self._current_level = 0.0
        
        logger.info("Audio monitoring stopped")
    
    def update_audio_level(self, audio_data: bytes) -> float:
        """Calculate and return audio level from audio data.
        
        Uses RMS (Root Mean Square) to calculate the audio level.
        The result is normalized to a range of 0.0 to 1.0.
        
        Args:
            audio_data: Raw audio data as bytes (16-bit PCM format)
            
        Returns:
            Audio level from 0.0 (silent) to 1.0 (maximum)
        """
        if not audio_data or len(audio_data) == 0:
            with self._lock:
                self._current_level = 0.0
            return 0.0
        
        # Convert bytes to 16-bit integers
        # Format: '<h' means little-endian signed short (16-bit)
        try:
            # Calculate number of samples (2 bytes per 16-bit sample)
            num_samples = len(audio_data) // 2
            
            # Unpack audio data into integers
            samples = struct.unpack(f'<{num_samples}h', audio_data[:num_samples * 2])
            
            # Calculate RMS (Root Mean Square)
            if num_samples > 0:
                sum_squares = sum(sample ** 2 for sample in samples)
                rms = (sum_squares / num_samples) ** 0.5
                
                # Normalize to 0.0 - 1.0 range
                # 16-bit audio has max value of 32767
                level = min(1.0, rms / 32767.0)
            else:
                level = 0.0
            
            with self._lock:
                self._current_level = level
            
            return level
            
        except struct.error as e:
            # Handle unpacking errors gracefully
            logger.debug("Error unpacking audio data: %s", e)
            with self._lock:
                self._current_level = 0.0
            return 0.0
    
    def get_elapsed_time(self) -> float:
        """Get elapsed recording time in seconds.
        
        Returns:
            Elapsed time in seconds since monitoring started, or 0.0 if not monitoring
        """
        with self._lock:
            if self._start_time is None or not self._is_monitoring:
                return 0.0
            return time.time() - self._start_time
    
    def _monitoring_loop(self) -> None:
        """Internal monitoring loop that runs in a separate thread.
        
        Periodically calls the callback with current audio level and elapsed time.
        Updates at approximately 10 Hz (every 100ms) to avoid excessive overhead.
        """
        while not self._stop_event.is_set():
            with self._lock:
                if not self._is_monitoring:
                    break
                
                level = self._current_level
            
            # Get elapsed time without holding lock
            elapsed = self.get_elapsed_time()
            
            # Call the callback with current values
            try:
                self._callback(level, elapsed)
            except Exception as e:
                # Ignore callback errors to prevent monitoring from stopping
                logger.error("Error in monitoring callback: %s", e)
            
            # Wait for 100ms or until stop event is set
            self._stop_event.wait(timeout=0.1)
