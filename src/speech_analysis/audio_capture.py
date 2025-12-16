"""
Audio Capture Module

This module provides audio capture functionality from microphone devices.
It includes:
- AudioConfig: Configuration dataclass for audio capture settings
- AudioCapture: Main class for capturing audio from microphone
- Circular buffer management for audio data
- Thread-safe audio processing
- Error handling for microphone access issues
"""

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    import pyaudio

import numpy as np
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque

from .exceptions import (
    MicrophoneNotFoundError,
    MicrophoneInUseError,
    AudioDriverError
)


@dataclass
class AudioConfig:
    """
    Configuration for audio capture.
    
    Attributes:
        sample_rate: Sampling rate in Hz (default: 16000 for speech)
        channels: Number of audio channels (1=mono, 2=stereo)
        format: Audio format (pyaudio format constant)
        chunk_size: Number of samples per buffer chunk
        buffer_duration: Maximum buffer duration in seconds
        device_index: Specific device index to use (None = default device)
    """
    sample_rate: int = 16000
    channels: int = 1
    format: int = pyaudio.paInt16
    chunk_size: int = 1024
    buffer_duration: int = 30
    device_index: Optional[int] = None
    
    def get_bytes_per_sample(self) -> int:
        """Get number of bytes per sample based on format."""
        if self.format == pyaudio.paInt16:
            return 2
        elif self.format == pyaudio.paInt32:
            return 4
        elif self.format == pyaudio.paFloat32:
            return 4
        else:
            return 2  # Default to 16-bit
    
    def get_max_buffer_size(self) -> int:
        """Calculate maximum buffer size in samples."""
        return self.sample_rate * self.buffer_duration


class AudioCapture:
    """
    Audio capture class for recording from microphone.
    
    This class handles:
    - Audio recording from microphone
    - Circular buffer management (max 30 seconds)
    - Thread-safe audio chunk delivery
    - Microphone device management
    - Error handling
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize audio capture.
        
        Args:
            config: AudioConfig object with capture settings
        
        Raises:
            AudioDriverError: If PyAudio initialization fails
        """
        self.config = config or AudioConfig()
        
        # Initialize PyAudio
        try:
            self.pyaudio = pyaudio.PyAudio()
        except Exception as e:
            raise AudioDriverError(f"Failed to initialize audio driver: {str(e)}")
        
        # Audio stream
        self.stream: Optional[pyaudio.Stream] = None
        
        # Recording state
        self._is_recording = False
        self._recording_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Circular buffer for audio data (max 30 seconds)
        max_buffer_size = self.config.get_max_buffer_size()
        self._audio_buffer = deque(maxlen=max_buffer_size)
        self._buffer_lock = threading.Lock()
        
        # Queue for delivering audio chunks to consumers
        self._chunk_queue: queue.Queue = queue.Queue(maxsize=100)
        
        # Full audio storage
        self._full_audio: List[np.ndarray] = []
        
    def __del__(self):
        """Cleanup resources on deletion."""
        self.stop_recording()
        if self.pyaudio:
            self.pyaudio.terminate()
    
    def list_available_devices(self) -> List[Tuple[int, str, int]]:
        """
        List all available audio input devices.
        
        Returns:
            List of tuples: (device_index, device_name, max_input_channels)
        """
        devices = []
        info = self.pyaudio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount', 0)
        
        for i in range(num_devices):
            try:
                device_info = self.pyaudio.get_device_info_by_host_api_device_index(0, i)
                # Only include input devices
                if device_info.get('maxInputChannels', 0) > 0:
                    devices.append((
                        i,
                        device_info.get('name', 'Unknown'),
                        device_info.get('maxInputChannels', 0)
                    ))
            except Exception:
                continue
        
        return devices
    
    def test_microphone(self, device_index: Optional[int] = None) -> Tuple[bool, str]:
        """
        Test if a microphone is working.
        
        Args:
            device_index: Device index to test (None = default device)
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        test_device = device_index if device_index is not None else self.config.device_index
        
        try:
            # Try to open a test stream
            test_stream = self.pyaudio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=test_device,
                frames_per_buffer=self.config.chunk_size,
                start=False
            )
            
            # Try to read a small amount of data
            test_stream.start_stream()
            data = test_stream.read(self.config.chunk_size, exception_on_overflow=False)
            test_stream.stop_stream()
            test_stream.close()
            
            if data:
                return True, "Microphone is working properly"
            else:
                return False, "Microphone returned no data"
                
        except OSError as e:
            if "Invalid number of channels" in str(e):
                return False, f"Invalid channel configuration: {str(e)}"
            elif "Device unavailable" in str(e) or "Unanticipated host error" in str(e):
                return False, "Microphone is in use by another application"
            else:
                return False, f"Microphone access error: {str(e)}"
        except Exception as e:
            return False, f"Microphone test failed: {str(e)}"
    
    def start_recording(self) -> bool:
        """
        Start recording audio from microphone.
        
        Returns:
            True if recording started successfully, False otherwise
        
        Raises:
            MicrophoneNotFoundError: If no microphone is available
            MicrophoneInUseError: If microphone is already in use
            AudioDriverError: If audio driver error occurs
        """
        if self._is_recording:
            return True  # Already recording
        
        # Check if microphone is available
        devices = self.list_available_devices()
        if not devices:
            raise MicrophoneNotFoundError("No microphone devices found")
        
        # Test microphone before starting
        success, message = self.test_microphone()
        if not success:
            if "in use" in message.lower():
                raise MicrophoneInUseError(message)
            else:
                raise AudioDriverError(message)
        
        try:
            # Open audio stream
            self.stream = self.pyaudio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self.config.device_index,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._audio_callback
            )
            
            # Start the stream
            self.stream.start_stream()
            
            # Set recording state
            self._is_recording = True
            self._stop_event.clear()
            
            # Start recording thread
            self._recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
            self._recording_thread.start()
            
            return True
            
        except OSError as e:
            if "Device unavailable" in str(e) or "Unanticipated host error" in str(e):
                raise MicrophoneInUseError(f"Microphone is in use: {str(e)}")
            else:
                raise AudioDriverError(f"Failed to open audio stream: {str(e)}")
        except Exception as e:
            raise AudioDriverError(f"Failed to start recording: {str(e)}")
    
    def stop_recording(self) -> None:
        """
        Stop recording audio and release microphone resources.
        
        This method ensures cleanup within 1 second as per requirements.
        """
        if not self._is_recording:
            return
        
        # Signal stop
        self._is_recording = False
        self._stop_event.set()
        
        # Wait for recording thread to finish (max 1 second)
        if self._recording_thread and self._recording_thread.is_alive():
            self._recording_thread.join(timeout=1.0)
        
        # Stop and close stream
        if self.stream:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass  # Ignore errors during cleanup
            finally:
                self.stream = None
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback function for PyAudio stream.
        
        This is called by PyAudio in a separate thread.
        """
        if status:
            # Log status flags if needed (overflow, etc.)
            pass
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Add to circular buffer (thread-safe)
        with self._buffer_lock:
            self._audio_buffer.extend(audio_data)
        
        # Add to full audio storage
        self._full_audio.append(audio_data.copy())
        
        # Try to put in chunk queue (non-blocking)
        try:
            self._chunk_queue.put_nowait(audio_data.copy())
        except queue.Full:
            # Queue is full, skip this chunk
            pass
        
        return (in_data, pyaudio.paContinue)
    
    def _recording_loop(self):
        """
        Main recording loop (runs in separate thread).
        
        This thread monitors the recording state and handles cleanup.
        """
        while self._is_recording and not self._stop_event.is_set():
            time.sleep(0.1)  # Check every 100ms
    
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get the next audio chunk from the queue.
        
        Args:
            timeout: Maximum time to wait for a chunk (seconds)
        
        Returns:
            Audio chunk as numpy array, or None if timeout
        """
        try:
            return self._chunk_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_full_audio(self) -> np.ndarray:
        """
        Get all recorded audio as a single numpy array.
        
        Returns:
            Complete audio data as numpy array
        """
        if not self._full_audio:
            return np.array([], dtype=np.int16)
        
        return np.concatenate(self._full_audio)
    
    def get_buffer_audio(self) -> np.ndarray:
        """
        Get audio from the circular buffer (last 30 seconds).
        
        Returns:
            Audio data from buffer as numpy array
        """
        with self._buffer_lock:
            if not self._audio_buffer:
                return np.array([], dtype=np.int16)
            return np.array(self._audio_buffer, dtype=np.int16)
    
    def is_recording(self) -> bool:
        """
        Check if currently recording.
        
        Returns:
            True if recording, False otherwise
        """
        return self._is_recording
    
    def get_recording_duration(self) -> float:
        """
        Get duration of recorded audio in seconds.
        
        Returns:
            Duration in seconds
        """
        audio = self.get_full_audio()
        if len(audio) == 0:
            return 0.0
        return len(audio) / self.config.sample_rate
    
    def clear_buffer(self) -> None:
        """Clear the audio buffer and full audio storage."""
        with self._buffer_lock:
            self._audio_buffer.clear()
        self._full_audio.clear()
        
        # Clear the chunk queue
        while not self._chunk_queue.empty():
            try:
                self._chunk_queue.get_nowait()
            except queue.Empty:
                break
