"""Audio capture implementation with improved error handling."""

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    import pyaudio

import numpy as np
import threading
import queue
import time
from typing import Optional, List
from collections import deque

from ..interfaces.audio import IAudioCapture
from ..models.audio import AudioChunk, AudioDevice
from ..exceptions import (
    MicrophoneNotFoundError,
    MicrophoneInUseError,
    AudioDriverError
)


class AudioCapture(IAudioCapture):
    """
    Audio capture implementation with improved error handling.
    
    Features:
    - Circular buffer (30 seconds max)
    - Non-blocking chunk delivery via queue
    - Device enumeration and testing
    - Graceful error handling
    - Thread-safe operations
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        buffer_duration: int = 30,
        format: int = None
    ):
        """
        Initialize audio capture.
        
        Args:
            sample_rate: Sampling rate in Hz (default: 16000 for speech)
            channels: Number of audio channels (1=mono, 2=stereo)
            chunk_size: Number of samples per buffer chunk
            buffer_duration: Maximum buffer duration in seconds
            format: Audio format (pyaudio format constant)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.buffer_duration = buffer_duration
        self.format = format or pyaudio.paInt16
        
        # Initialize PyAudio
        try:
            self.pyaudio = pyaudio.PyAudio()
        except Exception as e:
            raise AudioDriverError(f"Failed to initialize audio driver: {str(e)}")
        
        # Audio stream
        self.stream: Optional[pyaudio.Stream] = None
        self.current_device: Optional[AudioDevice] = None
        
        # Recording state
        self._is_capturing = False
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Circular buffer for audio data (max buffer_duration seconds)
        max_buffer_size = sample_rate * buffer_duration
        self._audio_buffer = deque(maxlen=max_buffer_size)
        self._buffer_lock = threading.Lock()
        
        # Queue for delivering audio chunks to consumers
        self._chunk_queue: queue.Queue = queue.Queue(maxsize=100)
        
        # Timestamp tracking
        self._start_time = 0.0
        self._chunk_count = 0
    
    def __del__(self):
        """Cleanup resources on deletion."""
        self.stop_capture()
        if hasattr(self, 'pyaudio') and self.pyaudio:
            self.pyaudio.terminate()
    
    def start_capture(self, device_index: Optional[int] = None) -> bool:
        """
        Start capturing audio from microphone.
        
        Args:
            device_index: Optional device index to use. If None, uses default.
            
        Returns:
            True if capture started successfully, False otherwise.
            
        Raises:
            MicrophoneNotFoundError: If no microphone is available
            MicrophoneInUseError: If microphone is already in use
            AudioDriverError: If audio driver error occurs
        """
        if self._is_capturing:
            return True  # Already capturing
        
        # Check if microphone is available
        devices = self.list_devices()
        if not devices:
            raise MicrophoneNotFoundError("No microphone devices found")
        
        # Validate device index
        if device_index is not None:
            valid_indices = [d.index for d in devices]
            if device_index not in valid_indices:
                raise MicrophoneNotFoundError(
                    f"Device index {device_index} not found. "
                    f"Available indices: {valid_indices}"
                )
        
        # Test microphone before starting
        success, message = self._test_microphone(device_index)
        if not success:
            if "in use" in message.lower():
                raise MicrophoneInUseError(message)
            else:
                raise AudioDriverError(message)
        
        try:
            # Open audio stream
            self.stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            # Store current device info
            if device_index is not None:
                for device in devices:
                    if device.index == device_index:
                        self.current_device = device
                        break
            else:
                # Find default device
                for device in devices:
                    if device.is_default:
                        self.current_device = device
                        break
            
            # Start the stream
            self.stream.start_stream()
            
            # Set capturing state
            self._is_capturing = True
            self._stop_event.clear()
            self._start_time = time.time()
            self._chunk_count = 0
            
            # Start capture monitoring thread
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True
            )
            self._capture_thread.start()
            
            return True
            
        except OSError as e:
            if "Device unavailable" in str(e) or "Unanticipated host error" in str(e):
                raise MicrophoneInUseError(f"Microphone is in use: {str(e)}")
            else:
                raise AudioDriverError(f"Failed to open audio stream: {str(e)}")
        except Exception as e:
            raise AudioDriverError(f"Failed to start capture: {str(e)}")
    
    def stop_capture(self) -> None:
        """
        Stop capturing and cleanup resources.
        
        Ensures cleanup within 1 second as per requirements.
        """
        if not self._is_capturing:
            return
        
        # Signal stop
        self._is_capturing = False
        self._stop_event.set()
        
        # Wait for capture thread to finish (max 1 second)
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        
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
                self.current_device = None
    
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[AudioChunk]:
        """
        Get next audio chunk from queue.
        
        Args:
            timeout: Maximum time to wait for chunk in seconds.
            
        Returns:
            AudioChunk if available, None if timeout or stopped.
        """
        try:
            audio_data = self._chunk_queue.get(timeout=timeout)
            
            # Create AudioChunk with metadata
            timestamp = time.time() - self._start_time
            duration = len(audio_data) / self.sample_rate
            
            return AudioChunk(
                data=audio_data,
                sample_rate=self.sample_rate,
                timestamp=timestamp,
                duration=duration,
                metadata={
                    'chunk_index': self._chunk_count,
                    'channels': self.channels,
                    'format': 'int16'
                }
            )
        except queue.Empty:
            return None
    
    def is_capturing(self) -> bool:
        """
        Check if currently capturing.
        
        Returns:
            True if capturing, False otherwise.
        """
        return self._is_capturing
    
    def list_devices(self) -> List[AudioDevice]:
        """
        List available audio input devices.
        
        Returns:
            List of available audio devices.
        """
        devices = []
        
        try:
            # Get default input device
            try:
                default_info = self.pyaudio.get_default_input_device_info()
                default_index = default_info.get('index', -1)
            except:
                default_index = -1
            
            # Enumerate all devices
            info = self.pyaudio.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount', 0)
            
            for i in range(num_devices):
                try:
                    device_info = self.pyaudio.get_device_info_by_host_api_device_index(0, i)
                    
                    # Only include input devices
                    max_input_channels = device_info.get('maxInputChannels', 0)
                    if max_input_channels > 0:
                        device = AudioDevice(
                            index=i,
                            name=device_info.get('name', 'Unknown'),
                            max_input_channels=max_input_channels,
                            default_sample_rate=device_info.get('defaultSampleRate', 44100),
                            is_default=(i == default_index),
                            metadata={
                                'host_api': device_info.get('hostApi', 0),
                                'max_output_channels': device_info.get('maxOutputChannels', 0)
                            }
                        )
                        devices.append(device)
                except Exception:
                    continue
        except Exception as e:
            raise AudioDriverError(f"Failed to enumerate devices: {str(e)}")
        
        return devices
    
    def get_current_device(self) -> Optional[AudioDevice]:
        """
        Get currently active audio device.
        
        Returns:
            Current AudioDevice if capturing, None otherwise.
        """
        return self.current_device if self._is_capturing else None
    
    def _test_microphone(self, device_index: Optional[int] = None) -> tuple[bool, str]:
        """
        Test if a microphone is working.
        
        Args:
            device_index: Device index to test (None = default device)
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Try to open a test stream
            test_stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                start=False
            )
            
            # Try to read a small amount of data
            test_stream.start_stream()
            data = test_stream.read(self.chunk_size, exception_on_overflow=False)
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
        
        # Try to put in chunk queue (non-blocking)
        try:
            self._chunk_queue.put_nowait(audio_data.copy())
            self._chunk_count += 1
        except queue.Full:
            # Queue is full, skip this chunk
            pass
        
        return (in_data, pyaudio.paContinue)
    
    def _capture_loop(self):
        """
        Main capture loop (runs in separate thread).
        
        This thread monitors the capture state and handles cleanup.
        """
        while self._is_capturing and not self._stop_event.is_set():
            time.sleep(0.1)  # Check every 100ms
    
    def get_buffer_audio(self) -> np.ndarray:
        """
        Get audio from the circular buffer (last buffer_duration seconds).
        
        Returns:
            Audio data from buffer as numpy array
        """
        with self._buffer_lock:
            if not self._audio_buffer:
                return np.array([], dtype=np.int16)
            return np.array(self._audio_buffer, dtype=np.int16)
    
    def clear_buffer(self) -> None:
        """Clear the audio buffer and chunk queue."""
        with self._buffer_lock:
            self._audio_buffer.clear()
        
        # Clear the chunk queue
        while not self._chunk_queue.empty():
            try:
                self._chunk_queue.get_nowait()
            except queue.Empty:
                break
