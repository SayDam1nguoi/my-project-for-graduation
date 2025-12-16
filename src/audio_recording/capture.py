"""Audio capture component for recording from microphone."""

import logging
from typing import List, Optional

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    pyaudio = None

from .config import AudioRecordingConfig
from .exceptions import DeviceError, AudioFormatError
from .models import AudioDevice


logger = logging.getLogger(__name__)


class AudioCapture:
    """Captures audio from microphone using PyAudio.
    
    This class handles microphone input and audio capture, including:
    - Device enumeration and selection
    - Audio stream management
    - Chunk-based audio reading
    
    Attributes:
        config: Audio configuration settings
        _pyaudio: PyAudio instance
        _stream: Active audio stream (None when not capturing)
        _current_device_id: Currently selected device ID
    """
    
    def __init__(self, config: AudioRecordingConfig):
        """Initialize AudioCapture.
        
        Args:
            config: Audio configuration
            
        Raises:
            DeviceError: If PyAudio is not available
        """
        if not PYAUDIO_AVAILABLE:
            raise DeviceError(
                "PyAudio is not installed. Please install it with: pip install pyaudio"
            )
        
        self.config = config
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._stream = None
        self._current_device_id: Optional[int] = config.device_id
        
        logger.info("AudioCapture initialized with config: %s", config)
    
    def _ensure_pyaudio(self):
        """Ensure PyAudio instance is created."""
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()
            logger.debug("PyAudio instance created")
    
    def get_devices(self) -> List[AudioDevice]:
        """Get list of available audio input devices.
        
        Returns:
            List of AudioDevice objects representing available microphones
            
        Raises:
            DeviceError: If no input devices are found
        """
        self._ensure_pyaudio()
        
        devices = []
        device_count = self._pyaudio.get_device_count()
        default_input_index = None
        
        try:
            default_input_info = self._pyaudio.get_default_input_device_info()
            default_input_index = default_input_info['index']
        except (OSError, IOError) as e:
            logger.warning("No default input device found: %s", e)
        
        for i in range(device_count):
            try:
                info = self._pyaudio.get_device_info_by_index(i)
                
                # Only include input devices (max_input_channels > 0)
                if info['maxInputChannels'] > 0:
                    device = AudioDevice(
                        id=i,
                        name=info['name'],
                        channels=info['maxInputChannels'],
                        default_sample_rate=int(info['defaultSampleRate']),
                        is_default=(i == default_input_index)
                    )
                    devices.append(device)
                    logger.debug("Found input device: %s", device)
            except Exception as e:
                logger.warning("Error getting device info for index %d: %s", i, e)
                continue
        
        if not devices:
            raise DeviceError("No audio input devices found")
        
        logger.info("Found %d input device(s)", len(devices))
        return devices
    
    def start_capture(self, device_id: Optional[int] = None) -> bool:
        """Start capturing audio from microphone.
        
        Args:
            device_id: Optional device ID to use. If None, uses configured device
                      or system default.
                      
        Returns:
            True if capture started successfully
            
        Raises:
            DeviceError: If device cannot be accessed or is invalid
            AudioFormatError: If audio format is not supported by device
            StateError: If already capturing
        """
        if self._stream is not None:
            from .exceptions import StateError
            raise StateError("Already capturing audio")
        
        self._ensure_pyaudio()
        
        # Determine which device to use
        if device_id is not None:
            self._current_device_id = device_id
        elif self._current_device_id is None:
            # Use system default
            try:
                default_info = self._pyaudio.get_default_input_device_info()
                self._current_device_id = default_info['index']
                logger.info("Using default input device: %s", default_info['name'])
            except (OSError, IOError) as e:
                raise DeviceError(f"No default input device available: {e}")
        
        # Get audio format for PyAudio
        audio_format = self._get_pyaudio_format()
        
        try:
            # Open audio stream
            self._stream = self._pyaudio.open(
                format=audio_format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self._current_device_id,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=None  # We'll use blocking read
            )
            
            logger.info(
                "Audio capture started: device=%d, rate=%d, channels=%d, format=%d",
                self._current_device_id,
                self.config.sample_rate,
                self.config.channels,
                audio_format
            )
            return True
            
        except (OSError, IOError) as e:
            error_msg = str(e).lower()
            if "invalid device" in error_msg or "device unavailable" in error_msg:
                raise DeviceError(f"Device {self._current_device_id} is not available: {e}")
            elif "permission" in error_msg or "access" in error_msg:
                raise DeviceError(
                    f"Permission denied for microphone access. "
                    f"Please grant microphone permissions in system settings."
                )
            elif "sample rate" in error_msg or "format" in error_msg:
                raise AudioFormatError(
                    f"Audio format not supported by device: "
                    f"rate={self.config.sample_rate}, "
                    f"channels={self.config.channels}, "
                    f"bit_depth={self.config.bit_depth}"
                )
            else:
                raise DeviceError(f"Failed to start audio capture: {e}")
    
    def stop_capture(self) -> None:
        """Stop capturing audio and close stream.
        
        This method is safe to call even if not currently capturing.
        """
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
                logger.info("Audio capture stopped")
            except Exception as e:
                logger.error("Error stopping audio stream: %s", e)
            finally:
                self._stream = None
    
    def read_chunk(self) -> Optional[bytes]:
        """Read one chunk of audio data.
        
        Returns:
            Audio data as bytes, or None if not capturing or error occurred
            
        Raises:
            DeviceError: If device is disconnected or error reading from device
        """
        if self._stream is None:
            logger.warning("Attempted to read chunk when not capturing")
            return None
        
        try:
            # Read audio data (blocking)
            audio_data = self._stream.read(
                self.config.chunk_size,
                exception_on_overflow=False  # Don't raise exception on buffer overflow
            )
            return audio_data
            
        except (OSError, IOError) as e:
            logger.error("Error reading audio chunk: %s", e)
            raise DeviceError(f"Device error while reading audio: {e}")
    
    def _get_pyaudio_format(self) -> int:
        """Get PyAudio format constant for configured bit depth.
        
        Returns:
            PyAudio format constant
            
        Raises:
            AudioFormatError: If bit depth is not supported
        """
        if self.config.bit_depth == 16:
            return pyaudio.paInt16
        elif self.config.bit_depth == 24:
            return pyaudio.paInt24
        else:
            raise AudioFormatError(
                f"Unsupported bit depth: {self.config.bit_depth}. "
                f"Supported values: 16, 24"
            )
    
    def cleanup(self) -> None:
        """Clean up resources.
        
        Stops capture if active and terminates PyAudio instance.
        Should be called when done with AudioCapture.
        """
        self.stop_capture()
        
        if self._pyaudio is not None:
            try:
                self._pyaudio.terminate()
                logger.info("PyAudio terminated")
            except Exception as e:
                logger.error("Error terminating PyAudio: %s", e)
            finally:
                self._pyaudio = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
        return False
