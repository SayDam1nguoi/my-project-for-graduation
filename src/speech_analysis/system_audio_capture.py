"""
System Audio Capture Module

Thu âm thanh từ system (màn hình/video đang phát) thay vì microphone.
Sử dụng WASAPI loopback trên Windows.
"""

import numpy as np
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque

try:
    import pyaudiowpatch as pyaudio
    WASAPI_AVAILABLE = True
except ImportError:
    try:
        import pyaudio
        WASAPI_AVAILABLE = False
    except ImportError:
        pyaudio = None
        WASAPI_AVAILABLE = False

from .exceptions import AudioDriverError


@dataclass
class SystemAudioConfig:
    """Cấu hình cho system audio capture."""
    sample_rate: int = 16000
    channels: int = 1  # Mono cho speech recognition
    format: int = pyaudio.paInt16
    chunk_size: int = 1024
    buffer_duration: int = 30
    
    def get_max_buffer_size(self) -> int:
        """Tính max buffer size."""
        return self.sample_rate * self.buffer_duration


class SystemAudioCapture:
    """
    Thu âm thanh từ system (WASAPI loopback).
    
    Cho phép thu âm từ:
    - Video đang phát
    - Âm thanh từ trình duyệt
    - Bất kỳ âm thanh nào từ máy tính
    """
    
    def __init__(self, config: Optional[SystemAudioConfig] = None):
        """
        Khởi tạo system audio capture.
        
        Args:
            config: Cấu hình capture
            
        Raises:
            AudioDriverError: Nếu không hỗ trợ WASAPI loopback
        """
        self.config = config or SystemAudioConfig()
        
        if not WASAPI_AVAILABLE:
            raise AudioDriverError(
                "PyAudioWPatch không được cài đặt. "
                "Cài đặt với: pip install PyAudioWPatch"
            )
        
        # Initialize PyAudio
        try:
            self.pyaudio = pyaudio.PyAudio()
        except Exception as e:
            raise AudioDriverError(f"Không thể khởi tạo audio driver: {e}")
        
        # Audio stream
        self.stream: Optional[pyaudio.Stream] = None
        self.loopback_device = None
        
        # Recording state
        self._is_recording = False
        self._recording_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Circular buffer
        max_buffer_size = self.config.get_max_buffer_size()
        self._audio_buffer = deque(maxlen=max_buffer_size)
        self._buffer_lock = threading.Lock()
        
        # Queue for chunks
        self._chunk_queue: queue.Queue = queue.Queue(maxsize=100)
        
        # Full audio storage
        self._full_audio: List[np.ndarray] = []
        
    def __del__(self):
        """Cleanup."""
        self.stop_recording()
        if self.pyaudio:
            self.pyaudio.terminate()
    
    def find_loopback_device(self) -> Optional[dict]:
        """
        Tìm WASAPI loopback device.
        
        Returns:
            Device info dict hoặc None
        """
        try:
            # Get default WASAPI info
            wasapi_info = self.pyaudio.get_host_api_info_by_type(pyaudio.paWASAPI)
        except (OSError, AttributeError):
            return None
        
        # Get default output device
        default_output = wasapi_info.get('defaultOutputDevice')
        
        if default_output == -1:
            return None
        
        # Get loopback device - PyAudioWPatch specific
        try:
            # Try to get loopback device using PyAudioWPatch method
            loopback_device = self.pyaudio.get_device_info_by_index(default_output)
            
            # For PyAudioWPatch, we need to check if loopback is supported
            # The device should have isLoopbackDevice flag or maxInputChannels > 0
            if (loopback_device.get('maxInputChannels', 0) > 0 or 
                loopback_device.get('isLoopbackDevice', False)):
                return loopback_device
            
            # If not, try to find any device with "Stereo Mix" or loopback in name
            num_devices = self.pyaudio.get_device_count()
            for i in range(num_devices):
                try:
                    device = self.pyaudio.get_device_info_by_index(i)
                    name = device.get('name', '').lower()
                    if ('stereo mix' in name or 'loopback' in name or 
                        'what u hear' in name or 'wave out mix' in name):
                        if device.get('maxInputChannels', 0) > 0:
                            return device
                except Exception:
                    continue
            
            # Last resort: return default output as loopback
            return loopback_device
            
        except Exception as e:
            print(f"Debug: Error finding loopback: {e}")
            pass
        
        return None
    
    def list_loopback_devices(self) -> List[Tuple[int, str]]:
        """
        Liệt kê các loopback devices.
        
        Returns:
            List of (device_index, device_name)
        """
        devices = []
        
        try:
            wasapi_info = self.pyaudio.get_host_api_info_by_type(pyaudio.paWASAPI)
            num_devices = wasapi_info.get('deviceCount', 0)
            
            for i in range(num_devices):
                try:
                    device_info = self.pyaudio.get_device_info_by_host_api_device_index(
                        wasapi_info['index'], i
                    )
                    
                    # Check if it's a loopback device
                    if (device_info.get('maxInputChannels', 0) > 0 and
                        'loopback' in device_info.get('name', '').lower()):
                        devices.append((
                            device_info['index'],
                            device_info['name']
                        ))
                except Exception:
                    continue
        except Exception:
            pass
        
        return devices
    
    def start_recording(self) -> bool:
        """
        Bắt đầu thu âm từ system.
        
        Returns:
            True nếu thành công
            
        Raises:
            AudioDriverError: Nếu không thể bắt đầu
        """
        if self._is_recording:
            return True
        
        # Find loopback device
        self.loopback_device = self.find_loopback_device()
        
        if not self.loopback_device:
            raise AudioDriverError(
                "Không tìm thấy WASAPI loopback device. "
                "Đảm bảo có audio output device."
            )
        
        try:
            # Get device's native sample rate
            device_sample_rate = int(self.loopback_device['defaultSampleRate'])
            
            # Open stream với loopback
            self.stream = self.pyaudio.open(
                format=self.config.format,
                channels=self.loopback_device['maxInputChannels'],
                rate=device_sample_rate,
                input=True,
                input_device_index=self.loopback_device['index'],
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._audio_callback
            )
            
            # Start stream
            self.stream.start_stream()
            
            # Set state
            self._is_recording = True
            self._stop_event.clear()
            
            # Start thread
            self._recording_thread = threading.Thread(
                target=self._recording_loop,
                daemon=True
            )
            self._recording_thread.start()
            
            print(f"✓ Đang thu âm từ: {self.loopback_device['name']}")
            print(f"  Sample rate: {device_sample_rate} Hz")
            print(f"  Channels: {self.loopback_device['maxInputChannels']}")
            
            return True
            
        except Exception as e:
            raise AudioDriverError(f"Không thể bắt đầu thu âm: {e}")
    
    def stop_recording(self) -> None:
        """Dừng thu âm."""
        if not self._is_recording:
            return
        
        self._is_recording = False
        self._stop_event.set()
        
        if self._recording_thread and self._recording_thread.is_alive():
            self._recording_thread.join(timeout=1.0)
        
        if self.stream:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            finally:
                self.stream = None
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback cho PyAudio stream."""
        # Convert to numpy
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Convert stereo to mono if needed
        if self.loopback_device and self.loopback_device['maxInputChannels'] == 2:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)
        
        # Resample if needed (device rate -> target rate)
        if self.loopback_device:
            device_rate = int(self.loopback_device['defaultSampleRate'])
            if device_rate != self.config.sample_rate:
                # Simple resampling
                ratio = self.config.sample_rate / device_rate
                new_length = int(len(audio_data) * ratio)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), new_length),
                    np.arange(len(audio_data)),
                    audio_data
                ).astype(np.int16)
        
        # Add to buffer
        with self._buffer_lock:
            self._audio_buffer.extend(audio_data)
        
        self._full_audio.append(audio_data.copy())
        
        # Add to queue
        try:
            self._chunk_queue.put_nowait(audio_data.copy())
        except queue.Full:
            pass
        
        return (in_data, pyaudio.paContinue)
    
    def _recording_loop(self):
        """Recording loop."""
        while self._is_recording and not self._stop_event.is_set():
            time.sleep(0.1)
    
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Lấy audio chunk tiếp theo."""
        try:
            return self._chunk_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_full_audio(self) -> np.ndarray:
        """Lấy toàn bộ audio đã thu."""
        if not self._full_audio:
            return np.array([], dtype=np.int16)
        return np.concatenate(self._full_audio)
    
    def get_buffer_audio(self) -> np.ndarray:
        """Lấy audio từ circular buffer."""
        with self._buffer_lock:
            if not self._audio_buffer:
                return np.array([], dtype=np.int16)
            return np.array(self._audio_buffer, dtype=np.int16)
    
    def is_recording(self) -> bool:
        """Kiểm tra đang thu âm không."""
        return self._is_recording
    
    def get_recording_duration(self) -> float:
        """Lấy thời lượng đã thu (giây)."""
        audio = self.get_full_audio()
        if len(audio) == 0:
            return 0.0
        return len(audio) / self.config.sample_rate
    
    def clear_buffer(self) -> None:
        """Xóa buffer."""
        with self._buffer_lock:
            self._audio_buffer.clear()
        self._full_audio.clear()
        
        while not self._chunk_queue.empty():
            try:
                self._chunk_queue.get_nowait()
            except queue.Empty:
                break


def is_system_audio_available() -> bool:
    """
    Kiểm tra system audio capture có khả dụng không.
    
    Returns:
        True nếu có thể thu âm từ system
    """
    return WASAPI_AVAILABLE
