"""
Video Audio Player Module

Phát audio từ video file đồng thời với việc hiển thị video.
"""

import numpy as np
import threading
import queue
import time
from typing import Optional

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    import pyaudio

try:
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    try:
        import moviepy.editor as mp
        VideoFileClip = mp.VideoFileClip
        MOVIEPY_AVAILABLE = True
    except ImportError:
        MOVIEPY_AVAILABLE = False
        VideoFileClip = None


class VideoAudioPlayer:
    """
    Phát audio từ video file.
    
    Sử dụng MoviePy để extract audio và PyAudio để phát.
    """
    
    def __init__(self, video_path: str):
        """
        Khởi tạo player.
        
        Args:
            video_path: Đường dẫn đến video file
        """
        self.video_path = video_path
        
        if not MOVIEPY_AVAILABLE:
            raise ImportError(
                "MoviePy không được cài đặt. "
                "Cài đặt với: pip install moviepy"
            )
        
        # Load video
        try:
            self.video = VideoFileClip(video_path)
            self.audio = self.video.audio
        except Exception as e:
            raise RuntimeError(f"Không thể load video: {e}")
        
        if self.audio is None:
            raise RuntimeError("Video không có audio track")
        
        # Audio properties
        self.sample_rate = self.audio.fps
        self.n_channels = self.audio.nchannels
        self.duration = self.audio.duration
        
        # PyAudio
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        
        # Playback state
        self.is_playing = False
        self.current_time = 0.0
        self.playback_thread = None
        self._stop_event = threading.Event()
        
        # Audio buffer
        self.audio_queue = queue.Queue(maxsize=10)
        
    def __del__(self):
        """Cleanup."""
        self.stop()
        if self.video:
            self.video.close()
        if self.pyaudio:
            self.pyaudio.terminate()
    
    def start(self, start_time: float = 0.0):
        """
        Bắt đầu phát audio.
        
        Args:
            start_time: Thời điểm bắt đầu (giây)
        """
        if self.is_playing:
            return
        
        self.current_time = start_time
        self.is_playing = True
        self._stop_event.clear()
        
        # Open audio stream
        self.stream = self.pyaudio.open(
            format=pyaudio.paFloat32,
            channels=self.n_channels,
            rate=int(self.sample_rate),
            output=True,
            frames_per_buffer=1024
        )
        
        # Start playback thread
        self.playback_thread = threading.Thread(
            target=self._playback_loop,
            daemon=True
        )
        self.playback_thread.start()
    
    def stop(self):
        """Dừng phát audio."""
        if not self.is_playing:
            return
        
        self.is_playing = False
        self._stop_event.set()
        
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
    
    def pause(self):
        """Tạm dừng."""
        self.is_playing = False
    
    def resume(self):
        """Tiếp tục."""
        if not self.stream:
            self.start(self.current_time)
        else:
            self.is_playing = True
    
    def seek(self, time: float):
        """
        Nhảy đến thời điểm.
        
        Args:
            time: Thời điểm (giây)
        """
        was_playing = self.is_playing
        self.stop()
        
        if was_playing:
            self.start(time)
        else:
            self.current_time = time
    
    def _playback_loop(self):
        """Loop phát audio."""
        chunk_duration = 0.1  # 100ms chunks
        
        while self.is_playing and not self._stop_event.is_set():
            if self.current_time >= self.duration:
                # Hết video
                break
            
            try:
                # Get audio chunk
                end_time = min(self.current_time + chunk_duration, self.duration)
                
                # Extract audio data using subclipped method
                # MoviePy 2.x uses subclipped() instead of subclip()
                try:
                    # Try new API first (MoviePy 2.x)
                    audio_clip = self.audio.subclipped(self.current_time, end_time)
                except AttributeError:
                    # Fallback to old API (MoviePy 1.x)
                    audio_clip = self.audio.subclip(self.current_time, end_time)
                
                audio_data = audio_clip.to_soundarray(fps=int(self.sample_rate))
                
                # Convert to float32
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                # Normalize to [-1, 1] if needed
                if np.abs(audio_data).max() > 1.0:
                    audio_data = audio_data / 32768.0
                
                # Play
                if self.stream and self.is_playing:
                    self.stream.write(audio_data.tobytes())
                
                # Update time
                self.current_time = end_time
                
            except Exception as e:
                print(f"Audio playback error: {e}")
                time.sleep(0.01)
    
    def get_current_time(self) -> float:
        """Lấy thời điểm hiện tại."""
        return self.current_time
    
    def get_duration(self) -> float:
        """Lấy tổng thời lượng."""
        return self.duration
    
    def get_audio_data(self, start_time: float, end_time: float) -> Optional[np.ndarray]:
        """
        Lấy audio data cho transcription.
        
        Args:
            start_time: Thời điểm bắt đầu (giây)
            end_time: Thời điểm kết thúc (giây)
            
        Returns:
            Audio data as numpy array (mono, 16kHz, int16)
        """
        try:
            # Extract audio using subclipped method
            # MoviePy 2.x uses subclipped() instead of subclip()
            try:
                # Try new API first (MoviePy 2.x)
                audio_clip = self.audio.subclipped(start_time, end_time)
            except AttributeError:
                # Fallback to old API (MoviePy 1.x)
                audio_clip = self.audio.subclip(start_time, end_time)
            
            audio_data = audio_clip.to_soundarray(fps=16000)  # Resample to 16kHz for STT
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Convert to int16
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                audio_data = (audio_data * 32768.0).astype(np.int16)
            
            return audio_data
            
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None


def check_moviepy_installed() -> bool:
    """Kiểm tra MoviePy có được cài đặt không."""
    return MOVIEPY_AVAILABLE
