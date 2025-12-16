"""
Voice Activity Detection (VAD) Module

Phát hiện phần có giọng nói trong audio để tối ưu hóa STT.
Sử dụng WebRTC VAD hoặc energy-based detection.
"""

import numpy as np
from typing import Tuple, List
import struct


class VADDetector:
    """
    Voice Activity Detector sử dụng energy-based method.
    
    Phát hiện phần có giọng nói dựa trên:
    - Energy level (RMS)
    - Zero Crossing Rate (ZCR)
    - Spectral features
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        energy_threshold: float = 0.01,
        zcr_threshold: float = 0.1,
        min_speech_duration_ms: int = 300,
        min_silence_duration_ms: int = 500
    ):
        """
        Khởi tạo VAD detector.
        
        Args:
            sample_rate: Tần số lấy mẫu (Hz)
            frame_duration_ms: Độ dài frame (ms)
            energy_threshold: Ngưỡng năng lượng
            zcr_threshold: Ngưỡng zero crossing rate
            min_speech_duration_ms: Thời gian tối thiểu của speech
            min_silence_duration_ms: Thời gian tối thiểu của silence
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        self.energy_threshold = energy_threshold
        self.zcr_threshold = zcr_threshold
        
        self.min_speech_frames = int(min_speech_duration_ms / frame_duration_ms)
        self.min_silence_frames = int(min_silence_duration_ms / frame_duration_ms)
        
        # State tracking
        self.is_speech = False
        self.speech_frames = 0
        self.silence_frames = 0
    
    def calculate_energy(self, frame: np.ndarray) -> float:
        """Tính năng lượng (RMS) của frame."""
        return float(np.sqrt(np.mean(frame ** 2)))
    
    def calculate_zcr(self, frame: np.ndarray) -> float:
        """Tính Zero Crossing Rate."""
        signs = np.sign(frame)
        signs[signs == 0] = -1
        zcr = np.sum(np.abs(np.diff(signs))) / (2 * len(frame))
        return float(zcr)
    
    def is_speech_frame(self, frame: np.ndarray) -> bool:
        """
        Kiểm tra frame có phải speech không.
        
        Args:
            frame: Audio frame (float32, normalized)
            
        Returns:
            True nếu là speech, False nếu là silence
        """
        # Tính features
        energy = self.calculate_energy(frame)
        zcr = self.calculate_zcr(frame)
        
        # Decision logic
        is_speech = (
            energy > self.energy_threshold and
            zcr > self.zcr_threshold
        )
        
        return is_speech
    
    def process_audio(self, audio: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Xử lý audio và trả về phần có speech.
        
        Args:
            audio: Audio data (float32, normalized)
            
        Returns:
            - Filtered audio (chỉ phần có speech)
            - List of speech segments [(start_sample, end_sample), ...]
        """
        # Chia thành frames
        num_frames = len(audio) // self.frame_size
        speech_segments = []
        speech_mask = np.zeros(len(audio), dtype=bool)
        
        current_segment_start = None
        
        for i in range(num_frames):
            start = i * self.frame_size
            end = start + self.frame_size
            frame = audio[start:end]
            
            is_speech = self.is_speech_frame(frame)
            
            # State machine
            if is_speech:
                self.speech_frames += 1
                self.silence_frames = 0
                
                # Bắt đầu segment mới
                if not self.is_speech and self.speech_frames >= self.min_speech_frames:
                    self.is_speech = True
                    current_segment_start = start
                
                # Đánh dấu frame là speech
                if self.is_speech:
                    speech_mask[start:end] = True
            else:
                self.silence_frames += 1
                self.speech_frames = 0
                
                # Kết thúc segment
                if self.is_speech and self.silence_frames >= self.min_silence_frames:
                    self.is_speech = False
                    if current_segment_start is not None:
                        speech_segments.append((current_segment_start, start))
                        current_segment_start = None
        
        # Kết thúc segment cuối nếu còn
        if self.is_speech and current_segment_start is not None:
            speech_segments.append((current_segment_start, len(audio)))
        
        # Extract speech audio
        speech_audio = audio[speech_mask]
        
        return speech_audio, speech_segments
    
    def get_speech_segments(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """
        Lấy danh sách các segment có speech (theo giây).
        
        Args:
            audio: Audio data
            
        Returns:
            List of (start_time, end_time) in seconds
        """
        _, segments_samples = self.process_audio(audio)
        
        segments_time = [
            (start / self.sample_rate, end / self.sample_rate)
            for start, end in segments_samples
        ]
        
        return segments_time
    
    def reset(self):
        """Reset state của detector."""
        self.is_speech = False
        self.speech_frames = 0
        self.silence_frames = 0


class WebRTCVAD:
    """
    Wrapper cho WebRTC VAD (nếu có cài đặt).
    
    WebRTC VAD chính xác hơn energy-based method.
    """
    
    def __init__(self, sample_rate: int = 16000, aggressiveness: int = 2):
        """
        Khởi tạo WebRTC VAD.
        
        Args:
            sample_rate: Tần số lấy mẫu (8000, 16000, 32000, 48000)
            aggressiveness: Độ aggressive (0-3, 3 = most aggressive)
        """
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(aggressiveness)
            self.sample_rate = sample_rate
            self.available = True
        except ImportError:
            print("Warning: webrtcvad not installed. Using energy-based VAD.")
            self.available = False
            self.fallback = VADDetector(sample_rate=sample_rate)
    
    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        """
        Kiểm tra frame có phải speech không.
        
        Args:
            frame: Audio frame (bytes, int16)
            sample_rate: Sample rate
            
        Returns:
            True nếu là speech
        """
        if not self.available:
            # Fallback to energy-based
            audio_float = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
            return self.fallback.is_speech_frame(audio_float)
        
        return self.vad.is_speech(frame, sample_rate)
    
    def process_audio(self, audio: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Xử lý audio với WebRTC VAD."""
        if not self.available:
            return self.fallback.process_audio(audio)
        
        # Convert to int16 bytes
        audio_int16 = (audio * 32768).astype(np.int16)
        
        # Process in 30ms frames
        frame_duration_ms = 30
        frame_size = int(self.sample_rate * frame_duration_ms / 1000)
        num_frames = len(audio_int16) // frame_size
        
        speech_mask = np.zeros(len(audio), dtype=bool)
        speech_segments = []
        current_segment_start = None
        
        for i in range(num_frames):
            start = i * frame_size
            end = start + frame_size
            frame = audio_int16[start:end].tobytes()
            
            is_speech = self.is_speech(frame, self.sample_rate)
            
            if is_speech:
                speech_mask[start:end] = True
                if current_segment_start is None:
                    current_segment_start = start
            else:
                if current_segment_start is not None:
                    speech_segments.append((current_segment_start, start))
                    current_segment_start = None
        
        # Kết thúc segment cuối
        if current_segment_start is not None:
            speech_segments.append((current_segment_start, len(audio)))
        
        speech_audio = audio[speech_mask]
        
        return speech_audio, speech_segments


def create_vad(
    sample_rate: int = 16000,
    use_webrtc: bool = True,
    **kwargs
) -> VADDetector:
    """
    Factory function để tạo VAD detector.
    
    Args:
        sample_rate: Sample rate
        use_webrtc: Sử dụng WebRTC VAD nếu có
        **kwargs: Additional arguments
        
    Returns:
        VAD detector instance
    """
    if use_webrtc:
        try:
            return WebRTCVAD(sample_rate=sample_rate)
        except:
            pass
    
    return VADDetector(sample_rate=sample_rate, **kwargs)
