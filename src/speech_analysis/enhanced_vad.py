"""
Enhanced Voice Activity Detection (VAD) Module

Phát hiện phần có giọng nói trong audio với độ chính xác cao.
Hỗ trợ nhiều phương pháp: Silero VAD, WebRTC VAD, Energy-based VAD.
"""

import numpy as np
from typing import List, Tuple, Optional
import time
import logging

# Import existing VAD detector as fallback
from .vad_detector import VADDetector, WebRTCVAD


class EnhancedVADDetector:
    """
    Enhanced Voice Activity Detector với nhiều phương pháp phát hiện.
    
    Hỗ trợ:
    - Silero VAD (neural network-based, high accuracy)
    - WebRTC VAD (rule-based, fast)
    - Energy-based VAD (fallback)
    
    Tự động fallback nếu phương pháp chính không khả dụng.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        method: str = "silero",  # "silero", "webrtc", "energy"
        frame_duration_ms: int = 30,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30
    ):
        """
        Khởi tạo Enhanced VAD Detector.
        
        Args:
            sample_rate: Tần số lấy mẫu (Hz)
            method: Phương pháp VAD ("silero", "webrtc", "energy")
            frame_duration_ms: Độ dài frame (ms)
            min_speech_duration_ms: Thời gian tối thiểu của speech segment
            min_silence_duration_ms: Thời gian tối thiểu của silence để tách segment
            speech_pad_ms: Padding thêm vào đầu/cuối speech segment
        """
        self.sample_rate = sample_rate
        self.method = method
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize VAD model based on method
        self._init_vad_model()
    
    def _init_vad_model(self):
        """Initialize VAD model based on selected method."""
        self.vad_model = None
        self.actual_method = self.method
        
        if self.method == "silero":
            try:
                import torch
                # Load Silero VAD model
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                self.vad_model = model
                self.get_speech_timestamps = utils[0]
                self.logger.info("Silero VAD initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load Silero VAD: {e}. Falling back to WebRTC VAD.")
                self.actual_method = "webrtc"
                self._init_webrtc_vad()
        
        elif self.method == "webrtc":
            self._init_webrtc_vad()
        
        elif self.method == "energy":
            self._init_energy_vad()
        
        else:
            raise ValueError(f"Unknown VAD method: {self.method}")
    
    def _init_webrtc_vad(self):
        """Initialize WebRTC VAD."""
        try:
            self.vad_model = WebRTCVAD(
                sample_rate=self.sample_rate,
                aggressiveness=2
            )
            if self.vad_model.available:
                self.logger.info("WebRTC VAD initialized successfully")
            else:
                self.logger.warning("WebRTC VAD not available. Falling back to energy-based VAD.")
                self.actual_method = "energy"
                self._init_energy_vad()
        except Exception as e:
            self.logger.warning(f"Failed to initialize WebRTC VAD: {e}. Falling back to energy-based VAD.")
            self.actual_method = "energy"
            self._init_energy_vad()
    
    def _init_energy_vad(self):
        """Initialize energy-based VAD."""
        self.vad_model = VADDetector(
            sample_rate=self.sample_rate,
            frame_duration_ms=self.frame_duration_ms,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms
        )
        self.logger.info("Energy-based VAD initialized successfully")
    
    def detect_speech(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """
        Phát hiện speech segments trong audio.
        
        Args:
            audio: Audio data (float32, normalized -1 to 1)
            
        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        if self.actual_method == "silero":
            return self._detect_speech_silero(audio)
        elif self.actual_method == "webrtc":
            return self._detect_speech_webrtc(audio)
        else:  # energy
            return self._detect_speech_energy(audio)
    
    def _detect_speech_silero(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Detect speech using Silero VAD."""
        import torch
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.vad_model,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms
        )
        
        # Convert to seconds
        segments = [
            (ts['start'] / self.sample_rate, ts['end'] / self.sample_rate)
            for ts in speech_timestamps
        ]
        
        return segments
    
    def _detect_speech_webrtc(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Detect speech using WebRTC VAD."""
        _, segments_samples = self.vad_model.process_audio(audio)
        
        # Convert to seconds
        segments = [
            (start / self.sample_rate, end / self.sample_rate)
            for start, end in segments_samples
        ]
        
        return segments
    
    def _detect_speech_energy(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Detect speech using energy-based VAD."""
        return self.vad_model.get_speech_segments(audio)
    
    def extract_speech_segments(
        self,
        audio: np.ndarray,
        return_timestamps: bool = False
    ) -> List[np.ndarray] | List[Tuple[np.ndarray, float, float]]:
        """
        Trích xuất các speech segments từ audio.
        
        Args:
            audio: Audio data (float32, normalized)
            return_timestamps: Nếu True, trả về (audio_segment, start_time, end_time)
            
        Returns:
            List of audio segments hoặc list of (segment, start, end) tuples
        """
        speech_timestamps = self.detect_speech(audio)
        
        segments = []
        for start_time, end_time in speech_timestamps:
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # Ensure indices are within bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            segment = audio[start_sample:end_sample]
            
            if return_timestamps:
                segments.append((segment, start_time, end_time))
            else:
                segments.append(segment)
        
        return segments
    
    def get_speech_ratio(self, audio: np.ndarray) -> float:
        """
        Tính tỷ lệ speech trong audio.
        
        Args:
            audio: Audio data
            
        Returns:
            Tỷ lệ speech (0.0 - 1.0)
        """
        speech_timestamps = self.detect_speech(audio)
        
        if not speech_timestamps:
            return 0.0
        
        # Calculate total speech duration
        total_speech_duration = sum(
            end - start for start, end in speech_timestamps
        )
        
        # Calculate total audio duration
        total_audio_duration = len(audio) / self.sample_rate
        
        if total_audio_duration == 0:
            return 0.0
        
        speech_ratio = total_speech_duration / total_audio_duration
        
        # Clamp to [0, 1]
        return min(1.0, max(0.0, speech_ratio))
    
    def process_audio_with_latency_tracking(
        self,
        audio: np.ndarray
    ) -> Tuple[List[Tuple[float, float]], float]:
        """
        Xử lý audio và theo dõi latency.
        
        Args:
            audio: Audio data
            
        Returns:
            Tuple of (speech_segments, processing_latency_ms)
        """
        start_time = time.time()
        segments = self.detect_speech(audio)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        
        return segments, latency_ms
    
    def get_method_info(self) -> dict:
        """
        Lấy thông tin về phương pháp VAD đang sử dụng.
        
        Returns:
            Dictionary chứa thông tin về VAD method
        """
        return {
            'requested_method': self.method,
            'actual_method': self.actual_method,
            'sample_rate': self.sample_rate,
            'frame_duration_ms': self.frame_duration_ms,
            'min_speech_duration_ms': self.min_speech_duration_ms,
            'min_silence_duration_ms': self.min_silence_duration_ms
        }


def create_enhanced_vad(
    sample_rate: int = 16000,
    method: str = "silero",
    **kwargs
) -> EnhancedVADDetector:
    """
    Factory function để tạo Enhanced VAD detector.
    
    Args:
        sample_rate: Sample rate
        method: VAD method ("silero", "webrtc", "energy")
        **kwargs: Additional arguments
        
    Returns:
        EnhancedVADDetector instance
    """
    return EnhancedVADDetector(
        sample_rate=sample_rate,
        method=method,
        **kwargs
    )
