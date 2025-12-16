"""Voice Activity Detection implementations."""

import numpy as np
from typing import List, Optional
from abc import ABC, abstractmethod

from ..interfaces.audio import IVADDetector
from ..models.audio import SpeechSegment


class VADDetector(IVADDetector):
    """
    Base VAD detector with factory method for creating specific implementations.
    """
    
    @staticmethod
    def create(method: str = 'energy', **kwargs) -> 'IVADDetector':
        """
        Factory method to create VAD detector.
        
        Args:
            method: VAD method ('silero', 'webrtc', 'energy')
            **kwargs: Additional arguments for the detector
            
        Returns:
            VAD detector instance
        """
        if method == 'silero':
            return SileroVAD(**kwargs)
        elif method == 'webrtc':
            return WebRTCVAD(**kwargs)
        elif method == 'energy':
            return EnergyVAD(**kwargs)
        else:
            raise ValueError(f"Unknown VAD method: {method}")


class SileroVAD(IVADDetector):
    """
    Silero VAD - Neural network-based VAD (recommended).
    
    Provides high accuracy speech detection using a pre-trained neural network.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100
    ):
        """
        Initialize Silero VAD.
        
        Args:
            sample_rate: Sample rate (8000 or 16000)
            threshold: Speech probability threshold (0-1)
            min_speech_duration_ms: Minimum speech duration in ms
            min_silence_duration_ms: Minimum silence duration in ms
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        
        # Try to load Silero VAD model
        self.model = None
        self.available = False
        
        try:
            import torch
            # Load Silero VAD model
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.get_speech_timestamps = utils[0]
            self.available = True
        except Exception as e:
            print(f"Warning: Silero VAD not available: {e}")
            print("Falling back to Energy-based VAD")
            self.fallback = EnergyVAD(
                sample_rate=sample_rate,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms
            )
    
    def detect_speech(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> List[SpeechSegment]:
        """
        Detect speech segments in audio.
        
        Args:
            audio: Input audio data (float32, normalized)
            sample_rate: Sample rate of audio
            
        Returns:
            List of detected speech segments
        """
        if not self.available:
            return self.fallback.detect_speech(audio, sample_rate)
        
        import torch
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio = self._resample(audio, sample_rate, self.sample_rate)
            sample_rate = self.sample_rate
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            sampling_rate=sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms
        )
        
        # Convert to SpeechSegment objects
        segments = []
        for ts in speech_timestamps:
            start_sample = ts['start']
            end_sample = ts['end']
            
            segment_audio = audio[start_sample:end_sample]
            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate
            
            segment = SpeechSegment(
                audio=segment_audio,
                start_time=start_time,
                end_time=end_time,
                confidence=1.0,  # Silero doesn't provide confidence per segment
                is_speech=True
            )
            segments.append(segment)
        
        return segments
    
    def is_speech(self, audio: np.ndarray, sample_rate: int) -> bool:
        """
        Check if audio contains speech.
        
        Args:
            audio: Input audio data
            sample_rate: Sample rate of audio
            
        Returns:
            True if audio contains speech, False otherwise
        """
        probability = self.get_speech_probability(audio, sample_rate)
        return probability > self.threshold
    
    def get_speech_probability(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> float:
        """
        Get probability that audio contains speech.
        
        Args:
            audio: Input audio data
            sample_rate: Sample rate of audio
            
        Returns:
            Speech probability (0-1)
        """
        if not self.available:
            return self.fallback.get_speech_probability(audio, sample_rate)
        
        import torch
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio = self._resample(audio, sample_rate, self.sample_rate)
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Get speech probability
        speech_prob = self.model(audio_tensor, self.sample_rate).item()
        
        return speech_prob
    
    def set_sensitivity(self, sensitivity: float) -> None:
        """
        Set VAD sensitivity.
        
        Args:
            sensitivity: Sensitivity level (0-1). Higher = more sensitive
        """
        # Map sensitivity to threshold (inverse relationship)
        self.threshold = 1.0 - sensitivity
    
    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        from scipy import signal
        num_samples = int(len(audio) * target_sr / orig_sr)
        resampled = signal.resample(audio, num_samples)
        return resampled.astype(np.float32)


class WebRTCVAD(IVADDetector):
    """
    WebRTC VAD - Fast, rule-based VAD.
    
    Provides fast speech detection using WebRTC's VAD algorithm.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        aggressiveness: int = 2,
        frame_duration_ms: int = 30,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100
    ):
        """
        Initialize WebRTC VAD.
        
        Args:
            sample_rate: Sample rate (8000, 16000, 32000, or 48000)
            aggressiveness: Aggressiveness mode (0-3, 3 = most aggressive)
            frame_duration_ms: Frame duration (10, 20, or 30 ms)
            min_speech_duration_ms: Minimum speech duration in ms
            min_silence_duration_ms: Minimum silence duration in ms
        """
        self.sample_rate = sample_rate
        self.aggressiveness = aggressiveness
        self.frame_duration_ms = frame_duration_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        
        # Validate sample rate
        if sample_rate not in [8000, 16000, 32000, 48000]:
            print(f"Warning: WebRTC VAD requires sample rate in [8000, 16000, 32000, 48000]")
            print(f"Falling back to Energy-based VAD")
            self.available = False
            self.fallback = EnergyVAD(
                sample_rate=sample_rate,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms
            )
            return
        
        # Try to load WebRTC VAD
        self.vad = None
        self.available = False
        
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(aggressiveness)
            self.available = True
        except ImportError:
            print("Warning: webrtcvad not installed. Falling back to Energy-based VAD")
            self.fallback = EnergyVAD(
                sample_rate=sample_rate,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms
            )
    
    def detect_speech(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> List[SpeechSegment]:
        """
        Detect speech segments in audio.
        
        Args:
            audio: Input audio data
            sample_rate: Sample rate of audio
            
        Returns:
            List of detected speech segments
        """
        if not self.available:
            return self.fallback.detect_speech(audio, sample_rate)
        
        # Convert to int16
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio_int16 = (audio * 32768).astype(np.int16)
        else:
            audio_int16 = audio.astype(np.int16)
        
        # Calculate frame size
        frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        
        # Process frames
        is_speech_frames = []
        num_frames = len(audio_int16) // frame_size
        
        for i in range(num_frames):
            start = i * frame_size
            end = start + frame_size
            frame = audio_int16[start:end].tobytes()
            
            try:
                is_speech = self.vad.is_speech(frame, sample_rate)
                is_speech_frames.append(is_speech)
            except:
                is_speech_frames.append(False)
        
        # Find speech segments with minimum duration filtering
        segments = []
        min_speech_frames = self.min_speech_duration_ms // self.frame_duration_ms
        min_silence_frames = self.min_silence_duration_ms // self.frame_duration_ms
        
        in_speech = False
        speech_start = 0
        silence_count = 0
        
        for i, is_speech in enumerate(is_speech_frames):
            if is_speech:
                if not in_speech:
                    speech_start = i
                    in_speech = True
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count >= min_silence_frames:
                        # End of speech segment
                        speech_end = i - silence_count
                        if speech_end - speech_start >= min_speech_frames:
                            # Create segment
                            start_sample = speech_start * frame_size
                            end_sample = speech_end * frame_size
                            
                            segment_audio = audio[start_sample:end_sample]
                            start_time = start_sample / sample_rate
                            end_time = end_sample / sample_rate
                            
                            segment = SpeechSegment(
                                audio=segment_audio,
                                start_time=start_time,
                                end_time=end_time,
                                confidence=1.0,
                                is_speech=True
                            )
                            segments.append(segment)
                        
                        in_speech = False
                        silence_count = 0
        
        # Handle final segment
        if in_speech and num_frames - speech_start >= min_speech_frames:
            start_sample = speech_start * frame_size
            end_sample = len(audio)
            
            segment_audio = audio[start_sample:end_sample]
            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate
            
            segment = SpeechSegment(
                audio=segment_audio,
                start_time=start_time,
                end_time=end_time,
                confidence=1.0,
                is_speech=True
            )
            segments.append(segment)
        
        return segments
    
    def is_speech(self, audio: np.ndarray, sample_rate: int) -> bool:
        """
        Check if audio contains speech.
        
        Args:
            audio: Input audio data
            sample_rate: Sample rate of audio
            
        Returns:
            True if audio contains speech, False otherwise
        """
        if not self.available:
            return self.fallback.is_speech(audio, sample_rate)
        
        # Convert to int16
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio_int16 = (audio * 32768).astype(np.int16)
        else:
            audio_int16 = audio.astype(np.int16)
        
        # Check if any frame contains speech
        frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        num_frames = len(audio_int16) // frame_size
        
        speech_frames = 0
        for i in range(num_frames):
            start = i * frame_size
            end = start + frame_size
            frame = audio_int16[start:end].tobytes()
            
            try:
                if self.vad.is_speech(frame, sample_rate):
                    speech_frames += 1
            except:
                pass
        
        # Consider speech if > 30% of frames contain speech
        return speech_frames > num_frames * 0.3
    
    def get_speech_probability(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> float:
        """
        Get probability that audio contains speech.
        
        Args:
            audio: Input audio data
            sample_rate: Sample rate of audio
            
        Returns:
            Speech probability (0-1)
        """
        if not self.available:
            return self.fallback.get_speech_probability(audio, sample_rate)
        
        # Convert to int16
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio_int16 = (audio * 32768).astype(np.int16)
        else:
            audio_int16 = audio.astype(np.int16)
        
        # Calculate percentage of speech frames
        frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        num_frames = len(audio_int16) // frame_size
        
        if num_frames == 0:
            return 0.0
        
        speech_frames = 0
        for i in range(num_frames):
            start = i * frame_size
            end = start + frame_size
            frame = audio_int16[start:end].tobytes()
            
            try:
                if self.vad.is_speech(frame, sample_rate):
                    speech_frames += 1
            except:
                pass
        
        return speech_frames / num_frames
    
    def set_sensitivity(self, sensitivity: float) -> None:
        """
        Set VAD sensitivity.
        
        Args:
            sensitivity: Sensitivity level (0-1). Higher = more sensitive
        """
        # Map sensitivity to aggressiveness (inverse relationship)
        # sensitivity 0.0 -> aggressiveness 3 (most aggressive, least sensitive)
        # sensitivity 1.0 -> aggressiveness 0 (least aggressive, most sensitive)
        self.aggressiveness = int((1.0 - sensitivity) * 3)
        
        if self.available:
            import webrtcvad
            self.vad = webrtcvad.Vad(self.aggressiveness)


class EnergyVAD(IVADDetector):
    """
    Energy-based VAD - Simple threshold-based VAD.
    
    Provides basic speech detection using energy and zero-crossing rate.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        energy_threshold: float = 0.01,
        zcr_threshold: float = 0.1,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100
    ):
        """
        Initialize Energy-based VAD.
        
        Args:
            sample_rate: Sample rate
            frame_duration_ms: Frame duration in ms
            energy_threshold: Energy threshold (RMS)
            zcr_threshold: Zero-crossing rate threshold
            min_speech_duration_ms: Minimum speech duration in ms
            min_silence_duration_ms: Minimum silence duration in ms
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        self.energy_threshold = energy_threshold
        self.zcr_threshold = zcr_threshold
        
        self.min_speech_frames = int(min_speech_duration_ms / frame_duration_ms)
        self.min_silence_frames = int(min_silence_duration_ms / frame_duration_ms)
    
    def detect_speech(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> List[SpeechSegment]:
        """
        Detect speech segments in audio.
        
        Args:
            audio: Input audio data
            sample_rate: Sample rate of audio
            
        Returns:
            List of detected speech segments
        """
        # Process frames
        num_frames = len(audio) // self.frame_size
        is_speech_frames = []
        
        for i in range(num_frames):
            start = i * self.frame_size
            end = start + self.frame_size
            frame = audio[start:end]
            
            is_speech = self._is_speech_frame(frame)
            is_speech_frames.append(is_speech)
        
        # Find speech segments with minimum duration filtering
        segments = []
        in_speech = False
        speech_start = 0
        silence_count = 0
        
        for i, is_speech in enumerate(is_speech_frames):
            if is_speech:
                if not in_speech:
                    speech_start = i
                    in_speech = True
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count >= self.min_silence_frames:
                        # End of speech segment
                        speech_end = i - silence_count
                        if speech_end - speech_start >= self.min_speech_frames:
                            # Create segment
                            start_sample = speech_start * self.frame_size
                            end_sample = speech_end * self.frame_size
                            
                            segment_audio = audio[start_sample:end_sample]
                            start_time = start_sample / sample_rate
                            end_time = end_sample / sample_rate
                            
                            segment = SpeechSegment(
                                audio=segment_audio,
                                start_time=start_time,
                                end_time=end_time,
                                confidence=1.0,
                                is_speech=True
                            )
                            segments.append(segment)
                        
                        in_speech = False
                        silence_count = 0
        
        # Handle final segment
        if in_speech and num_frames - speech_start >= self.min_speech_frames:
            start_sample = speech_start * self.frame_size
            end_sample = len(audio)
            
            segment_audio = audio[start_sample:end_sample]
            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate
            
            segment = SpeechSegment(
                audio=segment_audio,
                start_time=start_time,
                end_time=end_time,
                confidence=1.0,
                is_speech=True
            )
            segments.append(segment)
        
        return segments
    
    def is_speech(self, audio: np.ndarray, sample_rate: int) -> bool:
        """
        Check if audio contains speech.
        
        Args:
            audio: Input audio data
            sample_rate: Sample rate of audio
            
        Returns:
            True if audio contains speech, False otherwise
        """
        # Check if any frame contains speech
        num_frames = len(audio) // self.frame_size
        speech_frames = 0
        
        for i in range(num_frames):
            start = i * self.frame_size
            end = start + self.frame_size
            frame = audio[start:end]
            
            if self._is_speech_frame(frame):
                speech_frames += 1
        
        # Consider speech if > 30% of frames contain speech
        return speech_frames > num_frames * 0.3
    
    def get_speech_probability(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> float:
        """
        Get probability that audio contains speech.
        
        Args:
            audio: Input audio data
            sample_rate: Sample rate of audio
            
        Returns:
            Speech probability (0-1)
        """
        # Calculate percentage of speech frames
        num_frames = len(audio) // self.frame_size
        
        if num_frames == 0:
            return 0.0
        
        speech_frames = 0
        for i in range(num_frames):
            start = i * self.frame_size
            end = start + self.frame_size
            frame = audio[start:end]
            
            if self._is_speech_frame(frame):
                speech_frames += 1
        
        return speech_frames / num_frames
    
    def set_sensitivity(self, sensitivity: float) -> None:
        """
        Set VAD sensitivity.
        
        Args:
            sensitivity: Sensitivity level (0-1). Higher = more sensitive
        """
        # Adjust thresholds based on sensitivity
        # Higher sensitivity = lower thresholds
        self.energy_threshold = 0.02 * (1.0 - sensitivity)
        self.zcr_threshold = 0.15 * (1.0 - sensitivity)
    
    def _is_speech_frame(self, frame: np.ndarray) -> bool:
        """
        Check if frame contains speech.
        
        Args:
            frame: Audio frame
            
        Returns:
            True if speech, False otherwise
        """
        # Calculate energy (RMS)
        energy = np.sqrt(np.mean(frame ** 2))
        
        # Calculate zero-crossing rate
        signs = np.sign(frame)
        signs[signs == 0] = -1
        zcr = np.sum(np.abs(np.diff(signs))) / (2 * len(frame))
        
        # Decision logic
        is_speech = (energy > self.energy_threshold and zcr > self.zcr_threshold)
        
        return is_speech
