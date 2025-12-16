"""Audio preprocessing pipeline implementation."""

import numpy as np
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
import scipy.signal as signal

from ..interfaces.audio import IAudioPreprocessor


@dataclass
class PreprocessingStep:
    """Represents a preprocessing step in the pipeline."""
    
    name: str
    func: Callable
    enabled: bool = True
    kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


class AudioPreprocessor(IAudioPreprocessor):
    """
    Audio preprocessing pipeline with modular steps.
    
    Preprocessing steps:
    1. Noise Reduction: Spectral subtraction, Wiener filtering
    2. Normalization: Volume normalization to optimal range
    3. Pre-emphasis: Boost high frequencies (speech)
    4. Resampling: Convert to target sample rate if needed
    """
    
    def __init__(self, target_sample_rate: int = 16000):
        """
        Initialize audio preprocessor.
        
        Args:
            target_sample_rate: Target sample rate for output audio
        """
        self.target_sample_rate = target_sample_rate
        self.steps: Dict[str, PreprocessingStep] = {}
        
        # Register default preprocessing steps
        self._register_default_steps()
    
    def _register_default_steps(self):
        """Register default preprocessing steps."""
        # Noise reduction
        self.add_step(
            'noise_reduction',
            self._noise_reduction,
            noise_floor=0.01
        )
        
        # Normalization
        self.add_step(
            'normalization',
            self._normalize,
            target_rms=0.1
        )
        
        # Pre-emphasis
        self.add_step(
            'pre_emphasis',
            self._pre_emphasis,
            coef=0.97
        )
    
    def preprocess(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply preprocessing pipeline to audio.
        
        Args:
            audio: Input audio data (float32, normalized to [-1, 1])
            sample_rate: Sample rate of audio
            
        Returns:
            Preprocessed audio data
        """
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            else:
                audio = audio.astype(np.float32)
        
        # Resample if needed (do this first)
        if sample_rate != self.target_sample_rate:
            audio = self._resample(audio, sample_rate, self.target_sample_rate)
            sample_rate = self.target_sample_rate
        
        # Apply each enabled preprocessing step
        for step in self.steps.values():
            if step.enabled:
                try:
                    audio = step.func(audio, sample_rate, **step.kwargs)
                except Exception as e:
                    # Log error but continue with other steps
                    print(f"Warning: Preprocessing step '{step.name}' failed: {e}")
                    continue
        
        return audio
    
    def add_step(
        self,
        step_name: str,
        step_func: Callable,
        **kwargs
    ) -> None:
        """
        Add preprocessing step to pipeline.
        
        Args:
            step_name: Name of the preprocessing step
            step_func: Function to apply for this step
            **kwargs: Additional arguments for the step function
        """
        step = PreprocessingStep(
            name=step_name,
            func=step_func,
            enabled=True,
            kwargs=kwargs
        )
        self.steps[step_name] = step
    
    def remove_step(self, step_name: str) -> None:
        """
        Remove preprocessing step from pipeline.
        
        Args:
            step_name: Name of the step to remove
        """
        if step_name in self.steps:
            del self.steps[step_name]
    
    def get_pipeline_info(self) -> List[Dict[str, Any]]:
        """
        Get information about current pipeline.
        
        Returns:
            List of dictionaries containing step information
        """
        info = []
        for step in self.steps.values():
            info.append({
                'name': step.name,
                'enabled': step.enabled,
                'kwargs': step.kwargs
            })
        return info
    
    def enable_step(self, step_name: str) -> None:
        """
        Enable a preprocessing step.
        
        Args:
            step_name: Name of the step to enable
        """
        if step_name in self.steps:
            self.steps[step_name].enabled = True
    
    def disable_step(self, step_name: str) -> None:
        """
        Disable a preprocessing step.
        
        Args:
            step_name: Name of the step to disable
        """
        if step_name in self.steps:
            self.steps[step_name].enabled = False
    
    # Preprocessing step implementations
    
    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Input audio
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio
        """
        if orig_sr == target_sr:
            return audio
        
        # Calculate resampling ratio
        num_samples = int(len(audio) * target_sr / orig_sr)
        
        # Use scipy's resample for high-quality resampling
        resampled = signal.resample(audio, num_samples)
        
        return resampled.astype(np.float32)
    
    def _noise_reduction(
        self,
        audio: np.ndarray,
        sample_rate: int,
        noise_floor: float = 0.01
    ) -> np.ndarray:
        """
        Apply noise reduction using spectral subtraction.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            noise_floor: Minimum noise floor threshold
            
        Returns:
            Noise-reduced audio
        """
        # Simple spectral subtraction
        # Estimate noise from first 0.5 seconds (assumed to be silence)
        noise_duration = min(0.5, len(audio) / sample_rate)
        noise_samples = int(noise_duration * sample_rate)
        
        if noise_samples < 100:
            # Not enough samples for noise estimation
            return audio
        
        # Estimate noise spectrum
        noise_segment = audio[:noise_samples]
        noise_fft = np.fft.rfft(noise_segment)
        noise_magnitude = np.abs(noise_fft)
        
        # Process audio in frames
        frame_size = 2048
        hop_size = frame_size // 2
        
        output = np.zeros_like(audio)
        window = np.hanning(frame_size)
        
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size] * window
            
            # FFT
            frame_fft = np.fft.rfft(frame)
            frame_magnitude = np.abs(frame_fft)
            frame_phase = np.angle(frame_fft)
            
            # Spectral subtraction
            # Interpolate noise magnitude to match frame size
            if len(noise_magnitude) != len(frame_magnitude):
                noise_mag_interp = np.interp(
                    np.linspace(0, 1, len(frame_magnitude)),
                    np.linspace(0, 1, len(noise_magnitude)),
                    noise_magnitude
                )
            else:
                noise_mag_interp = noise_magnitude
            
            clean_magnitude = np.maximum(
                frame_magnitude - noise_mag_interp,
                noise_floor * frame_magnitude
            )
            
            # Reconstruct signal
            clean_fft = clean_magnitude * np.exp(1j * frame_phase)
            clean_frame = np.fft.irfft(clean_fft, n=frame_size)
            
            # Overlap-add
            output[i:i + frame_size] += clean_frame * window
        
        # Normalize
        output = output / np.max(np.abs(output) + 1e-8)
        
        return output
    
    def _normalize(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_rms: float = 0.1
    ) -> np.ndarray:
        """
        Normalize audio volume to target RMS level.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            target_rms: Target RMS level
            
        Returns:
            Normalized audio
        """
        # Calculate current RMS
        current_rms = np.sqrt(np.mean(audio ** 2))
        
        if current_rms < 1e-8:
            # Audio is silent
            return audio
        
        # Calculate scaling factor
        scale = target_rms / current_rms
        
        # Apply scaling with clipping prevention
        normalized = audio * scale
        
        # Clip to [-1, 1] range
        normalized = np.clip(normalized, -1.0, 1.0)
        
        return normalized
    
    def _pre_emphasis(
        self,
        audio: np.ndarray,
        sample_rate: int,
        coef: float = 0.97
    ) -> np.ndarray:
        """
        Apply pre-emphasis filter to boost high frequencies.
        
        This helps with speech recognition by emphasizing
        consonants and reducing low-frequency noise.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            coef: Pre-emphasis coefficient (typically 0.95-0.97)
            
        Returns:
            Pre-emphasized audio
        """
        # Apply first-order high-pass filter: y[n] = x[n] - coef * x[n-1]
        emphasized = np.append(audio[0], audio[1:] - coef * audio[:-1])
        
        return emphasized
    
    def _spectral_cleaning(
        self,
        audio: np.ndarray,
        sample_rate: int,
        freq_range: tuple = (300, 3400)
    ) -> np.ndarray:
        """
        Apply spectral cleaning to remove frequencies outside speech range.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            freq_range: Frequency range to keep (Hz)
            
        Returns:
            Spectrally cleaned audio
        """
        # Design bandpass filter
        nyquist = sample_rate / 2
        low = freq_range[0] / nyquist
        high = freq_range[1] / nyquist
        
        # Ensure frequencies are in valid range
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        # Create Butterworth bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered = signal.filtfilt(b, a, audio)
        
        return filtered
