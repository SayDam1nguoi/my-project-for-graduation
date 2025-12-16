"""
Laptop Audio Enhancer Module

This module provides audio enhancement specifically optimized for laptop microphones.
It applies a series of filters and processing steps to improve audio quality for
speech-to-text recognition on CPU-only systems.

The enhancement pipeline includes:
1. High-pass filter (remove low-frequency noise)
2. Noise gate (suppress quiet background noise)
3. Volume normalization (ensure consistent levels)
4. Spectral noise reduction (reduce background noise)
5. Soft limiting (prevent clipping)
"""

import numpy as np
from scipy import signal
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LaptopAudioEnhancer:
    """
    Audio enhancement pipeline for laptop microphones.
    
    This class applies multiple audio processing techniques to improve
    the quality of audio captured from laptop microphones, which typically
    have lower quality and higher noise levels than dedicated microphones.
    
    Attributes:
        sample_rate: Audio sample rate in Hz (default: 16000)
        highpass_cutoff: Cutoff frequency for high-pass filter in Hz
        noise_gate_threshold: Amplitude threshold for noise gate
        target_rms: Target RMS level for volume normalization
        max_gain: Maximum gain multiplier for volume boost
        noise_reduction_strength: Strength of spectral noise reduction (0-1)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        highpass_cutoff: float = 85.0,
        noise_gate_threshold: float = 0.015,
        target_rms: float = 0.125,
        max_gain: float = 1.3,
        noise_reduction_strength: float = 0.5
    ):
        """
        Initialize the audio enhancer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            highpass_cutoff: Cutoff frequency for high-pass filter (80-100 Hz)
            noise_gate_threshold: Threshold below which audio is suppressed (0.01-0.02)
            target_rms: Target RMS level for normalization (0.1-0.15)
            max_gain: Maximum gain multiplier (typically 1.3)
            noise_reduction_strength: Spectral subtraction strength (0-1)
        """
        self.sample_rate = sample_rate
        self.highpass_cutoff = highpass_cutoff
        self.noise_gate_threshold = noise_gate_threshold
        self.target_rms = target_rms
        self.max_gain = max_gain
        self.noise_reduction_strength = noise_reduction_strength
        
        # Pre-compute filter coefficients for efficiency
        self._init_highpass_filter()
        
        logger.info(
            f"LaptopAudioEnhancer initialized: "
            f"sample_rate={sample_rate}Hz, "
            f"highpass={highpass_cutoff}Hz, "
            f"noise_gate={noise_gate_threshold}, "
            f"target_rms={target_rms}"
        )
    
    def _init_highpass_filter(self):
        """Initialize high-pass filter coefficients."""
        # Design Butterworth high-pass filter
        # Order 4 provides good balance between steepness and phase response
        nyquist = self.sample_rate / 2.0
        normalized_cutoff = self.highpass_cutoff / nyquist
        
        # Ensure cutoff is valid (must be < 1.0)
        normalized_cutoff = min(normalized_cutoff, 0.99)
        
        self.hp_b, self.hp_a = signal.butter(
            4,  # Filter order
            normalized_cutoff,
            btype='high',
            analog=False
        )
    
    def enhance_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply all enhancement steps to audio data.
        
        This is the main entry point for audio enhancement. It orchestrates
        all the enhancement steps in the correct order.
        
        Args:
            audio_data: Input audio as numpy array (float32, range -1 to 1)
            
        Returns:
            Enhanced audio as numpy array (float32, range -1 to 1)
            
        Raises:
            ValueError: If audio_data is empty or invalid
        """
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Audio data cannot be empty")
        
        # Ensure audio is float32
        audio = audio_data.astype(np.float32)
        
        # Apply enhancement pipeline in order
        audio = self._apply_highpass_filter(audio)
        audio = self._apply_noise_gate(audio)
        audio = self._normalize_volume(audio)
        audio = self._reduce_noise(audio)
        audio = self._soft_limit(audio)
        
        return audio
    
    def _apply_highpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply high-pass filter to remove low-frequency noise.
        
        This removes rumble, fan noise, and other low-frequency interference
        that is common with laptop microphones.
        
        Args:
            audio: Input audio array
            
        Returns:
            Filtered audio array
        """
        # Apply the filter using scipy's filtfilt for zero-phase filtering
        # filtfilt applies the filter forwards and backwards to eliminate phase distortion
        try:
            filtered = signal.filtfilt(self.hp_b, self.hp_a, audio)
            return filtered.astype(np.float32)
        except Exception as e:
            logger.warning(f"High-pass filter failed: {e}, returning original audio")
            return audio
    
    def _apply_noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply noise gate to suppress quiet background noise.
        
        Audio below the threshold is attenuated, while louder speech
        segments are preserved. This helps reduce constant background noise.
        
        Args:
            audio: Input audio array
            
        Returns:
            Noise-gated audio array
        """
        # Calculate absolute amplitude
        amplitude = np.abs(audio)
        
        # Create gate mask: 1.0 for audio above threshold, scaled down below
        # Use smooth transition to avoid clicks
        gate_mask = np.where(
            amplitude > self.noise_gate_threshold,
            1.0,
            amplitude / self.noise_gate_threshold * 0.1  # Reduce to 10% below threshold
        )
        
        # Apply gate
        gated = audio * gate_mask
        
        return gated.astype(np.float32)
    
    def _normalize_volume(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio volume to target RMS level.
        
        This ensures consistent volume levels across different recordings
        and speaking volumes. Applies gain limiting to prevent excessive
        amplification.
        
        Args:
            audio: Input audio array
            
        Returns:
            Volume-normalized audio array
        """
        # Calculate current RMS level
        current_rms = np.sqrt(np.mean(audio ** 2))
        
        # Avoid division by zero
        if current_rms < 1e-6:
            logger.debug("Audio RMS too low for normalization")
            return audio
        
        # Calculate required gain
        required_gain = self.target_rms / current_rms
        
        # Limit gain to max_gain to avoid over-amplification
        gain = min(required_gain, self.max_gain)
        
        # Apply gain
        normalized = audio * gain
        
        logger.debug(f"Volume normalization: RMS {current_rms:.4f} -> {current_rms * gain:.4f}, gain={gain:.2f}x")
        
        return normalized.astype(np.float32)
    
    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply spectral noise reduction using spectral subtraction.
        
        This estimates the noise profile from quiet segments and subtracts
        it from the entire signal in the frequency domain.
        
        Args:
            audio: Input audio array
            
        Returns:
            Noise-reduced audio array
        """
        # For short audio segments, skip noise reduction
        if len(audio) < 512:
            return audio
        
        try:
            # Use first 10% of audio as noise profile (assuming it contains some noise)
            noise_sample_length = max(512, len(audio) // 10)
            noise_sample = audio[:noise_sample_length]
            
            # Pad noise sample to match audio length for consistent FFT size
            noise_padded = np.pad(noise_sample, (0, len(audio) - len(noise_sample)), mode='constant')
            
            # Compute noise spectrum
            noise_fft = np.fft.rfft(noise_padded)
            noise_magnitude = np.abs(noise_fft)
            
            # Compute signal spectrum
            signal_fft = np.fft.rfft(audio)
            signal_magnitude = np.abs(signal_fft)
            signal_phase = np.angle(signal_fft)
            
            # Spectral subtraction
            # Subtract scaled noise magnitude from signal magnitude
            reduced_magnitude = signal_magnitude - self.noise_reduction_strength * noise_magnitude
            
            # Ensure non-negative (floor at small positive value)
            reduced_magnitude = np.maximum(reduced_magnitude, signal_magnitude * 0.1)
            
            # Reconstruct signal with original phase
            reduced_fft = reduced_magnitude * np.exp(1j * signal_phase)
            
            # Inverse FFT
            reduced_audio = np.fft.irfft(reduced_fft, n=len(audio))
            
            return reduced_audio.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}, returning original audio")
            return audio
    
    def _soft_limit(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply soft limiting to prevent clipping.
        
        Uses a soft knee compression curve to smoothly limit peaks
        while preserving the overall dynamics of the audio.
        
        Args:
            audio: Input audio array
            
        Returns:
            Soft-limited audio array with values in range [-1.0, 1.0]
        """
        # Soft limiting threshold (start compressing above this)
        threshold = 0.8
        
        # Apply soft knee compression
        # For values below threshold, pass through unchanged
        # For values above threshold, apply smooth compression
        
        def soft_clip(x):
            """Soft clipping function using tanh."""
            if np.abs(x) <= threshold:
                return x
            else:
                # Smooth compression above threshold
                sign = np.sign(x)
                excess = np.abs(x) - threshold
                compressed = threshold + (1.0 - threshold) * np.tanh(excess / (1.0 - threshold))
                return sign * compressed
        
        # Vectorize the soft clip function
        vectorized_soft_clip = np.vectorize(soft_clip)
        limited = vectorized_soft_clip(audio)
        
        # Final hard clip to ensure we're in range (should rarely trigger)
        limited = np.clip(limited, -1.0, 1.0)
        
        return limited.astype(np.float32)
