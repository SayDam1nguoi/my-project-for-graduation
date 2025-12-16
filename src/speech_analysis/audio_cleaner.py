"""
Audio Cleaner Module for Enhanced Vietnamese STT

This module provides audio cleaning capabilities including:
- Noise reduction using noisereduce library
- Audio normalization to optimal range
- Audio enhancement filters for low-quality audio
- SNR (Signal-to-Noise Ratio) calculation
"""

import numpy as np
import noisereduce as nr
from scipy import signal
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AudioCleaner:
    """
    Audio cleaner for preprocessing audio before STT processing.
    
    Provides noise reduction, normalization, and enhancement capabilities
    to improve STT accuracy in noisy environments.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio cleaner with sample rate.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
        """
        self.sample_rate = sample_rate
        logger.info(f"AudioCleaner initialized with sample_rate={sample_rate}")
    
    def clean_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Clean audio with noise reduction and normalization.
        
        Args:
            audio: Raw audio data (float32, normalized -1 to 1)
            
        Returns:
            Cleaned audio data
        """
        try:
            # Apply noise reduction
            cleaned = self.reduce_noise(audio)
            
            # Normalize audio levels
            cleaned = self.normalize_audio(cleaned)
            
            # Check if enhancement is needed
            snr = calculate_snr(audio)
            if snr < 10.0:  # Low quality threshold
                logger.debug(f"Low SNR detected ({snr:.2f} dB), applying enhancement")
                cleaned = self.enhance_audio(cleaned)
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Audio cleaning failed: {e}. Returning normalized audio.")
            # Fallback: just normalize
            return self.normalize_audio(audio)
    
    def reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction using noisereduce library.
        
        Args:
            audio: Input audio data
            
        Returns:
            Noise-reduced audio data
        """
        try:
            # Use stationary noise reduction
            # noisereduce will estimate noise from the audio itself
            reduced = nr.reduce_noise(
                y=audio,
                sr=self.sample_rate,
                stationary=True,
                prop_decrease=0.8  # Reduce noise by 80%
            )
            return reduced
            
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio levels to optimal range [-1.0, 1.0].
        
        Args:
            audio: Input audio data
            
        Returns:
            Normalized audio data in range [-1.0, 1.0]
        """
        # Handle empty audio
        if len(audio) == 0:
            logger.warning("Audio is empty")
            return audio
        
        # Avoid division by zero
        max_val = np.abs(audio).max()
        if max_val < 1e-10:
            logger.warning("Audio is silent or near-silent")
            return audio
        
        # Normalize to [-1.0, 1.0] range
        normalized = audio / max_val
        
        # Ensure values are within bounds (handle floating point errors)
        normalized = np.clip(normalized, -1.0, 1.0)
        
        return normalized
    
    def enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply enhancement filters for low-quality audio.
        
        Applies:
        - High-pass filter to remove low-frequency rumble
        - Pre-emphasis filter to boost high frequencies
        
        Args:
            audio: Input audio data
            
        Returns:
            Enhanced audio data
        """
        try:
            enhanced = audio.copy()
            
            # High-pass filter to remove low-frequency noise (< 80 Hz)
            nyquist = self.sample_rate / 2
            cutoff = 80.0 / nyquist
            b, a = signal.butter(4, cutoff, btype='high')
            enhanced = signal.filtfilt(b, a, enhanced)
            
            # Pre-emphasis filter to boost high frequencies
            # This helps with speech recognition
            pre_emphasis = 0.97
            enhanced = np.append(enhanced[0], enhanced[1:] - pre_emphasis * enhanced[:-1])
            
            # Normalize after filtering
            enhanced = self.normalize_audio(enhanced)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Audio enhancement failed: {e}")
            return audio


def calculate_snr(audio: np.ndarray, noise_duration: float = 0.5, sample_rate: int = 16000) -> float:
    """
    Calculate Signal-to-Noise Ratio (SNR) of audio.
    
    Estimates SNR by comparing the power of the entire signal
    to the power of the initial segment (assumed to be noise).
    
    Args:
        audio: Input audio data
        noise_duration: Duration in seconds to use for noise estimation (default: 0.5)
        sample_rate: Audio sample rate in Hz (default: 16000)
        
    Returns:
        SNR in decibels (dB)
    """
    if len(audio) == 0:
        return 0.0
    
    # Calculate signal power
    signal_power = np.mean(audio ** 2)
    
    # Estimate noise power from initial segment
    noise_samples = int(noise_duration * sample_rate)
    if noise_samples >= len(audio):
        # If audio is too short, use first 10% as noise estimate
        noise_samples = max(1, len(audio) // 10)
    
    noise_segment = audio[:noise_samples]
    noise_power = np.mean(noise_segment ** 2)
    
    # Avoid division by zero
    if noise_power < 1e-10:
        return 100.0  # Very high SNR (essentially no noise)
    
    # Calculate SNR in dB
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    return snr_db
