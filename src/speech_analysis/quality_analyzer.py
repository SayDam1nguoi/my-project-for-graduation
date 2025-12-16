"""
Speech Quality Analyzer Module

This module provides speech quality analysis functionality including:
- Clarity Score: Measures speech clarity based on SNR, spectral clarity, and ZCR
- Fluency Score: Measures speech fluency based on speech rate, pauses, and rhythm
- Quality recommendations based on scores
- Real-time quality updates with callback support

The analyzer uses audio signal processing techniques to evaluate speech quality
and provide actionable feedback for improvement.
"""

import numpy as np
import librosa
from scipy import signal
from scipy.stats import variation
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Callable, Tuple
import threading
import time


@dataclass
class QualityReport:
    """
    Quality report containing speech analysis results.
    
    Attributes:
        clarity_score: Overall clarity score (0-100)
        fluency_score: Overall fluency score (0-100)
        snr: Signal-to-Noise Ratio in dB
        speech_rate: Speech rate in syllables per second
        pause_count: Number of pauses detected
        avg_pause_duration: Average pause duration in seconds
        recommendations: List of improvement recommendations
        timestamp: When the report was generated
        is_clear: Whether speech is considered clear (clarity >= 60)
        is_fluent: Whether speech is considered fluent (fluency >= 60)
    """
    clarity_score: float
    fluency_score: float
    snr: float
    speech_rate: float
    pause_count: int
    avg_pause_duration: float
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    is_clear: bool = field(init=False)
    is_fluent: bool = field(init=False)
    
    def __post_init__(self):
        """Set derived fields after initialization."""
        self.is_clear = self.clarity_score >= 60.0
        self.is_fluent = self.fluency_score >= 60.0


@dataclass
class QualityConfig:
    """
    Configuration for quality analysis.
    
    Attributes:
        clarity_threshold: Minimum clarity score to be considered clear
        fluency_threshold: Minimum fluency score to be considered fluent
        update_interval: Interval for real-time updates in seconds
        min_speech_duration: Minimum audio duration for analysis in seconds
    """
    clarity_threshold: float = 60.0
    fluency_threshold: float = 60.0
    update_interval: float = 2.0
    min_speech_duration: float = 1.0


class SpeechQualityAnalyzer:
    """
    Analyzer for speech quality metrics.
    
    This class analyzes audio data to compute:
    - Clarity Score: Based on SNR, spectral clarity, and ZCR stability
    - Fluency Score: Based on speech rate, pause analysis, and rhythm consistency
    
    The analyzer can operate in real-time mode with callbacks for GUI updates.
    """
    
    def __init__(self, sample_rate: int = 16000, config: Optional[QualityConfig] = None):
        """
        Initialize speech quality analyzer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            config: Quality configuration settings
        """
        self.sample_rate = sample_rate
        self.config = config or QualityConfig()
        
        # Real-time update support
        self._update_callback: Optional[Callable[[QualityReport], None]] = None
        self._update_thread: Optional[threading.Thread] = None
        self._stop_updates = threading.Event()
        self._latest_audio: Optional[np.ndarray] = None
        self._audio_lock = threading.Lock()
        
    def analyze_clarity(self, audio: np.ndarray) -> float:
        """
        Calculate clarity score based on SNR, spectral clarity, and ZCR.
        
        Formula: Clarity = 0.4×SNR + 0.4×Spectral + 0.2×ZCR
        
        Args:
            audio: Audio data as numpy array (int16 or float)
        
        Returns:
            Clarity score (0-100)
        """
        if len(audio) == 0:
            return 0.0
        
        # Convert to float if needed
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)
        
        # Calculate SNR component
        snr_score = self._calculate_snr_score(audio_float)
        
        # Calculate spectral clarity component
        spectral_score = self._calculate_spectral_clarity(audio_float)
        
        # Calculate ZCR stability component
        zcr_score = self._calculate_zcr_stability(audio_float)
        
        # Combine scores with weights
        clarity = (0.4 * snr_score) + (0.4 * spectral_score) + (0.2 * zcr_score)
        
        # Ensure score is in range [0, 100]
        return np.clip(clarity, 0.0, 100.0)
    
    def analyze_fluency(self, audio: np.ndarray) -> float:
        """
        Calculate fluency score based on speech rate, pauses, and rhythm.
        
        Formula: Fluency = 0.3×SpeechRate + 0.4×Pause + 0.3×Rhythm
        
        Args:
            audio: Audio data as numpy array (int16 or float)
        
        Returns:
            Fluency score (0-100)
        """
        if len(audio) == 0:
            return 0.0
        
        # Convert to float if needed
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)
        
        # Calculate speech rate component
        speech_rate_score = self._calculate_speech_rate_score(audio_float)
        
        # Calculate pause component
        pause_score = self._calculate_pause_score(audio_float)
        
        # Calculate rhythm consistency component
        rhythm_score = self._calculate_rhythm_score(audio_float)
        
        # Combine scores with weights
        fluency = (0.3 * speech_rate_score) + (0.4 * pause_score) + (0.3 * rhythm_score)
        
        # Ensure score is in range [0, 100]
        return np.clip(fluency, 0.0, 100.0)
    
    def get_quality_report(self, audio: np.ndarray) -> QualityReport:
        """
        Generate a complete quality report for the audio.
        
        Args:
            audio: Audio data as numpy array
        
        Returns:
            QualityReport with all metrics and recommendations
        """
        if len(audio) == 0:
            return QualityReport(
                clarity_score=0.0,
                fluency_score=0.0,
                snr=0.0,
                speech_rate=0.0,
                pause_count=0,
                avg_pause_duration=0.0,
                recommendations=["No audio data to analyze"]
            )
        
        # Convert to float if needed
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)
        
        # Calculate clarity and fluency scores
        clarity_score = self.analyze_clarity(audio)
        fluency_score = self.analyze_fluency(audio)
        
        # Calculate detailed metrics
        snr = self._calculate_snr(audio_float)
        speech_rate = self._calculate_speech_rate(audio_float)
        pause_count, avg_pause_duration = self._analyze_pauses(audio_float)
        
        # Create report
        report = QualityReport(
            clarity_score=clarity_score,
            fluency_score=fluency_score,
            snr=snr,
            speech_rate=speech_rate,
            pause_count=pause_count,
            avg_pause_duration=avg_pause_duration,
            timestamp=datetime.now()
        )
        
        # Generate recommendations
        report.recommendations = self.get_recommendations(report)
        
        return report
    
    def get_recommendations(self, report: QualityReport) -> List[str]:
        """
        Generate improvement recommendations based on quality scores.
        
        Args:
            report: QualityReport to analyze
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Clarity recommendations
        if report.clarity_score < self.config.clarity_threshold:
            recommendations.append("Giọng nói không rõ ràng")
            
            if report.snr < 10:
                recommendations.append("- Giảm nhiễu xung quanh hoặc di chuyển đến nơi yên tĩnh hơn")
            
            recommendations.append("- Phát âm rõ ràng hơn, đặc biệt là các phụ âm cuối")
            recommendations.append("- Nói với âm lượng vừa phải, không quá nhỏ")
        else:
            recommendations.append("Giọng nói rõ ràng, dễ hiểu")
        
        # Fluency recommendations
        if report.fluency_score < self.config.fluency_threshold:
            recommendations.append("Giọng nói chưa lưu loát")
            
            if report.speech_rate < 2.5:
                recommendations.append("- Tăng tốc độ nói lên một chút (hiện tại hơi chậm)")
            elif report.speech_rate > 5.5:
                recommendations.append("- Giảm tốc độ nói xuống một chút (hiện tại hơi nhanh)")
            
            if report.pause_count > 10 and report.avg_pause_duration > 1.0:
                recommendations.append("- Giảm số lần ngắt quãng và thời gian ngắt")
                recommendations.append("- Luyện tập để nói liền mạch hơn")
        else:
            recommendations.append("Giọng nói lưu loát, tự nhiên")
            
            if report.speech_rate >= 3.0 and report.speech_rate <= 5.0:
                recommendations.append("- Tốc độ nói phù hợp")
        
        # SNR specific recommendations
        if report.snr < 15:
            recommendations.append("- Kiểm tra microphone và giảm nhiễu nền")
        
        return recommendations

    
    # ==================== Clarity Score Components ====================
    
    def _calculate_snr(self, audio: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio in dB.
        
        Args:
            audio: Audio data (float, normalized)
        
        Returns:
            SNR in decibels
        """
        # Use voice activity detection to separate signal from noise
        # Simple energy-based approach
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        
        # Calculate frame energies
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energies = np.sum(frames ** 2, axis=0)
        
        if len(energies) == 0:
            return 0.0
        
        # Threshold-based VAD (simple approach)
        threshold = np.percentile(energies, 40)  # Bottom 40% is likely noise
        
        signal_frames = energies[energies > threshold]
        noise_frames = energies[energies <= threshold]
        
        if len(signal_frames) == 0 or len(noise_frames) == 0:
            return 20.0  # Default moderate SNR
        
        signal_power = np.mean(signal_frames)
        noise_power = np.mean(noise_frames)
        
        if noise_power == 0:
            return 40.0  # Very high SNR
        
        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)
    
    def _calculate_snr_score(self, audio: np.ndarray) -> float:
        """
        Calculate SNR score normalized to 0-100.
        
        Args:
            audio: Audio data (float, normalized)
        
        Returns:
            SNR score (0-100)
        """
        snr = self._calculate_snr(audio)
        
        # Map SNR (dB) to 0-100 score
        # Typical SNR ranges: 0-40 dB
        # Good speech: > 20 dB
        # Acceptable: 10-20 dB
        # Poor: < 10 dB
        
        if snr >= 30:
            score = 100.0
        elif snr >= 20:
            # Linear mapping from 20-30 dB to 80-100
            score = 80.0 + (snr - 20) * 2.0
        elif snr >= 10:
            # Linear mapping from 10-20 dB to 50-80
            score = 50.0 + (snr - 10) * 3.0
        else:
            # Linear mapping from 0-10 dB to 0-50
            score = max(0.0, snr * 5.0)
        
        return np.clip(score, 0.0, 100.0)
    
    def _calculate_spectral_clarity(self, audio: np.ndarray) -> float:
        """
        Calculate spectral clarity based on energy in speech frequency range.
        
        Speech frequencies are typically 300-3400 Hz.
        
        Args:
            audio: Audio data (float, normalized)
        
        Returns:
            Spectral clarity score (0-100)
        """
        # Compute power spectral density
        frequencies, psd = signal.welch(audio, fs=self.sample_rate, nperseg=1024)
        
        # Define speech frequency range (300-3400 Hz)
        speech_range = (frequencies >= 300) & (frequencies <= 3400)
        total_range = frequencies <= 8000  # Up to Nyquist/2
        
        if not np.any(speech_range) or not np.any(total_range):
            return 50.0  # Default moderate score
        
        # Calculate energy in speech range vs total energy
        speech_energy = np.sum(psd[speech_range])
        total_energy = np.sum(psd[total_range])
        
        if total_energy == 0:
            return 50.0
        
        # Ratio of speech energy to total energy
        speech_ratio = speech_energy / total_energy
        
        # Good speech should have 60-80% energy in speech range
        if speech_ratio >= 0.6:
            score = 100.0
        elif speech_ratio >= 0.4:
            # Linear mapping from 0.4-0.6 to 70-100
            score = 70.0 + (speech_ratio - 0.4) * 150.0
        elif speech_ratio >= 0.2:
            # Linear mapping from 0.2-0.4 to 40-70
            score = 40.0 + (speech_ratio - 0.2) * 150.0
        else:
            # Linear mapping from 0-0.2 to 0-40
            score = speech_ratio * 200.0
        
        return np.clip(score, 0.0, 100.0)
    
    def _calculate_zcr_stability(self, audio: np.ndarray) -> float:
        """
        Calculate Zero Crossing Rate stability as a measure of clarity.
        
        Stable ZCR indicates clear pronunciation.
        
        Args:
            audio: Audio data (float, normalized)
        
        Returns:
            ZCR stability score (0-100)
        """
        # Calculate ZCR for frames
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        
        zcr = librosa.feature.zero_crossing_rate(
            audio, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        if len(zcr) == 0:
            return 50.0
        
        # Calculate coefficient of variation (std/mean)
        # Lower variation = more stable = clearer speech
        mean_zcr = np.mean(zcr)
        std_zcr = np.std(zcr)
        
        if mean_zcr == 0:
            return 50.0
        
        cv = std_zcr / mean_zcr
        
        # Map coefficient of variation to score
        # Good speech: CV < 0.5
        # Acceptable: CV 0.5-1.0
        # Poor: CV > 1.0
        
        if cv <= 0.3:
            score = 100.0
        elif cv <= 0.6:
            # Linear mapping from 0.3-0.6 to 80-100
            score = 80.0 + (0.6 - cv) * 66.67
        elif cv <= 1.0:
            # Linear mapping from 0.6-1.0 to 50-80
            score = 50.0 + (1.0 - cv) * 75.0
        else:
            # Linear mapping from 1.0-2.0 to 0-50
            score = max(0.0, 50.0 - (cv - 1.0) * 50.0)
        
        return np.clip(score, 0.0, 100.0)
    
    # ==================== Fluency Score Components ====================
    
    def _calculate_speech_rate(self, audio: np.ndarray) -> float:
        """
        Calculate speech rate in syllables per second.
        
        Uses onset detection as a proxy for syllable counting.
        
        Args:
            audio: Audio data (float, normalized)
        
        Returns:
            Speech rate in syllables per second
        """
        # Detect onsets (approximates syllables)
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sample_rate,
            units='time',
            backtrack=True
        )
        
        # Calculate duration
        duration = len(audio) / self.sample_rate
        
        if duration == 0:
            return 0.0
        
        # Estimate syllables (onsets are a rough proxy)
        syllable_count = len(onsets)
        speech_rate = syllable_count / duration
        
        return float(speech_rate)
    
    def _calculate_speech_rate_score(self, audio: np.ndarray) -> float:
        """
        Calculate speech rate score normalized to 0-100.
        
        Optimal Vietnamese speech rate: 3-5 syllables/second
        
        Args:
            audio: Audio data (float, normalized)
        
        Returns:
            Speech rate score (0-100)
        """
        rate = self._calculate_speech_rate(audio)
        
        # Optimal range: 3-5 syllables/second for Vietnamese
        if 3.0 <= rate <= 5.0:
            score = 100.0
        elif 2.5 <= rate < 3.0:
            # Slightly slow: linear from 2.5-3.0 to 80-100
            score = 80.0 + (rate - 2.5) * 40.0
        elif 5.0 < rate <= 6.0:
            # Slightly fast: linear from 5.0-6.0 to 100-80
            score = 100.0 - (rate - 5.0) * 20.0
        elif 2.0 <= rate < 2.5:
            # Slow: linear from 2.0-2.5 to 50-80
            score = 50.0 + (rate - 2.0) * 60.0
        elif 6.0 < rate <= 7.0:
            # Fast: linear from 6.0-7.0 to 80-50
            score = 80.0 - (rate - 6.0) * 30.0
        elif rate < 2.0:
            # Very slow: linear from 0-2.0 to 0-50
            score = rate * 25.0
        else:  # rate > 7.0
            # Very fast: exponential decay
            score = max(0.0, 50.0 - (rate - 7.0) * 10.0)
        
        return np.clip(score, 0.0, 100.0)
    
    def _analyze_pauses(self, audio: np.ndarray) -> Tuple[int, float]:
        """
        Analyze pauses in speech.
        
        Args:
            audio: Audio data (float, normalized)
        
        Returns:
            Tuple of (pause_count, average_pause_duration)
        """
        # Calculate RMS energy for frames
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Threshold for silence detection
        threshold = np.percentile(rms, 30)  # Bottom 30% is likely silence
        
        # Detect silent frames
        is_silent = rms < threshold
        
        # Find pause regions (consecutive silent frames)
        pauses = []
        in_pause = False
        pause_start = 0
        
        for i, silent in enumerate(is_silent):
            if silent and not in_pause:
                # Start of pause
                in_pause = True
                pause_start = i
            elif not silent and in_pause:
                # End of pause
                pause_duration = (i - pause_start) * hop_length / self.sample_rate
                # Only count pauses longer than 0.3 seconds
                if pause_duration >= 0.3:
                    pauses.append(pause_duration)
                in_pause = False
        
        # Handle case where audio ends in pause
        if in_pause:
            pause_duration = (len(is_silent) - pause_start) * hop_length / self.sample_rate
            if pause_duration >= 0.3:
                pauses.append(pause_duration)
        
        pause_count = len(pauses)
        avg_pause_duration = np.mean(pauses) if pauses else 0.0
        
        return pause_count, float(avg_pause_duration)
    
    def _calculate_pause_score(self, audio: np.ndarray) -> float:
        """
        Calculate pause score based on pause frequency and duration.
        
        Args:
            audio: Audio data (float, normalized)
        
        Returns:
            Pause score (0-100)
        """
        pause_count, avg_pause_duration = self._analyze_pauses(audio)
        
        # Calculate audio duration
        duration = len(audio) / self.sample_rate
        
        if duration == 0:
            return 50.0
        
        # Calculate pause rate (pauses per minute)
        pause_rate = (pause_count / duration) * 60.0
        
        # Good speech: 2-6 pauses per minute, avg duration 0.3-0.8s
        # Score based on pause rate
        if 2.0 <= pause_rate <= 6.0:
            rate_score = 100.0
        elif 1.0 <= pause_rate < 2.0:
            rate_score = 70.0 + (pause_rate - 1.0) * 30.0
        elif 6.0 < pause_rate <= 10.0:
            rate_score = 100.0 - (pause_rate - 6.0) * 10.0
        elif pause_rate < 1.0:
            rate_score = pause_rate * 70.0
        else:  # pause_rate > 10.0
            rate_score = max(0.0, 60.0 - (pause_rate - 10.0) * 5.0)
        
        # Score based on average pause duration
        if 0.3 <= avg_pause_duration <= 0.8:
            duration_score = 100.0
        elif 0.8 < avg_pause_duration <= 1.5:
            duration_score = 100.0 - (avg_pause_duration - 0.8) * 40.0
        elif avg_pause_duration > 1.5:
            duration_score = max(0.0, 72.0 - (avg_pause_duration - 1.5) * 30.0)
        else:  # avg_pause_duration < 0.3
            duration_score = 80.0
        
        # Combine rate and duration scores
        score = 0.6 * rate_score + 0.4 * duration_score
        
        return np.clip(score, 0.0, 100.0)
    
    def _calculate_rhythm_score(self, audio: np.ndarray) -> float:
        """
        Calculate rhythm consistency score.
        
        Measures how consistent the speech rhythm is.
        
        Args:
            audio: Audio data (float, normalized)
        
        Returns:
            Rhythm score (0-100)
        """
        # Detect onsets
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sample_rate,
            units='time',
            backtrack=True
        )
        
        if len(onsets) < 3:
            return 50.0  # Not enough data
        
        # Calculate inter-onset intervals
        intervals = np.diff(onsets)
        
        if len(intervals) == 0:
            return 50.0
        
        # Calculate coefficient of variation
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval == 0:
            return 50.0
        
        cv = std_interval / mean_interval
        
        # Lower CV = more consistent rhythm = better score
        # Good speech: CV < 0.5
        # Acceptable: CV 0.5-1.0
        # Poor: CV > 1.0
        
        if cv <= 0.4:
            score = 100.0
        elif cv <= 0.7:
            # Linear mapping from 0.4-0.7 to 100-80
            score = 100.0 - (cv - 0.4) * 66.67
        elif cv <= 1.0:
            # Linear mapping from 0.7-1.0 to 80-60
            score = 80.0 - (cv - 0.7) * 66.67
        else:
            # Linear mapping from 1.0-2.0 to 60-0
            score = max(0.0, 60.0 - (cv - 1.0) * 60.0)
        
        return np.clip(score, 0.0, 100.0)
    
    # ==================== Real-time Update Support ====================
    
    def start_realtime_updates(
        self, 
        callback: Callable[[QualityReport], None],
        audio_source: Callable[[], np.ndarray]
    ) -> None:
        """
        Start real-time quality updates with callback.
        
        Args:
            callback: Function to call with quality reports
            audio_source: Function that returns current audio data
        """
        self._update_callback = callback
        self._stop_updates.clear()
        
        def update_loop():
            while not self._stop_updates.is_set():
                try:
                    # Get current audio
                    audio = audio_source()
                    
                    if audio is not None and len(audio) > 0:
                        # Generate quality report
                        report = self.get_quality_report(audio)
                        
                        # Call callback
                        if self._update_callback:
                            self._update_callback(report)
                
                except Exception as e:
                    # Log error but continue
                    print(f"Error in quality update: {e}")
                
                # Wait for update interval
                time.sleep(self.config.update_interval)
        
        self._update_thread = threading.Thread(target=update_loop, daemon=True)
        self._update_thread.start()
    
    def stop_realtime_updates(self) -> None:
        """Stop real-time quality updates."""
        self._stop_updates.set()
        
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=1.0)
        
        self._update_callback = None
        self._update_thread = None
