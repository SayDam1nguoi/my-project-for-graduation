"""
Quality Analyzer Implementation

Refactored quality analyzer that implements IQualityAnalyzer interface.
Provides audio quality analysis, transcription quality analysis, and recommendations.
"""

import numpy as np
import librosa
from scipy import signal
from typing import List
from datetime import datetime

from ..interfaces.processing import IQualityAnalyzer
from ..models.quality import (
    AudioQualityReport,
    TranscriptionQualityReport,
    QualityReport,
)
from ..models.transcription import TranscriptionResult


class QualityAnalyzer(IQualityAnalyzer):
    """
    Quality analyzer for audio and transcription quality assessment.
    
    Analyzes:
    - Audio quality: clarity, SNR, spectral characteristics
    - Transcription quality: fluency, confidence, hallucination risk
    - Provides actionable recommendations
    """
    
    def __init__(
        self,
        clarity_threshold: float = 0.6,
        fluency_threshold: float = 0.6,
        min_confidence: float = 0.5,
    ):
        """
        Initialize quality analyzer.
        
        Args:
            clarity_threshold: Minimum clarity score (0-1) for good quality.
            fluency_threshold: Minimum fluency score (0-1) for good quality.
            min_confidence: Minimum confidence threshold for filtering.
        """
        self.clarity_threshold = clarity_threshold
        self.fluency_threshold = fluency_threshold
        self.min_confidence = min_confidence
    
    def analyze_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> AudioQualityReport:
        """
        Analyze audio quality.
        
        Args:
            audio: Audio data to analyze.
            sample_rate: Sample rate of audio.
            
        Returns:
            Audio quality report.
        """
        if len(audio) == 0:
            return AudioQualityReport(
                clarity_score=0.0,
                snr=0.0,
                rms_level=0.0,
                zero_crossing_rate=0.0,
                spectral_centroid=0.0,
                is_silent=True,
                has_speech=False,
            )
        
        # Convert to float if needed
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)
        
        # Calculate metrics
        snr = self._calculate_snr(audio_float, sample_rate)
        rms_level = float(np.sqrt(np.mean(audio_float ** 2)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio_float)[0]))
        
        # Calculate spectral centroid
        spectral_centroid = float(
            np.mean(librosa.feature.spectral_centroid(y=audio_float, sr=sample_rate)[0])
        )
        
        # Determine if silent or has speech
        is_silent = rms_level < 0.01
        has_speech = not is_silent and snr > 5.0
        
        # Calculate clarity score (0-1)
        clarity_score = self._calculate_clarity_score(
            snr, audio_float, sample_rate
        )
        
        return AudioQualityReport(
            clarity_score=clarity_score,
            snr=snr,
            rms_level=rms_level,
            zero_crossing_rate=zcr,
            spectral_centroid=spectral_centroid,
            is_silent=is_silent,
            has_speech=has_speech,
        )
    
    def analyze_transcription(
        self,
        result: TranscriptionResult,
    ) -> TranscriptionQualityReport:
        """
        Analyze transcription quality.
        
        Args:
            result: Transcription result to analyze.
            
        Returns:
            Transcription quality report.
        """
        if not result.segments:
            return TranscriptionQualityReport(
                fluency_score=0.0,
                confidence_avg=0.0,
                confidence_min=0.0,
                confidence_max=0.0,
                speech_rate=0.0,
                hallucination_risk=0.0,
                segment_count=0,
            )
        
        # Calculate confidence statistics
        confidences = [seg.confidence for seg in result.segments]
        confidence_avg = float(np.mean(confidences))
        confidence_min = float(np.min(confidences))
        confidence_max = float(np.max(confidences))
        
        # Calculate speech rate (words per minute)
        total_words = sum(seg.word_count for seg in result.segments)
        total_duration = result.total_duration
        speech_rate = (total_words / total_duration * 60.0) if total_duration > 0 else 0.0
        
        # Calculate hallucination risk
        hallucination_risk = self._calculate_hallucination_risk(result)
        
        # Calculate fluency score (0-1)
        fluency_score = self._calculate_fluency_score(
            speech_rate, confidence_avg, result.segments
        )
        
        return TranscriptionQualityReport(
            fluency_score=fluency_score,
            confidence_avg=confidence_avg,
            confidence_min=confidence_min,
            confidence_max=confidence_max,
            speech_rate=speech_rate,
            hallucination_risk=hallucination_risk,
            segment_count=len(result.segments),
        )
    
    def get_recommendations(
        self,
        audio_report: AudioQualityReport,
        transcription_report: TranscriptionQualityReport,
    ) -> List[str]:
        """
        Get improvement recommendations based on quality reports.
        
        Args:
            audio_report: Audio quality report.
            transcription_report: Transcription quality report.
            
        Returns:
            List of recommendations.
        """
        recommendations = []
        
        # Audio quality recommendations
        if audio_report.clarity_score < self.clarity_threshold:
            recommendations.append("Chất lượng âm thanh chưa tốt")
            
            if audio_report.snr < 10:
                recommendations.append(
                    "- SNR thấp: Di chuyển đến nơi yên tĩnh hơn hoặc sử dụng microphone tốt hơn"
                )
            
            if audio_report.rms_level < 0.05:
                recommendations.append(
                    "- Âm lượng thấp: Nói to hơn hoặc di chuyển microphone gần hơn"
                )
            elif audio_report.rms_level > 0.8:
                recommendations.append(
                    "- Âm lượng quá cao: Giảm âm lượng hoặc di chuyển microphone xa hơn"
                )
        else:
            recommendations.append("✓ Chất lượng âm thanh tốt")
        
        # Transcription quality recommendations
        if transcription_report.fluency_score < self.fluency_threshold:
            recommendations.append("Chất lượng phiên âm chưa tốt")
            
            if transcription_report.confidence_avg < self.min_confidence:
                recommendations.append(
                    "- Độ tin cậy thấp: Cải thiện chất lượng âm thanh hoặc phát âm rõ ràng hơn"
                )
            
            if transcription_report.speech_rate < 80:
                recommendations.append(
                    "- Tốc độ nói chậm: Tăng tốc độ nói lên một chút"
                )
            elif transcription_report.speech_rate > 200:
                recommendations.append(
                    "- Tốc độ nói nhanh: Giảm tốc độ nói xuống một chút"
                )
        else:
            recommendations.append("✓ Chất lượng phiên âm tốt")
        
        # Hallucination risk recommendations
        if transcription_report.hallucination_risk > 0.3:
            recommendations.append(
                "⚠ Nguy cơ hallucination cao: Kiểm tra kết quả phiên âm cẩn thận"
            )
            
            if audio_report.is_silent:
                recommendations.append(
                    "- Âm thanh im lặng: Đảm bảo microphone đang hoạt động"
                )
            elif not audio_report.has_speech:
                recommendations.append(
                    "- Không phát hiện giọng nói: Kiểm tra nguồn âm thanh"
                )
        
        # Confidence-based recommendations
        if transcription_report.confidence_min < 0.3:
            recommendations.append(
                "- Một số đoạn có độ tin cậy rất thấp: Xem xét lọc hoặc kiểm tra lại"
            )
        
        return recommendations
    
    def create_quality_report(
        self,
        audio: np.ndarray,
        sample_rate: int,
        result: TranscriptionResult,
    ) -> QualityReport:
        """
        Create comprehensive quality report.
        
        Args:
            audio: Audio data.
            sample_rate: Sample rate of audio.
            result: Transcription result.
            
        Returns:
            Complete quality report.
        """
        audio_report = self.analyze_audio(audio, sample_rate)
        transcription_report = self.analyze_transcription(result)
        recommendations = self.get_recommendations(audio_report, transcription_report)
        
        return QualityReport(
            audio_quality=audio_report,
            transcription_quality=transcription_report,
            recommendations=recommendations,
        )
    
    # ==================== Private Helper Methods ====================
    
    def _calculate_snr(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Calculate Signal-to-Noise Ratio in dB.
        
        Args:
            audio: Audio data (float, normalized).
            sample_rate: Sample rate of audio.
        
        Returns:
            SNR in decibels.
        """
        # Handle very short audio
        if len(audio) < 400:  # Less than 25ms at 16kHz
            # Use simple RMS-based estimation
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0.1:
                return 20.0  # Assume moderate SNR for short audio
            else:
                return 5.0
        
        # Use voice activity detection to separate signal from noise
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop
        
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
        
        if noise_power == 0 or noise_power < 1e-10:
            return 40.0  # Very high SNR (clean signal)
        
        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)
    
    def _calculate_clarity_score(
        self,
        snr: float,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """
        Calculate clarity score (0-1) based on SNR and spectral characteristics.
        
        Args:
            snr: Signal-to-noise ratio in dB.
            audio: Audio data (float, normalized).
            sample_rate: Sample rate of audio.
        
        Returns:
            Clarity score (0-1).
        """
        # SNR component (0-1)
        if snr >= 30:
            snr_score = 1.0
        elif snr >= 20:
            snr_score = 0.8 + (snr - 20) * 0.02
        elif snr >= 10:
            snr_score = 0.5 + (snr - 10) * 0.03
        else:
            snr_score = max(0.0, snr * 0.05)
        
        # Spectral clarity component (0-1)
        spectral_score = self._calculate_spectral_clarity(audio, sample_rate)
        
        # ZCR stability component (0-1)
        zcr_score = self._calculate_zcr_stability(audio, sample_rate)
        
        # Combine scores with weights
        clarity = (0.4 * snr_score) + (0.4 * spectral_score) + (0.2 * zcr_score)
        
        return np.clip(clarity, 0.0, 1.0)
    
    def _calculate_spectral_clarity(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """
        Calculate spectral clarity based on energy in speech frequency range.
        
        Speech frequencies are typically 300-3400 Hz.
        
        Args:
            audio: Audio data (float, normalized).
            sample_rate: Sample rate of audio.
        
        Returns:
            Spectral clarity score (0-1).
        """
        # Compute power spectral density
        frequencies, psd = signal.welch(audio, fs=sample_rate, nperseg=1024)
        
        # Define speech frequency range (300-3400 Hz)
        speech_range = (frequencies >= 300) & (frequencies <= 3400)
        total_range = frequencies <= 8000  # Up to Nyquist/2
        
        if not np.any(speech_range) or not np.any(total_range):
            return 0.5  # Default moderate score
        
        # Calculate energy in speech range vs total energy
        speech_energy = np.sum(psd[speech_range])
        total_energy = np.sum(psd[total_range])
        
        if total_energy == 0:
            return 0.5
        
        # Ratio of speech energy to total energy
        speech_ratio = speech_energy / total_energy
        
        # Good speech should have 60-80% energy in speech range
        if speech_ratio >= 0.6:
            score = 1.0
        elif speech_ratio >= 0.4:
            score = 0.7 + (speech_ratio - 0.4) * 1.5
        elif speech_ratio >= 0.2:
            score = 0.4 + (speech_ratio - 0.2) * 1.5
        else:
            score = speech_ratio * 2.0
        
        return np.clip(score, 0.0, 1.0)
    
    def _calculate_zcr_stability(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """
        Calculate Zero Crossing Rate stability as a measure of clarity.
        
        Stable ZCR indicates clear pronunciation.
        
        Args:
            audio: Audio data (float, normalized).
            sample_rate: Sample rate of audio.
        
        Returns:
            ZCR stability score (0-1).
        """
        # Calculate ZCR for frames
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop
        
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        if len(zcr) == 0:
            return 0.5
        
        # Calculate coefficient of variation (std/mean)
        # Lower variation = more stable = clearer speech
        mean_zcr = np.mean(zcr)
        std_zcr = np.std(zcr)
        
        if mean_zcr == 0:
            return 0.5
        
        cv = std_zcr / mean_zcr
        
        # Map coefficient of variation to score
        # Good speech: CV < 0.5
        if cv <= 0.3:
            score = 1.0
        elif cv <= 0.6:
            score = 0.8 + (0.6 - cv) * 0.667
        elif cv <= 1.0:
            score = 0.5 + (1.0 - cv) * 0.75
        else:
            score = max(0.0, 0.5 - (cv - 1.0) * 0.5)
        
        return np.clip(score, 0.0, 1.0)
    
    def _calculate_fluency_score(
        self,
        speech_rate: float,
        confidence_avg: float,
        segments: List,
    ) -> float:
        """
        Calculate fluency score (0-1) based on speech rate and confidence.
        
        Args:
            speech_rate: Speech rate in words per minute.
            confidence_avg: Average confidence score.
            segments: List of transcription segments.
        
        Returns:
            Fluency score (0-1).
        """
        # Speech rate component (0-1)
        # Optimal Vietnamese speech rate: 120-180 words per minute
        if 120 <= speech_rate <= 180:
            rate_score = 1.0
        elif 100 <= speech_rate < 120:
            rate_score = 0.8 + (speech_rate - 100) * 0.01
        elif 180 < speech_rate <= 220:
            rate_score = 1.0 - (speech_rate - 180) * 0.005
        elif 80 <= speech_rate < 100:
            rate_score = 0.5 + (speech_rate - 80) * 0.015
        elif 220 < speech_rate <= 280:
            rate_score = 0.8 - (speech_rate - 220) * 0.005
        elif speech_rate < 80:
            rate_score = speech_rate * 0.00625
        else:  # speech_rate > 280
            rate_score = max(0.0, 0.5 - (speech_rate - 280) * 0.002)
        
        # Confidence component (0-1)
        confidence_score = confidence_avg
        
        # Pause analysis component (0-1)
        pause_score = self._calculate_pause_score(segments)
        
        # Combine scores with weights
        fluency = (0.3 * rate_score) + (0.4 * confidence_score) + (0.3 * pause_score)
        
        return np.clip(fluency, 0.0, 1.0)
    
    def _calculate_pause_score(self, segments: List) -> float:
        """
        Calculate pause score based on segment gaps.
        
        Args:
            segments: List of transcription segments.
        
        Returns:
            Pause score (0-1).
        """
        if len(segments) < 2:
            return 1.0  # No pauses to analyze
        
        # Calculate gaps between segments
        gaps = []
        for i in range(len(segments) - 1):
            gap = segments[i + 1].start_time - segments[i].end_time
            if gap > 0.1:  # Only count gaps > 100ms
                gaps.append(gap)
        
        if not gaps:
            return 1.0  # No significant pauses
        
        avg_gap = np.mean(gaps)
        gap_count = len(gaps)
        
        # Calculate total duration
        total_duration = segments[-1].end_time - segments[0].start_time
        gap_rate = (gap_count / total_duration * 60.0) if total_duration > 0 else 0.0
        
        # Good speech: 2-6 pauses per minute, avg duration 0.3-0.8s
        if 2.0 <= gap_rate <= 6.0 and 0.3 <= avg_gap <= 0.8:
            score = 1.0
        elif 1.0 <= gap_rate <= 10.0 and 0.2 <= avg_gap <= 1.5:
            score = 0.7
        else:
            score = 0.5
        
        return score
    
    def _calculate_hallucination_risk(self, result: TranscriptionResult) -> float:
        """
        Calculate hallucination risk (0-1) based on transcription characteristics.
        
        Args:
            result: Transcription result.
        
        Returns:
            Hallucination risk score (0-1).
        """
        if not result.segments:
            return 0.0
        
        risk_factors = []
        
        # Low confidence segments
        low_confidence_count = sum(
            1 for seg in result.segments if seg.confidence < 0.5
        )
        low_confidence_ratio = low_confidence_count / len(result.segments)
        risk_factors.append(low_confidence_ratio)
        
        # Check for repetitive patterns (compression ratio proxy)
        text_length = len(result.text)
        unique_words = len(set(result.text.split()))
        total_words = len(result.text.split())
        
        if total_words > 0:
            repetition_ratio = 1.0 - (unique_words / total_words)
            risk_factors.append(repetition_ratio)
        
        # Check metadata for no_speech_prob if available (weighted higher)
        if result.metadata.get("no_speech_prob") is not None:
            no_speech_prob = result.metadata["no_speech_prob"]
            # Weight no_speech_prob more heavily by adding it twice
            risk_factors.append(no_speech_prob)
            risk_factors.append(no_speech_prob)
        
        # Check for very short segments with low confidence
        short_low_conf = sum(
            1 for seg in result.segments
            if seg.duration < 0.5 and seg.confidence < 0.6
        )
        short_low_conf_ratio = short_low_conf / len(result.segments)
        risk_factors.append(short_low_conf_ratio)
        
        # Average risk factors
        hallucination_risk = np.mean(risk_factors) if risk_factors else 0.0
        
        return np.clip(hallucination_risk, 0.0, 1.0)
