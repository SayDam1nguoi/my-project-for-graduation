"""Tests for Quality Analyzer implementation."""

import pytest
import numpy as np
from datetime import datetime

from src.speech_analysis.refactored.implementations.quality_analyzer import QualityAnalyzer
from src.speech_analysis.refactored.models.transcription import (
    TranscriptionResult,
    TranscriptionSegment,
)
from src.speech_analysis.refactored.models.quality import (
    AudioQualityReport,
    TranscriptionQualityReport,
    QualityReport,
)


class TestQualityAnalyzer:
    """Test suite for QualityAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create quality analyzer instance."""
        return QualityAnalyzer(
            clarity_threshold=0.6,
            fluency_threshold=0.6,
            min_confidence=0.5,
        )
    
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data (1 second at 16kHz)."""
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Generate speech-like signal (mix of frequencies)
        audio = (
            0.3 * np.sin(2 * np.pi * 300 * t) +  # Low frequency
            0.4 * np.sin(2 * np.pi * 1000 * t) +  # Mid frequency
            0.2 * np.sin(2 * np.pi * 2500 * t)    # High frequency
        )
        # Add some noise
        audio += 0.05 * np.random.randn(len(audio))
        return audio.astype(np.float32)
    
    @pytest.fixture
    def sample_transcription_result(self):
        """Create sample transcription result."""
        segments = [
            TranscriptionSegment(
                text="Xin chào",
                start_time=0.0,
                end_time=1.0,
                confidence=0.95,
                language="vi",
            ),
            TranscriptionSegment(
                text="tôi là",
                start_time=1.2,
                end_time=2.0,
                confidence=0.90,
                language="vi",
            ),
            TranscriptionSegment(
                text="trợ lý AI",
                start_time=2.5,
                end_time=3.5,
                confidence=0.85,
                language="vi",
            ),
        ]
        
        return TranscriptionResult(
            text="Xin chào tôi là trợ lý AI",
            segments=segments,
            language="vi",
            confidence=0.90,
            processing_time=0.5,
            engine_name="test_engine",
        )
    
    # ==================== Audio Quality Analysis Tests ====================
    
    def test_analyze_audio_basic(self, analyzer, sample_audio):
        """Test basic audio quality analysis."""
        report = analyzer.analyze_audio(sample_audio, 16000)
        
        assert isinstance(report, AudioQualityReport)
        assert 0.0 <= report.clarity_score <= 1.0
        assert report.snr > 0
        assert report.rms_level > 0
        assert report.zero_crossing_rate >= 0
        assert report.spectral_centroid > 0
        assert not report.is_silent
        assert isinstance(report.timestamp, datetime)
    
    def test_analyze_audio_empty(self, analyzer):
        """Test audio analysis with empty audio."""
        empty_audio = np.array([], dtype=np.float32)
        report = analyzer.analyze_audio(empty_audio, 16000)
        
        assert report.clarity_score == 0.0
        assert report.snr == 0.0
        assert report.rms_level == 0.0
        assert report.is_silent
        assert not report.has_speech
    
    def test_analyze_audio_silent(self, analyzer):
        """Test audio analysis with silent audio."""
        silent_audio = np.zeros(16000, dtype=np.float32)
        report = analyzer.analyze_audio(silent_audio, 16000)
        
        assert report.is_silent
        assert not report.has_speech
        assert report.rms_level < 0.01
    
    def test_analyze_audio_noisy(self, analyzer):
        """Test audio analysis with noisy audio."""
        # Pure noise
        noisy_audio = np.random.randn(16000).astype(np.float32) * 0.1
        report = analyzer.analyze_audio(noisy_audio, 16000)
        
        assert isinstance(report, AudioQualityReport)
        # Noisy audio should have lower clarity
        assert report.clarity_score < 0.8
    
    def test_analyze_audio_high_quality(self, analyzer):
        """Test audio analysis with high quality audio."""
        # Clean speech-like signal with some noise for realistic SNR
        t = np.linspace(0, 1, 16000)
        clean_audio = (
            0.5 * np.sin(2 * np.pi * 500 * t) +
            0.3 * np.sin(2 * np.pi * 1500 * t)
        )
        # Add minimal noise to make SNR calculation realistic
        clean_audio += 0.01 * np.random.randn(len(clean_audio))
        clean_audio = clean_audio.astype(np.float32)
        
        report = analyzer.analyze_audio(clean_audio, 16000)
        
        assert report.clarity_score > 0.5
        # SNR calculation depends on VAD threshold, so just check it's positive
        assert report.snr > 0
        # Check that audio is not silent
        assert not report.is_silent
    
    def test_analyze_audio_int16_conversion(self, analyzer):
        """Test audio analysis with int16 audio data."""
        # Create int16 audio
        audio_int16 = (np.random.randn(16000) * 10000).astype(np.int16)
        report = analyzer.analyze_audio(audio_int16, 16000)
        
        assert isinstance(report, AudioQualityReport)
        assert 0.0 <= report.clarity_score <= 1.0
    
    # ==================== Transcription Quality Analysis Tests ====================
    
    def test_analyze_transcription_basic(self, analyzer, sample_transcription_result):
        """Test basic transcription quality analysis."""
        report = analyzer.analyze_transcription(sample_transcription_result)
        
        assert isinstance(report, TranscriptionQualityReport)
        assert 0.0 <= report.fluency_score <= 1.0
        assert 0.0 <= report.confidence_avg <= 1.0
        assert 0.0 <= report.confidence_min <= 1.0
        assert 0.0 <= report.confidence_max <= 1.0
        assert report.speech_rate >= 0
        assert 0.0 <= report.hallucination_risk <= 1.0
        assert report.segment_count == 3
    
    def test_analyze_transcription_empty(self, analyzer):
        """Test transcription analysis with no segments."""
        empty_result = TranscriptionResult(
            text="",
            segments=[],
            language="vi",
            confidence=0.0,
            processing_time=0.0,
            engine_name="test_engine",
        )
        
        report = analyzer.analyze_transcription(empty_result)
        
        assert report.fluency_score == 0.0
        assert report.confidence_avg == 0.0
        assert report.segment_count == 0
    
    def test_analyze_transcription_low_confidence(self, analyzer):
        """Test transcription analysis with low confidence segments."""
        segments = [
            TranscriptionSegment(
                text="test",
                start_time=0.0,
                end_time=1.0,
                confidence=0.3,
                language="vi",
            ),
            TranscriptionSegment(
                text="low",
                start_time=1.0,
                end_time=2.0,
                confidence=0.2,
                language="vi",
            ),
        ]
        
        result = TranscriptionResult(
            text="test low",
            segments=segments,
            language="vi",
            confidence=0.25,
            processing_time=0.5,
            engine_name="test_engine",
        )
        
        report = analyzer.analyze_transcription(result)
        
        assert report.confidence_avg < 0.5
        assert report.hallucination_risk > 0.3  # Should detect high risk
    
    def test_analyze_transcription_high_quality(self, analyzer):
        """Test transcription analysis with high quality segments."""
        segments = [
            TranscriptionSegment(
                text="This is a test",
                start_time=0.0,
                end_time=1.0,
                confidence=0.95,
                language="vi",
            ),
            TranscriptionSegment(
                text="of high quality",
                start_time=1.1,
                end_time=2.0,
                confidence=0.98,
                language="vi",
            ),
            TranscriptionSegment(
                text="transcription",
                start_time=2.1,
                end_time=3.0,
                confidence=0.96,
                language="vi",
            ),
        ]
        
        result = TranscriptionResult(
            text="This is a test of high quality transcription",
            segments=segments,
            language="vi",
            confidence=0.96,
            processing_time=0.5,
            engine_name="test_engine",
        )
        
        report = analyzer.analyze_transcription(result)
        
        assert report.confidence_avg > 0.9
        assert report.fluency_score > 0.5
        assert report.hallucination_risk < 0.3
    
    def test_analyze_transcription_repetitive_text(self, analyzer):
        """Test transcription analysis with repetitive text (hallucination indicator)."""
        segments = [
            TranscriptionSegment(
                text="test test test",
                start_time=0.0,
                end_time=1.0,
                confidence=0.7,
                language="vi",
            ),
            TranscriptionSegment(
                text="test test test",
                start_time=1.0,
                end_time=2.0,
                confidence=0.7,
                language="vi",
            ),
        ]
        
        result = TranscriptionResult(
            text="test test test test test test",
            segments=segments,
            language="vi",
            confidence=0.7,
            processing_time=0.5,
            engine_name="test_engine",
        )
        
        report = analyzer.analyze_transcription(result)
        
        # Repetitive text should increase hallucination risk
        assert report.hallucination_risk > 0.2
    
    # ==================== Recommendations Tests ====================
    
    def test_get_recommendations_good_quality(self, analyzer):
        """Test recommendations for good quality audio and transcription."""
        audio_report = AudioQualityReport(
            clarity_score=0.8,
            snr=25.0,
            rms_level=0.3,
            zero_crossing_rate=0.1,
            spectral_centroid=1500.0,
            is_silent=False,
            has_speech=True,
        )
        
        transcription_report = TranscriptionQualityReport(
            fluency_score=0.85,
            confidence_avg=0.9,
            confidence_min=0.8,
            confidence_max=0.95,
            speech_rate=150.0,
            hallucination_risk=0.1,
            segment_count=5,
        )
        
        recommendations = analyzer.get_recommendations(audio_report, transcription_report)
        
        assert len(recommendations) > 0
        assert any("tốt" in rec for rec in recommendations)  # Should mention good quality
    
    def test_get_recommendations_poor_audio(self, analyzer):
        """Test recommendations for poor audio quality."""
        audio_report = AudioQualityReport(
            clarity_score=0.3,
            snr=5.0,
            rms_level=0.02,
            zero_crossing_rate=0.1,
            spectral_centroid=1500.0,
            is_silent=False,
            has_speech=True,
        )
        
        transcription_report = TranscriptionQualityReport(
            fluency_score=0.7,
            confidence_avg=0.8,
            confidence_min=0.7,
            confidence_max=0.9,
            speech_rate=150.0,
            hallucination_risk=0.2,
            segment_count=5,
        )
        
        recommendations = analyzer.get_recommendations(audio_report, transcription_report)
        
        assert len(recommendations) > 0
        # Should recommend audio improvements
        assert any("SNR" in rec or "âm thanh" in rec for rec in recommendations)
    
    def test_get_recommendations_high_hallucination_risk(self, analyzer):
        """Test recommendations for high hallucination risk."""
        audio_report = AudioQualityReport(
            clarity_score=0.7,
            snr=20.0,
            rms_level=0.3,
            zero_crossing_rate=0.1,
            spectral_centroid=1500.0,
            is_silent=False,
            has_speech=True,
        )
        
        transcription_report = TranscriptionQualityReport(
            fluency_score=0.5,
            confidence_avg=0.4,
            confidence_min=0.2,
            confidence_max=0.6,
            speech_rate=150.0,
            hallucination_risk=0.7,
            segment_count=5,
        )
        
        recommendations = analyzer.get_recommendations(audio_report, transcription_report)
        
        assert len(recommendations) > 0
        # Should warn about hallucination risk
        assert any("hallucination" in rec for rec in recommendations)
    
    def test_get_recommendations_silent_audio(self, analyzer):
        """Test recommendations for silent audio."""
        audio_report = AudioQualityReport(
            clarity_score=0.0,
            snr=0.0,
            rms_level=0.005,
            zero_crossing_rate=0.0,
            spectral_centroid=0.0,
            is_silent=True,
            has_speech=False,
        )
        
        transcription_report = TranscriptionQualityReport(
            fluency_score=0.0,
            confidence_avg=0.0,
            confidence_min=0.0,
            confidence_max=0.0,
            speech_rate=0.0,
            hallucination_risk=0.5,
            segment_count=0,
        )
        
        recommendations = analyzer.get_recommendations(audio_report, transcription_report)
        
        assert len(recommendations) > 0
        # Should mention silent audio or microphone issue
        assert any("im lặng" in rec or "microphone" in rec for rec in recommendations)
    
    # ==================== Comprehensive Quality Report Tests ====================
    
    def test_create_quality_report(self, analyzer, sample_audio, sample_transcription_result):
        """Test creating comprehensive quality report."""
        report = analyzer.create_quality_report(
            sample_audio,
            16000,
            sample_transcription_result,
        )
        
        assert isinstance(report, QualityReport)
        assert isinstance(report.audio_quality, AudioQualityReport)
        assert isinstance(report.transcription_quality, TranscriptionQualityReport)
        assert len(report.recommendations) > 0
        assert 0.0 <= report.overall_score <= 1.0
    
    def test_overall_score_calculation(self, analyzer, sample_audio, sample_transcription_result):
        """Test overall score calculation."""
        report = analyzer.create_quality_report(
            sample_audio,
            16000,
            sample_transcription_result,
        )
        
        # Overall score should be weighted average
        expected_score = (
            report.audio_quality.clarity_score * 0.4 +
            report.transcription_quality.fluency_score * 0.3 +
            report.transcription_quality.confidence_avg * 0.3
        )
        
        assert abs(report.overall_score - expected_score) < 0.01
    
    # ==================== Edge Cases and Error Handling ====================
    
    def test_analyze_audio_very_short(self, analyzer):
        """Test audio analysis with very short audio."""
        # Create short audio with reasonable amplitude
        short_audio = (np.random.randn(100) * 0.3).astype(np.float32)
        report = analyzer.analyze_audio(short_audio, 16000)
        
        assert isinstance(report, AudioQualityReport)
        assert 0.0 <= report.clarity_score <= 1.0
        # Very short audio should still produce valid metrics
        assert report.snr > 0
    
    def test_analyze_transcription_single_segment(self, analyzer):
        """Test transcription analysis with single segment."""
        segment = TranscriptionSegment(
            text="test",
            start_time=0.0,
            end_time=1.0,
            confidence=0.8,
            language="vi",
        )
        
        result = TranscriptionResult(
            text="test",
            segments=[segment],
            language="vi",
            confidence=0.8,
            processing_time=0.5,
            engine_name="test_engine",
        )
        
        report = analyzer.analyze_transcription(result)
        
        assert isinstance(report, TranscriptionQualityReport)
        assert report.segment_count == 1
    
    def test_custom_thresholds(self):
        """Test analyzer with custom thresholds."""
        analyzer = QualityAnalyzer(
            clarity_threshold=0.8,
            fluency_threshold=0.7,
            min_confidence=0.6,
        )
        
        assert analyzer.clarity_threshold == 0.8
        assert analyzer.fluency_threshold == 0.7
        assert analyzer.min_confidence == 0.6
    
    def test_hallucination_risk_with_metadata(self, analyzer):
        """Test hallucination risk calculation with no_speech_prob metadata."""
        segments = [
            TranscriptionSegment(
                text="test",
                start_time=0.0,
                end_time=1.0,
                confidence=0.7,
                language="vi",
            ),
        ]
        
        result = TranscriptionResult(
            text="test",
            segments=segments,
            language="vi",
            confidence=0.7,
            processing_time=0.5,
            engine_name="test_engine",
            metadata={"no_speech_prob": 0.8},  # High no-speech probability
        )
        
        report = analyzer.analyze_transcription(result)
        
        # High no_speech_prob should increase hallucination risk
        # The risk is averaged across multiple factors, so check it's elevated
        assert report.hallucination_risk > 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
