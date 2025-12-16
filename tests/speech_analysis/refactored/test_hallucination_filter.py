"""Tests for hallucination filter implementation."""

import pytest
from src.speech_analysis.refactored.implementations.hallucination_filter import (
    HallucinationFilter,
    HallucinationFilterConfig,
)
from src.speech_analysis.refactored.interfaces.processing import (
    HallucinationPattern,
)
from src.speech_analysis.refactored.models.transcription import (
    TranscriptionSegment,
)


class TestHallucinationFilter:
    """Test hallucination filter functionality."""
    
    def test_filter_initialization(self):
        """Test filter initializes with default patterns."""
        filter = HallucinationFilter()
        
        pattern_counts = filter.get_pattern_counts()
        assert pattern_counts['exact_patterns'] > 0
        assert pattern_counts['partial_patterns'] > 0
        assert pattern_counts['keyword_patterns'] > 0
        assert pattern_counts['strong_indicators'] > 0
    
    def test_exact_pattern_matching(self):
        """Test exact phrase hallucination detection."""
        filter = HallucinationFilter()
        
        # Known hallucination phrases
        hallucinations = [
            "Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn",
            "Hãy subscribe cho kênh Ghiền Mì Gõ",
            "Đăng ký kênh để không bỏ lỡ video mới",
            "Nhấn subscribe và bật chuông thông báo",
            "Cảm ơn các bạn đã xem video",
        ]
        
        for text in hallucinations:
            metadata = {'confidence': 0.9, 'start_time': 5.0}
            assert filter.is_hallucination(text, metadata), f"Failed to detect: {text}"
    
    def test_partial_pattern_matching(self):
        """Test partial phrase hallucination detection."""
        filter = HallucinationFilter()
        
        # Text with multiple partial matches
        text = "Subscribe cho kênh và bỏ lỡ những video hấp dẫn"
        metadata = {'confidence': 0.9, 'start_time': 5.0}
        
        # Should be detected due to multiple partial matches
        assert filter.is_hallucination(text, metadata)
    
    def test_strong_indicator_matching(self):
        """Test strong indicator keyword detection."""
        filter = HallucinationFilter()
        
        # Single strong indicator should trigger
        strong_indicators = [
            "Subscribe cho kênh nhé",
            "Đăng ký kênh của tôi",
            "Ghiền mì gõ rất hay",
            "Like và share video này",
        ]
        
        for text in strong_indicators:
            metadata = {'confidence': 0.9, 'start_time': 5.0}
            assert filter.is_hallucination(text, metadata), f"Failed to detect: {text}"
    
    def test_valid_speech_not_filtered(self):
        """Test that valid speech is not filtered."""
        filter = HallucinationFilter()
        
        # Valid Vietnamese speech
        valid_texts = [
            "Xin chào, tôi tên là Minh",
            "Hôm nay chúng ta sẽ học về Python",
            "Tôi đang làm việc tại công ty ABC",
            "Bạn có thể giúp tôi không?",
            "Cảm ơn bạn đã giúp đỡ",
            "Tôi rất vui được gặp bạn",
        ]
        
        for text in valid_texts:
            metadata = {'confidence': 0.9, 'start_time': 5.0}
            assert not filter.is_hallucination(text, metadata), f"Incorrectly filtered: {text}"
    
    def test_confidence_based_filtering(self):
        """Test confidence-based filtering."""
        filter = HallucinationFilter()
        
        text = "Xin chào các bạn"
        
        # Low confidence should be filtered
        metadata = {'confidence': 0.5, 'start_time': 5.0}
        assert filter.is_hallucination(text, metadata)
        
        # High confidence should pass
        metadata = {'confidence': 0.9, 'start_time': 5.0}
        assert not filter.is_hallucination(text, metadata)
    
    def test_early_segment_filtering(self):
        """Test stricter filtering for early segments."""
        filter = HallucinationFilter()
        
        text = "Xin chào các bạn"
        
        # Early segment with medium confidence should be filtered
        metadata = {'confidence': 0.7, 'start_time': 1.0}
        assert filter.is_hallucination(text, metadata)
        
        # Later segment with same confidence should pass
        metadata = {'confidence': 0.7, 'start_time': 5.0}
        assert not filter.is_hallucination(text, metadata)
    
    def test_compression_ratio_filtering(self):
        """Test compression ratio filtering."""
        filter = HallucinationFilter()
        
        text = "Xin chào các bạn"
        
        # High compression ratio indicates repetition
        metadata = {
            'confidence': 0.9,
            'start_time': 5.0,
            'compression_ratio': 2.5,
        }
        assert filter.is_hallucination(text, metadata)
        
        # Normal compression ratio should pass
        metadata = {
            'confidence': 0.9,
            'start_time': 5.0,
            'compression_ratio': 1.5,
        }
        assert not filter.is_hallucination(text, metadata)
    
    def test_no_speech_probability_filtering(self):
        """Test no-speech probability filtering."""
        filter = HallucinationFilter()
        
        text = "Xin chào các bạn"
        
        # High no_speech_prob should be filtered
        metadata = {
            'confidence': 0.9,
            'start_time': 5.0,
            'no_speech_prob': 0.7,
        }
        assert filter.is_hallucination(text, metadata)
        
        # Low no_speech_prob should pass
        metadata = {
            'confidence': 0.9,
            'start_time': 5.0,
            'no_speech_prob': 0.3,
        }
        assert not filter.is_hallucination(text, metadata)
    
    def test_text_validation(self):
        """Test text validation (length, alphanumeric)."""
        filter = HallucinationFilter()
        
        metadata = {'confidence': 0.9, 'start_time': 5.0}
        
        # Too short
        assert filter.is_hallucination("ab", metadata)
        
        # Only punctuation
        assert filter.is_hallucination("...", metadata)
        assert filter.is_hallucination("!!!", metadata)
        
        # Empty
        assert filter.is_hallucination("", metadata)
        assert filter.is_hallucination("   ", metadata)
    
    def test_filter_segments(self):
        """Test filtering a list of segments."""
        filter = HallucinationFilter()
        
        segments = [
            TranscriptionSegment(
                text="Xin chào các bạn",
                start_time=0.0,
                end_time=2.0,
                confidence=0.9,
                language="vi",
            ),
            TranscriptionSegment(
                text="Hãy subscribe cho kênh",
                start_time=2.0,
                end_time=4.0,
                confidence=0.9,
                language="vi",
            ),
            TranscriptionSegment(
                text="Hôm nay chúng ta học Python",
                start_time=4.0,
                end_time=6.0,
                confidence=0.9,
                language="vi",
            ),
        ]
        
        filtered = filter.filter_segments(segments)
        
        # Should filter out the hallucination
        assert len(filtered) == 2
        assert "subscribe" not in filtered[0].text.lower()
        assert "subscribe" not in filtered[1].text.lower()
    
    def test_add_custom_pattern(self):
        """Test adding custom hallucination patterns."""
        filter = HallucinationFilter()
        
        # Add custom pattern
        pattern = HallucinationPattern(
            pattern="custom hallucination phrase",
            pattern_type="exact",
        )
        filter.add_pattern(pattern)
        
        # Should detect custom pattern
        metadata = {'confidence': 0.9, 'start_time': 5.0}
        assert filter.is_hallucination("Custom hallucination phrase", metadata)
    
    def test_remove_pattern(self):
        """Test removing hallucination patterns."""
        filter = HallucinationFilter()
        
        # Clear all patterns
        filter.clear_patterns()
        
        # Add a single pattern
        pattern = HallucinationPattern(
            pattern="test pattern",
            pattern_type="exact",
        )
        filter.add_pattern(pattern)
        
        # Should detect
        metadata = {'confidence': 0.9, 'start_time': 5.0}
        assert filter.is_hallucination("test pattern", metadata)
        
        # Remove pattern
        filter.remove_pattern("test pattern")
        
        # Should not detect anymore (but will be filtered by low confidence or other rules)
        # So we need high confidence and later time
        metadata = {'confidence': 0.9, 'start_time': 5.0}
        # After removing, it should pass (assuming no other filters catch it)
        # But we cleared all patterns, so it should pass
        assert not filter.is_hallucination("test pattern", metadata)
    
    def test_filter_statistics(self):
        """Test filter statistics tracking."""
        filter = HallucinationFilter()
        filter.reset_stats()
        
        segments = [
            TranscriptionSegment(
                text="Valid speech",
                start_time=0.0,
                end_time=2.0,
                confidence=0.9,
                language="vi",
            ),
            TranscriptionSegment(
                text="Subscribe cho kênh",
                start_time=2.0,
                end_time=4.0,
                confidence=0.9,
                language="vi",
            ),
            TranscriptionSegment(
                text="More valid speech",
                start_time=4.0,
                end_time=6.0,
                confidence=0.9,
                language="vi",
            ),
        ]
        
        filter.filter_segments(segments)
        
        stats = filter.get_filter_stats()
        assert stats.total_segments == 3
        assert stats.filtered_segments == 1
        assert stats.filter_rate == pytest.approx(1/3)
        assert len(stats.filter_reasons) > 0
    
    def test_custom_config(self):
        """Test custom filter configuration."""
        config = HallucinationFilterConfig(
            min_confidence=0.8,
            compression_ratio_threshold=1.5,
            no_speech_threshold=0.5,
        )
        filter = HallucinationFilter(config)
        
        text = "Xin chào các bạn"
        
        # Should be filtered with stricter confidence
        metadata = {'confidence': 0.7, 'start_time': 5.0}
        assert filter.is_hallucination(text, metadata)
        
        # Should pass with high confidence
        metadata = {'confidence': 0.9, 'start_time': 5.0}
        assert not filter.is_hallucination(text, metadata)


class TestHallucinationFilterEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_unicode_handling(self):
        """Test handling of Unicode Vietnamese characters."""
        filter = HallucinationFilter()
        
        # Vietnamese with diacritics
        text = "Xin chào, tôi tên là Nguyễn Văn A"
        metadata = {'confidence': 0.9, 'start_time': 5.0}
        assert not filter.is_hallucination(text, metadata)
    
    def test_mixed_case_patterns(self):
        """Test case-insensitive pattern matching."""
        filter = HallucinationFilter()
        
        # Different cases should all be detected
        cases = [
            "SUBSCRIBE CHO KÊNH",
            "Subscribe Cho Kênh",
            "subscribe cho kênh",
            "SuBsCrIbE cHo KêNh",
        ]
        
        for text in cases:
            metadata = {'confidence': 0.9, 'start_time': 5.0}
            assert filter.is_hallucination(text, metadata), f"Failed: {text}"
    
    def test_whitespace_handling(self):
        """Test handling of extra whitespace."""
        filter = HallucinationFilter()
        
        # Extra whitespace should be handled
        text = "  Subscribe   cho   kênh  "
        metadata = {'confidence': 0.9, 'start_time': 5.0}
        assert filter.is_hallucination(text, metadata)
    
    def test_boundary_confidence_values(self):
        """Test boundary confidence values."""
        filter = HallucinationFilter()
        
        text = "Xin chào các bạn"
        
        # Exactly at threshold
        metadata = {'confidence': 0.6, 'start_time': 5.0}
        assert not filter.is_hallucination(text, metadata)
        
        # Just below threshold
        metadata = {'confidence': 0.59, 'start_time': 5.0}
        assert filter.is_hallucination(text, metadata)
    
    def test_empty_metadata(self):
        """Test handling of missing metadata."""
        filter = HallucinationFilter()
        
        text = "Xin chào các bạn"
        
        # Empty metadata should use defaults
        metadata = {}
        # Should pass with default values
        assert not filter.is_hallucination(text, metadata)
    
    def test_pattern_type_validation(self):
        """Test validation of pattern types."""
        filter = HallucinationFilter()
        
        # Invalid pattern type should raise error
        with pytest.raises(ValueError):
            pattern = HallucinationPattern(
                pattern="test",
                pattern_type="invalid_type",
            )
            filter.add_pattern(pattern)
