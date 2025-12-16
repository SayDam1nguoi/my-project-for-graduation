"""Hallucination detection and filtering implementation."""

import re
from typing import List, Dict, Any, Set
from dataclasses import dataclass, field

from ..interfaces.processing import (
    IHallucinationFilter,
    HallucinationPattern,
    FilterStats,
)
from ..models.transcription import TranscriptionSegment


@dataclass
class HallucinationFilterConfig:
    """Configuration for hallucination filter."""
    
    # Confidence-based filtering
    min_confidence: float = 0.6
    early_segment_confidence: float = 0.8  # Stricter for first 3 seconds
    early_segment_threshold: float = 3.0  # Seconds
    
    # Compression ratio filtering
    compression_ratio_threshold: float = 2.0
    
    # No-speech probability filtering
    no_speech_threshold: float = 0.6
    
    # Pattern matching
    enable_exact_matching: bool = True
    enable_partial_matching: bool = True
    enable_keyword_matching: bool = True
    
    # Partial matching thresholds
    partial_match_threshold: int = 2  # Need 2+ partial matches
    keyword_match_threshold: int = 2  # Need 2+ keyword matches
    strong_indicator_threshold: int = 1  # Need 1+ strong indicator
    
    # Text validation
    min_text_length: int = 3
    
    # Language
    language: str = "vi"


class HallucinationFilter(IHallucinationFilter):
    """
    Comprehensive hallucination detection and filtering.
    
    Implements multiple detection strategies:
    1. Confidence-based filtering
    2. Compression ratio filtering
    3. No-speech probability filtering
    4. Pattern matching (exact, partial, keywords)
    5. Temporal filtering (early segments)
    6. Text validation
    """
    
    def __init__(self, config: HallucinationFilterConfig = None):
        """
        Initialize hallucination filter.
        
        Args:
            config: Filter configuration.
        """
        self.config = config or HallucinationFilterConfig()
        self._stats = FilterStats()
        
        # Pattern databases
        self._exact_patterns: Set[str] = set()
        self._partial_patterns: Set[str] = set()
        self._keyword_patterns: Set[str] = set()
        self._strong_indicators: Set[str] = set()
        
        # Initialize with default Vietnamese patterns
        self._initialize_default_patterns()
    
    def _initialize_default_patterns(self) -> None:
        """Initialize default hallucination patterns for Vietnamese."""
        # Exact phrase matches (30+ patterns)
        exact_patterns = [
            'hãy subscribe cho kênh ghiền mì gõ để không bỏ lỡ những video hấp dẫn',
            'hãy subscribe cho kênh ghiền mì gõ',
            'ghiền mì gõ để không bỏ lỡ những video hấp dẫn',
            'để không bỏ lỡ những video hấp dẫn',
            'đăng ký kênh để không bỏ lỡ video mới',
            'nhấn subscribe và bật chuông thông báo',
            'cảm ơn các bạn đã xem video',
            'cảm ơn đã theo dõi',
            'hẹn gặp lại trong video sau',
            'like và share video này',
            'đừng quên like và share',
            'bấm like và đăng ký kênh',
            'nhấn nút đăng ký bên dưới',
            'theo dõi kênh để xem thêm video',
            'subscribe kênh để cập nhật video mới',
            'bật chuông thông báo để không bỏ lỡ',
            'đăng ký và bật chuông thông báo',
            'cảm ơn bạn đã xem video của tôi',
            'hẹn gặp lại các bạn trong video tiếp theo',
            'chúc các bạn xem video vui vẻ',
            'xin chào và hẹn gặp lại',
            'see you in the next video',
            'thanks for watching',
            'please subscribe',
            'hit the bell icon',
            'turn on notifications',
            'like comment and subscribe',
            'smash that like button',
            'don\'t forget to subscribe',
            'check out my other videos',
            'link in the description',
        ]
        
        for pattern in exact_patterns:
            self._exact_patterns.add(pattern.lower())
        
        # Partial phrase matches (17+ patterns)
        partial_patterns = [
            'subscribe cho kênh',
            'đăng ký kênh',
            'bỏ lỡ những video',
            'video hấp dẫn',
            'ghiền mì gõ',
            'bật chuông thông báo',
            'like và share',
            'like share',
            'chia sẻ video',
            'theo dõi kênh',
            'cảm ơn đã xem',
            'hẹn gặp lại',
            'video tiếp theo',
            'video mới',
            'bấm subscribe',
            'nhấn subscribe',
            'đăng ký ngay',
        ]
        
        for pattern in partial_patterns:
            self._partial_patterns.add(pattern.lower())
        
        # Strong indicator keywords (20+ keywords)
        strong_indicators = [
            'subscribe',
            'đăng ký kênh',
            'ghiền mì gõ',
            'bấm chuông',
            'nhấn subscribe',
            'like và share',
            'like share',
            'bật thông báo',
            'notification',
            'bell icon',
            'smash that like',
            'hit the bell',
            'turn on notifications',
            'check out my channel',
            'link in description',
            'patreon',
            'merch',
            'sponsor',
            'affiliate link',
            'promo code',
        ]
        
        for indicator in strong_indicators:
            self._strong_indicators.add(indicator.lower())
        
        # General hallucination keywords
        keyword_patterns = [
            'subscribe',
            'đăng ký kênh',
            'đăng kí kênh',
            'theo dõi kênh',
            'bấm subscribe',
            'nhấn subscribe',
            'channel',
            'like và share',
            'like share',
            'chia sẻ video',
            'bấm chuông',
            'nhấn chuông',
            'bật thông báo',
            'notification',
            'ghiền mì gõ',
            'ghiền mì',
            'cảm ơn đã xem',
            'cảm ơn các bạn đã xem',
            'hẹn gặp lại',
            'see you',
            'bye bye',
            '[âm nhạc]',
            '[music]',
            '[tiếng vỗ tay]',
            '[applause]',
            '[cười]',
            '[laughter]',
        ]
        
        for keyword in keyword_patterns:
            self._keyword_patterns.add(keyword.lower())
    
    def is_hallucination(self, text: str, metadata: Dict[str, Any]) -> bool:
        """
        Check if text is likely hallucination.
        
        Args:
            text: Transcribed text to check.
            metadata: Additional metadata (confidence, no_speech_prob, etc.).
            
        Returns:
            True if likely hallucination, False otherwise.
        """
        if not text or not text.strip():
            return True
        
        text_lower = text.lower().strip()
        
        # Extract metadata
        confidence = metadata.get('confidence', 1.0)
        no_speech_prob = metadata.get('no_speech_prob', 0.0)
        compression_ratio = metadata.get('compression_ratio', 1.0)
        start_time = metadata.get('start_time', 0.0)
        
        # 1. Confidence-based filtering
        if start_time < self.config.early_segment_threshold:
            # Stricter confidence for early segments
            if confidence < self.config.early_segment_confidence:
                self._record_filter('early_segment_low_confidence')
                return True
        else:
            # Normal confidence threshold
            if confidence < self.config.min_confidence:
                self._record_filter('low_confidence')
                return True
        
        # 2. Compression ratio filtering
        if compression_ratio > self.config.compression_ratio_threshold:
            self._record_filter('high_compression_ratio')
            return True
        
        # 3. No-speech probability filtering
        if no_speech_prob > self.config.no_speech_threshold:
            self._record_filter('high_no_speech_prob')
            return True
        
        # 4. Text validation
        if len(text_lower) < self.config.min_text_length:
            self._record_filter('text_too_short')
            return True
        
        # Check if text is only punctuation/symbols
        if all(not c.isalnum() for c in text_lower):
            self._record_filter('no_alphanumeric')
            return True
        
        # 5. Exact pattern matching
        if self.config.enable_exact_matching:
            for pattern in self._exact_patterns:
                if pattern in text_lower:
                    self._record_filter('exact_pattern_match')
                    return True
        
        # 6. Partial pattern matching
        if self.config.enable_partial_matching:
            partial_match_count = sum(
                1 for pattern in self._partial_patterns
                if pattern in text_lower
            )
            if partial_match_count >= self.config.partial_match_threshold:
                self._record_filter('partial_pattern_match')
                return True
        
        # 7. Strong indicator matching (single match is enough)
        for indicator in self._strong_indicators:
            if indicator in text_lower:
                self._record_filter('strong_indicator_match')
                return True
        
        # 8. Keyword matching (need multiple matches)
        if self.config.enable_keyword_matching:
            keyword_count = sum(
                1 for keyword in self._keyword_patterns
                if keyword in text_lower
            )
            if keyword_count >= self.config.keyword_match_threshold:
                self._record_filter('keyword_match')
                return True
        
        return False
    
    def filter_segments(
        self,
        segments: List[TranscriptionSegment],
    ) -> List[TranscriptionSegment]:
        """
        Filter hallucination segments from list.
        
        Args:
            segments: List of transcription segments.
            
        Returns:
            Filtered list of segments.
        """
        filtered_segments = []
        
        for segment in segments:
            self._stats.total_segments += 1
            
            # Prepare metadata for hallucination check
            metadata = {
                'confidence': segment.confidence,
                'start_time': segment.start_time,
                'no_speech_prob': segment.metadata.get('no_speech_prob', 0.0),
                'compression_ratio': segment.metadata.get('compression_ratio', 1.0),
            }
            
            # Check if hallucination
            if not self.is_hallucination(segment.text, metadata):
                filtered_segments.append(segment)
            else:
                self._stats.filtered_segments += 1
        
        return filtered_segments
    
    def add_pattern(self, pattern: HallucinationPattern) -> None:
        """
        Add hallucination pattern to filter.
        
        Args:
            pattern: Hallucination pattern to add.
        """
        pattern_lower = pattern.pattern.lower()
        
        if pattern.pattern_type == "exact":
            self._exact_patterns.add(pattern_lower)
        elif pattern.pattern_type == "partial":
            self._partial_patterns.add(pattern_lower)
        elif pattern.pattern_type == "keyword":
            self._keyword_patterns.add(pattern_lower)
        elif pattern.pattern_type == "strong_indicator":
            self._strong_indicators.add(pattern_lower)
        else:
            raise ValueError(f"Unknown pattern type: {pattern.pattern_type}")
    
    def remove_pattern(self, pattern: str) -> None:
        """
        Remove hallucination pattern from filter.
        
        Args:
            pattern: Pattern string to remove.
        """
        pattern_lower = pattern.lower()
        
        # Try to remove from all pattern sets
        self._exact_patterns.discard(pattern_lower)
        self._partial_patterns.discard(pattern_lower)
        self._keyword_patterns.discard(pattern_lower)
        self._strong_indicators.discard(pattern_lower)
    
    def get_filter_stats(self) -> FilterStats:
        """
        Get statistics about filtered content.
        
        Returns:
            Filter statistics.
        """
        return self._stats
    
    def reset_stats(self) -> None:
        """Reset filter statistics."""
        self._stats = FilterStats()
    
    def _record_filter(self, reason: str) -> None:
        """
        Record a filter event with reason.
        
        Args:
            reason: Reason for filtering.
        """
        if reason not in self._stats.filter_reasons:
            self._stats.filter_reasons[reason] = 0
        self._stats.filter_reasons[reason] += 1
    
    def get_pattern_counts(self) -> Dict[str, int]:
        """
        Get count of patterns in each category.
        
        Returns:
            Dictionary with pattern counts.
        """
        return {
            'exact_patterns': len(self._exact_patterns),
            'partial_patterns': len(self._partial_patterns),
            'keyword_patterns': len(self._keyword_patterns),
            'strong_indicators': len(self._strong_indicators),
        }
    
    def clear_patterns(self, pattern_type: str = None) -> None:
        """
        Clear patterns of specified type or all patterns.
        
        Args:
            pattern_type: Type of patterns to clear, or None for all.
        """
        if pattern_type is None or pattern_type == "exact":
            self._exact_patterns.clear()
        if pattern_type is None or pattern_type == "partial":
            self._partial_patterns.clear()
        if pattern_type is None or pattern_type == "keyword":
            self._keyword_patterns.clear()
        if pattern_type is None or pattern_type == "strong_indicator":
            self._strong_indicators.clear()
