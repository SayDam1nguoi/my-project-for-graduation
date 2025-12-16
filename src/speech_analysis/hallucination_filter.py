"""
Hallucination Filter for Whisper Transcription

Filters out common hallucinations that Whisper generates when there's no actual speech,
such as "subscribe to my channel", "thanks for watching", etc.
"""

import re
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class HallucinationFilter:
    """
    Filters out hallucinated text from Whisper transcriptions.
    
    Common hallucinations occur when:
    - No actual speech in audio
    - Very low volume audio
    - Background noise only
    """
    
    def __init__(self):
        """Initialize hallucination filter with common patterns."""
        
        # Vietnamese hallucination patterns
        self.vietnamese_patterns = [
            # Subscribe/Like patterns
            r'.*đăng ký.*kênh.*',
            r'.*subscribe.*channel.*',
            r'.*like.*share.*',
            r'.*theo dõi.*kênh.*',
            r'.*nhấn.*chuông.*',
            r'.*bấm.*subscribe.*',
            
            # Thanks/Greeting patterns
            r'.*cảm ơn.*đã xem.*',
            r'.*thanks.*watching.*',
            r'.*xin chào.*các bạn.*',
            r'.*chào mừng.*đến với.*',
            r'.*hẹn gặp lại.*',
            
            # Common filler phrases
            r'.*video.*hôm nay.*',
            r'.*trong.*video.*này.*',
            r'.*ở.*video.*tiếp theo.*',
            
            # Repetitive patterns (same phrase 3+ times)
            r'(.{10,})\1{2,}',
            
            # Very short transcriptions (likely noise)
            r'^.{1,5}$',
            
            # Music/sound descriptions
            r'.*\[.*âm nhạc.*\].*',
            r'.*\[.*music.*\].*',
            r'.*\[.*tiếng.*\].*',
        ]
        
        # English hallucination patterns
        self.english_patterns = [
            r'.*subscribe.*channel.*',
            r'.*like.*comment.*',
            r'.*thanks.*watching.*',
            r'.*see you.*next.*',
            r'.*welcome.*back.*',
            r'.*don\'t forget.*',
            r'.*make sure.*subscribe.*',
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.vietnamese_patterns + self.english_patterns
        ]
        
        # Suspicious keywords (if text contains many of these, likely hallucination)
        self.suspicious_keywords = [
            'subscribe', 'đăng ký', 'like', 'share', 'kênh', 'channel',
            'video', 'cảm ơn', 'thanks', 'xin chào', 'chào mừng'
        ]
    
    def is_hallucination(self, text: str) -> Tuple[bool, str]:
        """
        Check if text is likely a hallucination.
        
        Args:
            text: Transcribed text to check
            
        Returns:
            Tuple of (is_hallucination, reason)
        """
        if not text or not text.strip():
            return True, "Empty text"
        
        text_lower = text.lower().strip()
        
        # Check against patterns
        for pattern in self.compiled_patterns:
            if pattern.match(text_lower):
                return True, f"Matched hallucination pattern: {pattern.pattern[:50]}..."
        
        # Check for excessive suspicious keywords
        keyword_count = sum(1 for keyword in self.suspicious_keywords if keyword in text_lower)
        if keyword_count >= 3:
            return True, f"Too many suspicious keywords ({keyword_count})"
        
        # Check for very repetitive text
        words = text_lower.split()
        if len(words) > 5:
            unique_words = set(words)
            repetition_ratio = len(words) / len(unique_words)
            if repetition_ratio > 3.0:
                return True, f"Highly repetitive (ratio: {repetition_ratio:.1f})"
        
        return False, ""
    
    def filter_text(self, text: str) -> Tuple[str, bool, str]:
        """
        Filter hallucinations from text.
        
        Args:
            text: Original transcribed text
            
        Returns:
            Tuple of (filtered_text, was_filtered, reason)
        """
        is_halluc, reason = self.is_hallucination(text)
        
        if is_halluc:
            logger.warning(f"Filtered hallucination: '{text[:100]}...' - Reason: {reason}")
            return "", True, reason
        
        return text, False, ""
    
    def filter_segments(self, segments: List[dict]) -> Tuple[List[dict], int]:
        """
        Filter hallucinations from a list of segments.
        
        Args:
            segments: List of segment dicts with 'text' field
            
        Returns:
            Tuple of (filtered_segments, num_filtered)
        """
        filtered_segments = []
        num_filtered = 0
        
        for segment in segments:
            text = segment.get('text', '')
            filtered_text, was_filtered, reason = self.filter_text(text)
            
            if not was_filtered:
                filtered_segments.append(segment)
            else:
                num_filtered += 1
                logger.debug(f"Filtered segment: {text[:50]}... - {reason}")
        
        if num_filtered > 0:
            logger.info(f"Filtered {num_filtered} hallucinated segments")
        
        return filtered_segments, num_filtered
    
    def get_confidence_penalty(self, text: str) -> float:
        """
        Get confidence penalty for potentially hallucinated text.
        
        Returns:
            Penalty multiplier (0.0 to 1.0)
            - 1.0 = no penalty (clean text)
            - 0.0 = maximum penalty (likely hallucination)
        """
        if not text or not text.strip():
            return 0.0
        
        text_lower = text.lower().strip()
        
        # Count suspicious indicators
        penalty_score = 1.0
        
        # Check keywords
        keyword_count = sum(1 for keyword in self.suspicious_keywords if keyword in text_lower)
        if keyword_count > 0:
            penalty_score *= (1.0 - (keyword_count * 0.15))  # -15% per keyword
        
        # Check patterns (lighter penalty)
        for pattern in self.compiled_patterns:
            if pattern.match(text_lower):
                penalty_score *= 0.5  # -50% if matches pattern
                break
        
        return max(0.0, min(1.0, penalty_score))


# Singleton instance
_filter_instance = None


def get_hallucination_filter() -> HallucinationFilter:
    """Get singleton hallucination filter instance."""
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = HallucinationFilter()
    return _filter_instance


def filter_hallucination(text: str) -> Tuple[str, bool, str]:
    """
    Convenience function to filter hallucinations.
    
    Args:
        text: Text to filter
        
    Returns:
        Tuple of (filtered_text, was_filtered, reason)
    """
    filter_obj = get_hallucination_filter()
    return filter_obj.filter_text(text)
