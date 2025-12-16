"""
Vietnamese STT Optimizer Module

Provides Vietnamese-specific optimizations for speech-to-text:
- Diacritic validation and correction
- Post-correction for low confidence words
- Vietnamese proper noun handling
- Tone mark validation
- Vietnamese-specific text normalization
"""

import re
import unicodedata
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import logging

from .custom_vocabulary import CustomVocabularyProcessor
from .speech_to_text import TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)


# Vietnamese diacritics and tone marks
VIETNAMESE_VOWELS = "aăâeêioôơuưy"
VIETNAMESE_VOWELS_UPPER = "AĂÂEÊIOÔƠUƯY"
VIETNAMESE_TONE_MARKS = "\u0300\u0301\u0303\u0309\u0323"  # grave, acute, tilde, hook, dot

# Vietnamese diacritic combinations
VIETNAMESE_CHARS = set(
    "aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵ"
    "AÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬEÈÉẺẼẸÊỀẾỂỄỆIÌÍỈĨỊOÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢUÙÚỦŨỤƯỪỨỬỮỰYỲÝỶỸỴ"
    "đĐ"
)


@dataclass
class CorrectionCandidate:
    """Represents a correction candidate for a word."""
    original: str
    corrected: str
    confidence: float
    reason: str


class VietnameseDiacriticValidator:
    """
    Validates and corrects Vietnamese diacritics.
    
    Ensures that Vietnamese text has proper tone marks and diacritics.
    """
    
    def __init__(self):
        """Initialize the validator."""
        self.valid_chars = VIETNAMESE_CHARS | set("abcdefghijklmnopqrstuvwxyz")
        self.valid_chars |= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.valid_chars |= set(" .,!?;:()[]{}\"'-")
        
        logger.info("Vietnamese diacritic validator initialized")
    
    def validate_text(self, text: str) -> Tuple[bool, float]:
        """
        Validate Vietnamese text for proper diacritics.
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (is_valid, accuracy_score)
            accuracy_score is percentage of words with proper diacritics
        """
        if not text:
            return True, 1.0
        
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        # Split into words
        words = re.findall(r'\b\w+\b', text)
        
        if not words:
            return True, 1.0
        
        # Count words with Vietnamese diacritics
        words_with_diacritics = 0
        total_vietnamese_words = 0
        
        for word in words:
            # Check if word contains Vietnamese characters
            has_vietnamese = any(c in VIETNAMESE_CHARS for c in word)
            
            if has_vietnamese:
                total_vietnamese_words += 1
                
                # Check if word has proper diacritics
                if self._has_proper_diacritics(word):
                    words_with_diacritics += 1
        
        # Calculate accuracy
        if total_vietnamese_words == 0:
            # No Vietnamese words, consider valid
            return True, 1.0
        
        accuracy = words_with_diacritics / total_vietnamese_words
        is_valid = accuracy >= 0.98  # 98% threshold from requirements
        
        return is_valid, accuracy
    
    def _has_proper_diacritics(self, word: str) -> bool:
        """
        Check if a word has proper Vietnamese diacritics.
        
        Args:
            word: Word to check
            
        Returns:
            True if word has proper diacritics
        """
        # Normalize
        word = unicodedata.normalize('NFC', word.lower())
        
        # Check if word contains only ASCII letters (no diacritics)
        is_ascii_only = all(ord(c) < 128 for c in word if c.isalpha())
        
        # If word is ASCII-only, check if it looks like Vietnamese
        if is_ascii_only and len(word) > 1:
            # Common Vietnamese patterns that indicate missing diacritics
            vietnamese_patterns = ['ng', 'nh', 'ch', 'th', 'ph', 'tr', 'gi', 'qu', 'kh']
            
            # Common Vietnamese words that are ASCII-only
            common_ascii_words = ['la', 'va', 'ma', 'hay', 'nhu', 'thi', 'se', 'da', 'co']
            
            # If it's a common word, accept it
            if word in common_ascii_words:
                return True
            
            # Check for Vietnamese patterns
            has_vietnamese_pattern = any(pattern in word for pattern in vietnamese_patterns)
            
            # Also check for typical Vietnamese syllable structure
            # Vietnamese words often have specific vowel patterns
            has_vietnamese_vowels = any(v in word for v in ['oa', 'oe', 'uy', 'ua', 'ie', 'uo'])
            
            if has_vietnamese_pattern or has_vietnamese_vowels:
                # Likely a Vietnamese word missing diacritics
                return False
            
            # If it doesn't look Vietnamese, consider it valid (might be English)
            return True
        
        # Check for Vietnamese characters with diacritics
        has_vietnamese_chars = any(c in VIETNAMESE_CHARS for c in word)
        
        # If it has Vietnamese characters, it has proper diacritics
        return has_vietnamese_chars or not is_ascii_only
    
    def calculate_diacritic_accuracy(self, text: str) -> float:
        """
        Calculate diacritic accuracy percentage.
        
        Args:
            text: Text to analyze
            
        Returns:
            Accuracy percentage (0.0 to 1.0)
        """
        _, accuracy = self.validate_text(text)
        return accuracy


class VietnamesePostCorrector:
    """
    Post-correction for low confidence Vietnamese words.
    
    Uses custom vocabulary and language model to correct words
    with low confidence scores.
    """
    
    def __init__(
        self,
        vocabulary_processor: Optional[CustomVocabularyProcessor] = None,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize post-corrector.
        
        Args:
            vocabulary_processor: Custom vocabulary processor
            confidence_threshold: Threshold below which to apply correction
        """
        self.vocabulary_processor = vocabulary_processor
        self.confidence_threshold = confidence_threshold
        
        logger.info(
            f"Vietnamese post-corrector initialized with threshold={confidence_threshold}"
        )
    
    def correct_transcription(
        self,
        result: TranscriptionResult
    ) -> TranscriptionResult:
        """
        Apply post-correction to transcription result.
        
        Args:
            result: Original transcription result
            
        Returns:
            Corrected transcription result
        """
        if not result.segments:
            return result
        
        corrected_segments = []
        corrections_applied = 0
        
        for segment in result.segments:
            # Check if segment needs correction
            if segment.confidence < self.confidence_threshold:
                corrected_segment = self._correct_segment(segment)
                corrected_segments.append(corrected_segment)
                
                if corrected_segment.text != segment.text:
                    corrections_applied += 1
                    logger.debug(
                        f"Corrected: '{segment.text}' -> '{corrected_segment.text}'"
                    )
            else:
                corrected_segments.append(segment)
        
        # Rebuild full text
        full_text = " ".join(seg.text for seg in corrected_segments)
        
        # Create corrected result
        corrected_result = TranscriptionResult(
            text=full_text,
            confidence=result.confidence,
            language=result.language,
            segments=corrected_segments,
            processing_time=result.processing_time
        )
        
        if corrections_applied > 0:
            logger.info(f"Applied {corrections_applied} corrections")
        
        return corrected_result
    
    def _correct_segment(
        self,
        segment: TranscriptionSegment
    ) -> TranscriptionSegment:
        """
        Correct a single segment.
        
        Args:
            segment: Segment to correct
            
        Returns:
            Corrected segment
        """
        text = segment.text
        
        # Apply vocabulary correction if available
        if self.vocabulary_processor and self.vocabulary_processor.has_vocabulary():
            text = self.vocabulary_processor.process_transcription(text)
        
        # Apply diacritic normalization
        text = self._normalize_diacritics(text)
        
        # Create corrected segment
        return TranscriptionSegment(
            text=text,
            start_time=segment.start_time,
            end_time=segment.end_time,
            confidence=segment.confidence
        )
    
    def _normalize_diacritics(self, text: str) -> str:
        """
        Normalize Vietnamese diacritics.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Ensure NFC normalization (canonical composition)
        text = unicodedata.normalize('NFC', text)
        
        return text


class VietnameseSTTOptimizer:
    """
    Main Vietnamese STT optimizer.
    
    Combines all Vietnamese-specific optimizations:
    - Custom vocabulary for proper nouns
    - Diacritic validation
    - Post-correction for low confidence words
    """
    
    def __init__(
        self,
        vocabulary_file: Optional[str] = None,
        confidence_threshold: float = 0.7,
        enable_post_correction: bool = True,
        enable_diacritic_validation: bool = True
    ):
        """
        Initialize Vietnamese STT optimizer.
        
        Args:
            vocabulary_file: Path to Vietnamese vocabulary JSON file
            confidence_threshold: Threshold for post-correction
            enable_post_correction: Enable post-correction
            enable_diacritic_validation: Enable diacritic validation
        """
        # Initialize vocabulary processor
        self.vocabulary_processor = None
        if vocabulary_file:
            try:
                self.vocabulary_processor = CustomVocabularyProcessor(vocabulary_file)
                self._load_vietnamese_vocabulary(vocabulary_file)
                logger.info(f"Loaded Vietnamese vocabulary from {vocabulary_file}")
            except Exception as e:
                logger.warning(f"Failed to load vocabulary: {e}")
        
        # Initialize diacritic validator
        self.diacritic_validator = None
        if enable_diacritic_validation:
            self.diacritic_validator = VietnameseDiacriticValidator()
        
        # Initialize post-corrector
        self.post_corrector = None
        if enable_post_correction:
            self.post_corrector = VietnamesePostCorrector(
                vocabulary_processor=self.vocabulary_processor,
                confidence_threshold=confidence_threshold
            )
        
        self.enable_post_correction = enable_post_correction
        self.enable_diacritic_validation = enable_diacritic_validation
        
        logger.info("Vietnamese STT optimizer initialized")
    
    def _load_vietnamese_vocabulary(self, vocabulary_file: str) -> None:
        """
        Load Vietnamese vocabulary from file.
        
        The file should be in JSON format with categories:
        - proper_nouns: Vietnamese proper nouns
        - technical_terms: Technical terms
        - common_phrases: Common phrases
        - organizations: Organization names
        
        Args:
            vocabulary_file: Path to vocabulary file
        """
        import json
        from pathlib import Path
        
        path = Path(vocabulary_file)
        if not path.exists():
            logger.warning(f"Vocabulary file not found: {vocabulary_file}")
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Add proper nouns with high priority
            if 'proper_nouns' in data:
                for noun in data['proper_nouns']:
                    self.vocabulary_processor.add_word(noun, priority=3)
            
            # Add technical terms
            if 'technical_terms' in data:
                for term in data['technical_terms']:
                    if ' ' in term:
                        self.vocabulary_processor.add_phrase(term, priority=2)
                    else:
                        self.vocabulary_processor.add_word(term, priority=2)
            
            # Add common phrases
            if 'common_phrases' in data:
                for phrase in data['common_phrases']:
                    self.vocabulary_processor.add_phrase(phrase, priority=2)
            
            # Add organizations
            if 'organizations' in data:
                for org in data['organizations']:
                    self.vocabulary_processor.add_phrase(org, priority=3)
            
            logger.info("Vietnamese vocabulary loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Vietnamese vocabulary: {e}")
    
    def optimize_transcription(
        self,
        result: TranscriptionResult
    ) -> TranscriptionResult:
        """
        Apply all Vietnamese optimizations to transcription.
        
        Args:
            result: Original transcription result
            
        Returns:
            Optimized transcription result
        """
        optimized_result = result
        
        # Apply post-correction for low confidence words
        if self.enable_post_correction and self.post_corrector:
            optimized_result = self.post_corrector.correct_transcription(optimized_result)
        
        # Validate diacritics
        if self.enable_diacritic_validation and self.diacritic_validator:
            is_valid, accuracy = self.diacritic_validator.validate_text(
                optimized_result.text
            )
            
            logger.info(f"Diacritic accuracy: {accuracy:.2%}")
            
            if not is_valid:
                logger.warning(
                    f"Diacritic accuracy below threshold: {accuracy:.2%} < 98%"
                )
        
        return optimized_result
    
    def get_optimization_info(self) -> Dict:
        """
        Get information about optimizer configuration.
        
        Returns:
            Dictionary with optimizer info
        """
        info = {
            'post_correction_enabled': self.enable_post_correction,
            'diacritic_validation_enabled': self.enable_diacritic_validation,
            'has_vocabulary': self.vocabulary_processor is not None
        }
        
        if self.vocabulary_processor:
            info['vocabulary_info'] = self.vocabulary_processor.get_vocabulary_info()
        
        if self.post_corrector:
            info['confidence_threshold'] = self.post_corrector.confidence_threshold
        
        return info


def create_vietnamese_optimizer(
    vocabulary_file: Optional[str] = None,
    confidence_threshold: float = 0.7
) -> VietnameseSTTOptimizer:
    """
    Create a Vietnamese STT optimizer with default settings.
    
    Args:
        vocabulary_file: Path to Vietnamese vocabulary file
        confidence_threshold: Confidence threshold for post-correction
        
    Returns:
        VietnameseSTTOptimizer instance
    """
    return VietnameseSTTOptimizer(
        vocabulary_file=vocabulary_file,
        confidence_threshold=confidence_threshold,
        enable_post_correction=True,
        enable_diacritic_validation=True
    )
